"""
Runtime Model Inference Library

This library provides a unified interface for loading Runtime Transformer models
and performing inference on raw training examples. It handles all data transformations
(standard, ablation, shuffled ablation) automatically based on the model's configuration.

Usage:
    from runtime_inference import RuntimeModelInference, load_training_examples
    
    # Load training examples
    examples = load_training_examples(
        splits_dir="path/to/training_splits",
        max_examples=100
    )
    
    # Initialize with checkpoint and config
    inference = RuntimeModelInference(
        checkpoint_path="path/to/checkpoint.pt",
        config_path="path/to/config.yaml",
        pace_lookup_path="path/to/pace_lookup.pickle",
        device="cpu"  # or "cuda" or "mps"
    )
    
    # Predict from raw training example
    result = inference.predict_from_raw_example(examples[0])
    
    # result contains:
    # - 'probabilities': numpy array of probabilities over all bins
    # - 'pace_values': numpy array of median pace values for each bin
    # - 'weighted_mean': predicted mean pace
    # - 'weighted_median': predicted median pace
    # - 'mode_pace': predicted mode pace
    # - 'mode_bin_idx': index of most likely bin

Data Loading Helpers:
    - load_training_examples(): Load training examples from split files
    - load_runners_from_splits(): Load all runners from split files
    - get_all_training_examples_from_runners(): Extract examples from runners
"""

import os
import pickle
import gzip
import math
import torch
import torch.nn as nn
import numpy as np
import yaml
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


# --- MODEL ARCHITECTURE (must match trainers) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RunTimeTransformer(nn.Module):
    def __init__(self, vocab_size, num_pace_bins, config):
        super().__init__()
        m = config['model']
        self.d_model = m['d_model']
        self.embedding = nn.Embedding(vocab_size, m['d_model'], padding_idx=0)
        
        # Match positional encoding to the maximum input length we expect
        data_cfg = config.get('data', {}) if isinstance(config.get('data'), dict) else {}
        stride = int(data_cfg.get('block_stride', 11))
        ablation_drop_week_deltas = bool(data_cfg.get('ablation_drop_week_deltas', False))
        ablation_last_age_front = bool(data_cfg.get('ablation_last_age_front', False))
        ablation_out_stride = int(data_cfg.get('ablation_out_stride', 8))
        swap = bool(data_cfg.get('swap_pace_time_tokens', True))
        drop_k = int(data_cfg.get('drop_final_time_tokens', 2 if swap else 0))
        max_len = int(m.get('max_seq_length', 512))

        if ablation_drop_week_deltas and ablation_last_age_front:
            if 'max_races_to_consider' in m:
                n_races = int(m['max_races_to_consider'])
                max_len = 1 + (ablation_out_stride * (n_races - 1)) + 7  # 1 + 8*(n-1) + 7
        elif 'max_races_to_consider' in m:
            max_len = int(m['max_races_to_consider']) * stride - 1 - drop_k

        self.pos_encoder = PositionalEncoding(m['d_model'], max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m['d_model'],
            nhead=m['nhead'],
            dim_feedforward=m['dim_feedforward'],
            dropout=m['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m['num_layers'])
        try:
            self.transformer.enable_nested_tensor = False
        except Exception:
            pass
        
        self.output_head = nn.Linear(m['d_model'], num_pace_bins)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.constant_(self.output_head.bias, 0)
        
    def forward(self, x, padding_mask):
        # Generate causal mask
        sz = x.size(1)
        causal_mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool().to(x.device)
        
        # Scale embedding by sqrt(d_model)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # transformer_output: [batch, seq_len, d_model]
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # We only care about the prediction from the LAST non-padding token
        valid_lens = (~padding_mask).sum(dim=1) - 1
        last_outputs = x[torch.arange(x.size(0)), valid_lens]
        
        logits = self.output_head(last_outputs)
        return logits


# --- SEQUENCE TRANSFORMATION FUNCTIONS ---
def swap_pace_time_and_drop_final(seq: list, block_stride: int = 11, drop_final_time_tokens: int = 2) -> list:
    """Swap pace and time-delta tokens within each block to improve causal coherence.

    Original (legacy) block order (stride=11):
      <features_i> <delta_next_i> <delta_final_i> <pace_i>
      e.g. [age, gender, cond, humidity, temp, feels, wind, dist] [weeks_to_next] [weeks_to_final] [pace]
      Final block: [features] [delta_0] [delta_0] [pace_final]

    Swapped (paper) block order:
      <features_i> <pace_i> <delta_next_i> <delta_final_i>
      Pace for race i now precedes the cadence gap to race i+1, so the model
      observes performance before the temporal context of the next event.
      Final block's trailing [delta_0, delta_0] are redundant and dropped
      (drop_final_time_tokens=2), ending the sequence on the target pace.
    """
    stride = block_stride
    if not seq or len(seq) < stride or (len(seq) % stride) != 0:
        return seq

    nblocks = len(seq) // stride
    out: list = []

    for bi in range(nblocks):
        block = seq[bi * stride:(bi + 1) * stride]
        # Based on real data: first 8 are features (including age), next 2 are time deltas, last is pace.
        feat = block[:8]  # [age, gender, conditions, humidity, temp, feels_like, wind, distance]
        t1, t2 = block[8], block[9]
        pace = block[10]

        if bi == nblocks - 1 and drop_final_time_tokens:
            # Only drop if they look like the expected redundant tokens; otherwise keep to avoid data corruption.
            should_drop = True
            if drop_final_time_tokens >= 1 and str(t2) != "week_delta_0":
                should_drop = False
            if drop_final_time_tokens >= 2 and str(t1) != "week_delta_0":
                should_drop = False

            if should_drop:
                # Final block: [8 features][pace] (drops the 2 time tokens)
                out.extend(feat)
                out.append(pace)
                continue

        # Historical blocks: [8 features][pace][time_1][time_2]
        out.extend(feat)
        out.append(pace)
        out.append(t1)
        out.append(t2)

    return out


def apply_ablation(seq: list, block_stride: int = 11) -> list:
    """
    Apply ablation transformations:
    1. Drop all week_delta_* tokens entirely
    2. Keep only the last age token and move it to the front
    3. Each race block (except final): 8 tokens [gender, conditions, humidity, temp, feels_like, wind, distance, pace]
    4. Final block: 7 tokens [gender, conditions, humidity, temp, feels_like, wind, distance] (no pace, we predict it)
    5. Final sequence: [age_last] + [8 tokens per race] * (n-1) + [7 tokens for final block] = 1 + 8*(n-1) + 7 = 8*n
    """
    stride = block_stride
    if not seq or len(seq) < stride or (len(seq) % stride) != 0:
        return seq

    nblocks = len(seq) // stride
    out: list = []
    last_age_token = None
    
    for bi in range(nblocks):
        block = seq[bi * stride:(bi + 1) * stride]
        # Original structure: [age, gender, conditions, humidity, temp, feels_like, wind, distance, week_delta_next, week_delta_final, pace]
        age_token = block[0]
        gender = block[1]
        conditions = block[2]
        humidity = block[3]
        temp = block[4]
        feels_like = block[5]
        wind = block[6]
        distance = block[7]
        # Skip week_delta_next (block[8]) and week_delta_final (block[9])
        pace = block[10]
        
        last_age_token = age_token
        
        if bi == nblocks - 1:
            # Final block: 7 tokens without pace
            out.extend([gender, conditions, humidity, temp, feels_like, wind, distance])
        else:
            # All other blocks: 8 tokens with pace
            out.extend([gender, conditions, humidity, temp, feels_like, wind, distance, pace])
    
    # Move the last age token to the front
    if last_age_token is not None:
        out = [last_age_token] + out
    
    return out


def apply_ablation_shuffled(seq: list, block_stride: int = 11, seed: Optional[int] = None) -> list:
    """
    Apply ablation transformations with shuffled historical races:
    1. Drop all week_delta_* tokens entirely
    2. Keep only the age token from the final race and move it to the front
    3. Shuffle all historical races (but keep final race fixed at the end)
    4. Each historical race block: 8 tokens [gender, conditions, humidity, temp, feels_like, wind, distance, pace]
    5. Final block: 7 tokens [gender, conditions, humidity, temp, feels_like, wind, distance] (no pace, we predict it)
    6. Final sequence: [age_at_final] + [shuffled 8-token blocks] + [7-token final block]
    """
    if seed is not None:
        random.seed(seed)
    
    stride = block_stride
    if not seq or len(seq) < stride or (len(seq) % stride) != 0:
        return seq

    nblocks = len(seq) // stride
    if nblocks < 2:
        # Need at least 2 blocks (one historical + final)
        return apply_ablation(seq, block_stride)
    
    # Extract blocks
    blocks = []
    final_age_token = None
    
    for bi in range(nblocks):
        block = seq[bi * stride:(bi + 1) * stride]
        age_token = block[0]
        gender = block[1]
        conditions = block[2]
        humidity = block[3]
        temp = block[4]
        feels_like = block[5]
        wind = block[6]
        distance = block[7]
        pace = block[10]
        
        if bi == nblocks - 1:
            # Final block: store age token and create 7-token block (no pace)
            final_age_token = age_token
            final_block = [gender, conditions, humidity, temp, feels_like, wind, distance]
        else:
            # Historical blocks: 8 tokens with pace
            blocks.append([gender, conditions, humidity, temp, feels_like, wind, distance, pace])
    
    # Shuffle historical blocks (but keep final block fixed)
    random.shuffle(blocks)
    
    # Build output sequence: [age_at_final] + [shuffled historical blocks] + [final block]
    out = []
    if final_age_token is not None:
        out.append(final_age_token)
    
    # Add all shuffled historical blocks (flattened)
    for block in blocks:
        out.extend(block)
    
    # Add final block (7 tokens, no pace)
    out.extend(final_block)
    
    return out


def transform_sequence(seq: list, config: dict, block_stride: int = 11, shuffle_seed: Optional[int] = None) -> list:
    """
    Apply the appropriate sequence transformation based on config.
    
    Args:
        seq: Raw sequence from training example (unpadded_example_sequence)
        config: Model configuration dict
        block_stride: Number of tokens per race block (default 11)
        shuffle_seed: Random seed for shuffling (only used if shuffling enabled)
    
    Returns:
        Transformed sequence ready for tokenization
    """
    data_cfg = config.get('data', {}) if isinstance(config.get('data'), dict) else {}
    
    ablation_drop_week_deltas = bool(data_cfg.get('ablation_drop_week_deltas', False))
    ablation_last_age_front = bool(data_cfg.get('ablation_last_age_front', False))
    ablation_shuffle_races = bool(data_cfg.get('ablation_shuffle_races', False))
    swap_pace_time = bool(data_cfg.get('swap_pace_time_tokens', True))
    drop_final_time_tokens = int(data_cfg.get('drop_final_time_tokens', 2 if swap_pace_time else 0))
    
    if ablation_drop_week_deltas and ablation_last_age_front:
        if ablation_shuffle_races:
            return apply_ablation_shuffled(seq, block_stride, seed=shuffle_seed)
        else:
            return apply_ablation(seq, block_stride)
    elif swap_pace_time:
        return swap_pace_time_and_drop_final(seq, block_stride, drop_final_time_tokens)
    else:
        # No transformation - return as-is (but still need to handle final pace token)
        return seq


def prepare_inference_sequence(seq: list, vocab: dict, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize and pad a transformed sequence for inference.
    
    Args:
        seq: Transformed sequence (list of tokens)
        vocab: Vocabulary mapping tokens to indices
        max_len: Maximum sequence length
    
    Returns:
        (input_ids, padding_mask) as tensors
    """
    # Next-token prediction: input is all tokens except the last one (pace token)
    input_tokens = seq[:-1]
    
    # Tokenize
    input_ids = [vocab.get(t, vocab.get('<unk>', 1)) for t in input_tokens]
    
    # Truncate if necessary (keep the end of the sequence)
    if len(input_ids) > max_len:
        input_ids = input_ids[-max_len:]
    
    # Pad at the back
    padding_len = max_len - len(input_ids)
    mask = [False] * len(input_ids) + [True] * padding_len
    input_ids = input_ids + [vocab.get('<pad>', 0)] * padding_len
    
    return torch.tensor([input_ids], dtype=torch.long), torch.tensor([mask], dtype=torch.bool)


# --- MAIN INFERENCE CLASS ---
class RuntimeModelInference:
    """
    Main class for Runtime model inference.
    
    Handles model loading, sequence transformation, and prediction.
    """
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        pace_lookup_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        enable_mps_fallback: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            config_path: Path to config YAML file (if None, tries to load from checkpoint directory)
            pace_lookup_path: Path to pace_lookup.pickle (if None, tries to infer from config)
            device: Device to run on ('cpu', 'cuda', 'mps', or torch.device). If None, auto-detects.
            enable_mps_fallback: Enable CPU fallback for MPS unsupported operations
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Device setup
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        if isinstance(device, str):
            # Handle CUDA device strings like "cuda:3"
            if device.startswith('cuda:'):
                device_id = int(device.split(':')[1])
                # Check if CUDA is available and device exists
                if not torch.cuda.is_available():
                    print(f"⚠️  Warning: CUDA not available, falling back to CPU")
                    self.device = torch.device('cpu')
                elif device_id >= torch.cuda.device_count():
                    print(f"⚠️  Warning: CUDA device {device_id} not available, using device 0")
                    self.device = torch.device('cuda:0')
                else:
                    self.device = torch.device(device)
            else:
                self.device = torch.device(device)
        else:
            self.device = device
        
        if enable_mps_fallback and self.device.type == 'mps':
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        # Load checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        # Always load to CPU first, then move to device (handles subprocess CUDA initialization issues)
        # Suppress FutureWarning about weights_only (we trust our own checkpoints)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load config
        if config_path is None:
            # Try to find config in checkpoint directory
            config_path = checkpoint_path.parent / "config_copy.yaml"
            if not config_path.exists():
                config_path = checkpoint_path.parent / "config_resolved.yaml"
        
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            else:
                # Try to get config from checkpoint
                if 'config' in checkpoint:
                    self.config = checkpoint['config']
                else:
                    raise ValueError(f"Config not found. Please provide config_path or ensure config is in checkpoint.")
        else:
            if 'config' in checkpoint:
                self.config = checkpoint['config']
            else:
                raise ValueError("Config not found in checkpoint and no config_path provided.")
        
        # Load vocab
        if 'vocab' in checkpoint:
            self.vocab = checkpoint['vocab']
        else:
            raise ValueError("Vocabulary not found in checkpoint.")
        
        # Load pace bins
        if pace_lookup_path is None:
            # Try to infer from config
            if 'data' in self.config and 'pace_lookup' in self.config['data']:
                pace_lookup_path = Path(self.config['data']['pace_lookup'])
                if not pace_lookup_path.is_absolute():
                    # Try relative to config file
                    pace_lookup_path = config_path.parent / pace_lookup_path
        
        if pace_lookup_path is not None:
            pace_lookup_path = Path(pace_lookup_path)
            if pace_lookup_path.exists():
                with open(pace_lookup_path, 'rb') as f:
                    pace_data = pickle.load(f)
                
                # Convert to list format
                if isinstance(pace_data, dict):
                    pace_bins = []
                    for token, info in pace_data.items():
                        if not isinstance(info, dict):
                            continue
                        median_val = info.get('median_pace', info.get('median', 0))
                        pace_bins.append({
                            'token': token,
                            'start': float(info['start']),
                            'end': float(info['end']),
                            'median': float(median_val)
                        })
                    self.pace_bins = sorted(pace_bins, key=lambda x: x['median'])
                else:
                    self.pace_bins = pace_data
            else:
                raise FileNotFoundError(f"Pace lookup not found: {pace_lookup_path}")
        else:
            raise ValueError("pace_lookup_path not provided and cannot be inferred from config.")
        
        # Determine max sequence length
        data_cfg = self.config.get('data', {}) if isinstance(self.config.get('data'), dict) else {}
        self.block_stride = int(data_cfg.get('block_stride', 11))
        
        ablation_drop_week_deltas = bool(data_cfg.get('ablation_drop_week_deltas', False))
        ablation_last_age_front = bool(data_cfg.get('ablation_last_age_front', False))
        ablation_out_stride = int(data_cfg.get('ablation_out_stride', 8))
        swap = bool(data_cfg.get('swap_pace_time_tokens', True))
        drop_k = int(data_cfg.get('drop_final_time_tokens', 2 if swap else 0))
        
        if 'max_races_to_consider' in self.config['model']:
            if ablation_drop_week_deltas and ablation_last_age_front:
                n_races = int(self.config['model']['max_races_to_consider'])
                self.max_len = 1 + (ablation_out_stride * (n_races - 1)) + 7
            else:
                self.max_len = int(self.config['model']['max_races_to_consider']) * self.block_stride - 1 - drop_k
        else:
            self.max_len = int(self.config['model'].get('max_seq_length', 512))
        
        # Load model
        print(f"Initializing model (vocab_size={len(self.vocab)}, num_pace_bins={len(self.pace_bins)})...")
        self.model = RunTimeTransformer(len(self.vocab), len(self.pace_bins), self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Pre-compute pace values for weighted predictions
        self.pace_values = torch.tensor([b['median'] for b in self.pace_bins], dtype=torch.float32).to(self.device)
        
        print(f"✓ Model loaded on {self.device}")
        print(f"  Max sequence length: {self.max_len}")
        print(f"  Block stride: {self.block_stride}")
        print(f"  Pace bins: {len(self.pace_bins)}")
    
    def predict_from_raw_example(
        self,
        raw_example,
        shuffle_seed: Optional[int] = None
    ) -> Dict:
        """
        Predict pace distribution from a raw training example.
        
        Args:
            raw_example: TrainingExample object or dict with 'unpadded_example_sequence' attribute/key
            shuffle_seed: Random seed for shuffling (only used if shuffling enabled)
        
        Returns:
            Dictionary with:
            - 'probabilities': numpy array of probabilities over all bins
            - 'pace_values': numpy array of median pace values for each bin
            - 'weighted_mean': predicted mean pace (seconds)
            - 'weighted_median': predicted median pace (seconds)
            - 'mode_pace': predicted mode pace (seconds)
            - 'mode_bin_idx': index of most likely bin
            - 'logits': raw logits from model
        """
        # Extract sequence
        if hasattr(raw_example, 'unpadded_example_sequence'):
            seq = raw_example.unpadded_example_sequence
        elif isinstance(raw_example, dict) and 'unpadded_example_sequence' in raw_example:
            seq = raw_example['unpadded_example_sequence']
        elif isinstance(raw_example, list):
            seq = raw_example
        else:
            raise ValueError("raw_example must be a TrainingExample, dict with 'unpadded_example_sequence', or list")
        
        # Transform sequence based on config
        transformed_seq = transform_sequence(seq, self.config, self.block_stride, shuffle_seed)
        
        # Prepare for inference
        input_ids, padding_mask = prepare_inference_sequence(transformed_seq, self.vocab, self.max_len)
        input_ids = input_ids.to(self.device)
        padding_mask = padding_mask.to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_ids, padding_mask)
            probs = torch.softmax(logits, dim=1)[0]  # Get first (and only) batch item
        
        # Convert to numpy
        probabilities = probs.cpu().numpy()
        pace_values = self.pace_values.cpu().numpy()
        
        # Normalize probabilities to ensure they sum to 1 (probability mass)
        # This matches the pattern used in Inspect_Model_Outputs and Examine_Distribution_Quantile_Predictions
        p_mass = probabilities / probabilities.sum() if probabilities.sum() > 0 else probabilities
        
        # Compute statistics
        # Weighted mean: sum of (probability_mass * pace_value) for each bin
        weighted_mean = float(np.sum(p_mass * pace_values))
        
        # Mode: bin with highest probability
        mode_bin_idx = int(np.argmax(probabilities))
        mode_pace = float(self.pace_bins[mode_bin_idx]['median'])
        
        # Compute median: find where cumulative probability mass reaches 0.5
        # This correctly accounts for the probability distribution over bins
        cumsum = np.cumsum(p_mass)
        median_idx = min(np.searchsorted(cumsum, 0.5), len(self.pace_bins) - 1)
        weighted_median = float(self.pace_bins[median_idx]['median'])
        
        return {
            'probabilities': probabilities,
            'pace_values': pace_values,
            'weighted_mean': weighted_mean,
            'weighted_median': weighted_median,
            'mode_pace': mode_pace,
            'mode_bin_idx': mode_bin_idx,
            'logits': logits[0].cpu().numpy()
        }
    
    def predict_batch(self, raw_examples: List, shuffle_seed: Optional[int] = None) -> List[Dict]:
        """
        Predict pace distributions for a batch of raw training examples.
        Much faster than calling predict_from_raw_example multiple times.
        
        Args:
            raw_examples: List of TrainingExample objects or dicts with 'unpadded_example_sequence'
            shuffle_seed: Random seed for shuffling (only used if shuffling enabled)
        
        Returns:
            List of dictionaries (same format as predict_from_raw_example)
        """
        if not raw_examples:
            return []
        
        # Extract and transform sequences
        transformed_seqs = []
        for raw_example in raw_examples:
            if hasattr(raw_example, 'unpadded_example_sequence'):
                seq = raw_example.unpadded_example_sequence
            elif isinstance(raw_example, dict) and 'unpadded_example_sequence' in raw_example:
                seq = raw_example['unpadded_example_sequence']
            elif isinstance(raw_example, list):
                seq = raw_example
            else:
                raise ValueError("raw_example must be a TrainingExample, dict with 'unpadded_example_sequence', or list")
            
            transformed_seq = transform_sequence(seq, self.config, self.block_stride, shuffle_seed)
            transformed_seqs.append(transformed_seq)
        
        # Prepare batch: tokenize and pad all sequences
        batch_input_ids = []
        batch_masks = []
        for seq in transformed_seqs:
            input_tokens = seq[:-1]  # Next-token prediction: exclude last pace token
            input_ids = [self.vocab.get(t, self.vocab.get('<unk>', 1)) for t in input_tokens]
            
            # Truncate if necessary
            if len(input_ids) > self.max_len:
                input_ids = input_ids[-self.max_len:]
            
            # Pad
            padding_len = self.max_len - len(input_ids)
            mask = [False] * len(input_ids) + [True] * padding_len
            input_ids = input_ids + [self.vocab.get('<pad>', 0)] * padding_len
            
            batch_input_ids.append(input_ids)
            batch_masks.append(mask)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
        padding_mask_tensor = torch.tensor(batch_masks, dtype=torch.bool).to(self.device)
        
        # Run batched inference
        with torch.no_grad():
            logits = self.model(input_ids_tensor, padding_mask_tensor)
            probs = torch.softmax(logits, dim=1)  # [batch_size, num_bins]
        
        # Convert to numpy and process each example
        probs_np = probs.cpu().numpy()
        pace_values = self.pace_values.cpu().numpy()
        results = []
        
        for i in range(len(raw_examples)):
            probabilities = probs_np[i]
            p_mass = probabilities / probabilities.sum() if probabilities.sum() > 0 else probabilities
            
            # Compute statistics
            weighted_mean = float(np.sum(p_mass * pace_values))
            mode_bin_idx = int(np.argmax(probabilities))
            mode_pace = float(self.pace_bins[mode_bin_idx]['median'])
            
            cumsum = np.cumsum(p_mass)
            median_idx = min(np.searchsorted(cumsum, 0.5), len(self.pace_bins) - 1)
            weighted_median = float(self.pace_bins[median_idx]['median'])
            
            results.append({
                'probabilities': probabilities,
                'pace_values': pace_values,
                'weighted_mean': weighted_mean,
                'weighted_median': weighted_median,
                'mode_pace': mode_pace,
                'mode_bin_idx': mode_bin_idx,
                'logits': logits[i].cpu().numpy()
            })
        
        return results


# --- DATA LOADING HELPER FUNCTIONS ---
@dataclass
class TrainingExample:
    """Training example dataclass matching the pipeline structure."""
    unpadded_example_sequence: list 
    actual_pace_seconds: float
    raw_pace_data: list

@dataclass
class RunnerForTraining:
    """Runner dataclass matching the pipeline structure."""
    name_gender_dedup_int: tuple  # (first_name, last_name, gender, dedup_int)
    training_examples: list
    split_assignment: int


def load_runners_from_splits(
    max_runners: Optional[int] = None,
    splits_dir: Union[str, Path] = None,
    num_files: Optional[int] = None,
    progress_interval: int = 10,
    glob_pattern: Union[str, Path] = None,
) -> List[RunnerForTraining]:
    """
    Load all runners from training split files.
    
    Args:
        max_runners: Optional cap on total number of runners to load
        splits_dir: Directory containing .pkl.gz training split files (deprecated, use glob_pattern)
        num_files: Maximum number of files to load (None = all files)
        progress_interval: Print progress every N files
        glob_pattern: Glob pattern to match files. Can be:
            - A full glob path like "path/to/splits/*.pkl.gz"
            - A directory path (will use "*.pkl.gz" pattern)
            - If None and splits_dir provided, uses splits_dir with "*.pkl.gz"
    
    Returns:
        List of RunnerForTraining objects
    """
    # Support both glob_pattern (new) and splits_dir (legacy)
    if glob_pattern is None:
        if splits_dir is None:
            raise ValueError("Either glob_pattern or splits_dir must be provided")
        splits_dir = Path(splits_dir)
        if not splits_dir.exists():
            raise FileNotFoundError(f"Training splits directory not found: {splits_dir}")
        split_files = sorted(list(splits_dir.glob("*.pkl.gz")))
    else:
        # Use glob_pattern directly - user specifies the exact pattern
        glob_str = str(glob_pattern)
        glob_path = Path(glob_str)
        
        # Extract parent directory and pattern
        parent_dir = glob_path.parent
        pattern = glob_path.name
        
        # Handle case where parent is root or empty (pattern like "*.pkl.gz")
        if str(parent_dir) == "." or str(parent_dir) == "":
            parent_dir = Path(".")
        
        if not parent_dir.exists():
            raise FileNotFoundError(f"Directory not found: {parent_dir}")
        
        # Use glob to find matching files with the exact pattern provided
        split_files = sorted(list(parent_dir.glob(pattern)))
        if not split_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern} in directory: {parent_dir}")
    
    if num_files is not None:
        split_files = split_files[:num_files]
    
    all_runners = []
    
    # Iterate over files and load runners, respecting max_runners if provided
    for file_idx, fpath in enumerate(split_files):
        if progress_interval > 0 and (file_idx + 1) % progress_interval == 0:
            print(f"  Processed {file_idx + 1}/{len(split_files)} files...")
        
        # If we've already reached the requested number of runners, stop early
        if max_runners is not None and len(all_runners) >= max_runners:
            break

        with gzip.open(fpath, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    for runner in batch:
                        all_runners.append(runner)
                        # Stop if we've reached the cap
                        if max_runners is not None and len(all_runners) >= max_runners:
                            break
                    # If we've hit the limit inside this batch, break out of the read loop
                    if max_runners is not None and len(all_runners) >= max_runners:
                        break
                except EOFError:
                    break
    
    print(f"✓ Loaded {len(all_runners)} runners from {len(split_files)} files")
    return all_runners


def load_training_examples(
    max_examples: int = 100,
    splits_dir: Union[str, Path] = None,
    min_sequence_length: int = 22,  # At least 1 prior race + 1 final race = 22 tokens (2 * 11)
    num_files: Optional[int] = None,
    random_sample: bool = False,
    seed: Optional[int] = None,
    glob_pattern: Union[str, Path] = None
) -> List[TrainingExample]:
    """
    Load training examples from split files.
    
    Args:
        max_examples: Maximum number of examples to load
        splits_dir: Directory containing .pkl.gz training split files (deprecated, use glob_pattern)
        min_sequence_length: Minimum sequence length to include
        num_files: Maximum number of files to load (None = all files)
        random_sample: If True, randomly sample examples; if False, take first N
        seed: Random seed for sampling (only used if random_sample=True)
        glob_pattern: Glob pattern to match files. Can be:
            - A full glob path like "path/to/splits/*.pkl.gz"
            - A directory path (will use "*.pkl.gz" pattern)
            - If None and splits_dir provided, uses splits_dir with "*.pkl.gz"
    
    Returns:
        List of TrainingExample objects
    """
    # Support both glob_pattern (new) and splits_dir (legacy)
    if glob_pattern is None:
        if splits_dir is None:
            raise ValueError("Either glob_pattern or splits_dir must be provided")
        splits_dir = Path(splits_dir)
        if not splits_dir.exists():
            raise FileNotFoundError(f"Training splits directory not found: {splits_dir}")
        split_files = sorted(list(splits_dir.glob("*.pkl.gz")))
    else:
        # Use glob_pattern directly - user specifies the exact pattern
        glob_str = str(glob_pattern)
        glob_path = Path(glob_str)
        
        # Extract parent directory and pattern
        parent_dir = glob_path.parent
        pattern = glob_path.name
        
        # Handle case where parent is root or empty (pattern like "*.pkl.gz")
        if str(parent_dir) == "." or str(parent_dir) == "":
            parent_dir = Path(".")
        
        if not parent_dir.exists():
            raise FileNotFoundError(f"Directory not found: {parent_dir}")
        
        # Use glob to find matching files with the exact pattern provided
        split_files = sorted(list(parent_dir.glob(pattern)))
        if not split_files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern} in directory: {parent_dir}")
    
    if num_files is not None:
        split_files = split_files[:num_files]
    
    examples = []
    
    if random_sample and seed is not None:
        random.seed(seed)
    
    for fpath in split_files:
        if len(examples) >= max_examples:
            break
        
        with gzip.open(fpath, 'rb') as f:
            while len(examples) < max_examples:
                try:
                    batch = pickle.load(f)
                    for runner in batch:
                        for ex in runner.training_examples:
                            if len(ex.unpadded_example_sequence) >= min_sequence_length:
                                examples.append(ex)
                                if len(examples) >= max_examples:
                                    break
                        if len(examples) >= max_examples:
                            break
                except EOFError:
                    break
        if len(examples) >= max_examples:
            break
    
    # Random sampling if requested
    if random_sample and len(examples) > max_examples:
        examples = random.sample(examples, max_examples)
    elif len(examples) > max_examples:
        examples = examples[:max_examples]
    
    print(f"✓ Loaded {len(examples)} training examples")
    return examples


def get_all_training_examples_from_runners(
    runners: List[RunnerForTraining],
    min_sequence_length: int = 22  # At least 1 prior race + 1 final race = 22 tokens (2 * 11)
) -> List[TrainingExample]:
    """
    Extract all training examples from a list of runners.
    
    Args:
        runners: List of RunnerForTraining objects
        min_sequence_length: Minimum sequence length to include
    
    Returns:
        List of TrainingExample objects
    """
    examples = []
    for runner in runners:
        for ex in runner.training_examples:
            if len(ex.unpadded_example_sequence) >= min_sequence_length:
                examples.append(ex)
    
    print(f"✓ Extracted {len(examples)} training examples from {len(runners)} runners")
    return examples

