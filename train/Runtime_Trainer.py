import os
import sys
import gzip
import pickle
import math
import bisect
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import wandb
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from scipy.stats import kstest
from datetime import datetime

# --- CONFIGURATION LOADING ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- DATA STRUCTURES (Matching Pipeline) ---
@dataclass
class TrainingExample:
    unpadded_example_sequence: list 
    actual_pace_seconds: float
    raw_pace_data: list

@dataclass
class RunnerForTraining:
    name_gender_dedup_int: tuple
    training_examples: list
    split_assignment: int

# --- GAUSSIAN SMOOTHING ---
def get_gaussian_targets(actual_pace, pace_bins, sigma, cache=None, cache_maxsize=None):
    """
    Implements Seconds-Aware Gaussian Smoothing using the Error Function (erf)
    as described in the RunTime White Paper.
    Uses a fixed sigma value for all bins.
    
    Args:
        actual_pace: The actual pace value (seconds)
        pace_bins: List of bin dicts with 'start', 'end'
        sigma: Fixed sigma value for all bins (if 0, uses one-hot encoding = cross-entropy)
        cache: Optional OrderedDict for caching results (key: actual_pace)
        cache_maxsize: Optional max cache size for LRU eviction
    """
    # Check cache first (for fixed sigma, key is just actual_pace)
    if cache is not None:
        cache_key = actual_pace
        if cache_key in cache:
            # Move to end (most recently used) - O(1) operation
            cache.move_to_end(cache_key)
            return cache[cache_key]
    
    # Handle sigma = 0 (cross-entropy / one-hot encoding)
    if sigma == 0 or abs(sigma) < 1e-10:
        # Find the bin containing actual_pace and create one-hot target
        targets = np.zeros(len(pace_bins))
        for idx, b in enumerate(pace_bins):
            if b['start'] <= actual_pace < b['end']:
                targets[idx] = 1.0
                break
        else:
            # If actual_pace is outside all bins, use nearest bin
            medians = np.array([b['median'] for b in pace_bins])
            idx = np.argmin(np.abs(medians - actual_pace))
            targets[idx] = 1.0
        
        result = torch.tensor(targets, dtype=torch.float32)
        # Store in cache if provided
        if cache is not None:
            cache_key = actual_pace
            if cache_maxsize is not None and len(cache) >= cache_maxsize:
                cache.popitem(last=False)
            cache[cache_key] = result
            cache.move_to_end(cache_key)
        return result
    
    targets = []
    # pace_bins is expected to be a list of dicts/tuples with 'start', 'end'
    # For each bin, calculate the integral of the Gaussian kernel
    sqrt2 = np.sqrt(2.0)
    for b in pace_bins:
        start = b['start']
        end = b['end']
        
        # math.erf expects a float
        val = 0.5 * (math.erf((end - actual_pace) / (sigma * sqrt2)) - 
                    math.erf((start - actual_pace) / (sigma * sqrt2)))
        targets.append(max(0, val))
    
    # Normalize to ensure it's a probability distribution
    targets = np.array(targets)
    sum_t = targets.sum()
    if sum_t > 0:
        targets = targets / sum_t
    else:
        # Fallback to one-hot if sigma is too small or something went wrong
        # Find nearest bin using hash table (if available) or binary search
        # Note: This fallback is rarely used, but we optimize it anyway
        if hasattr(pace_bins, '__len__') and len(pace_bins) > 0:
            # Use hash table if pace_to_bin_idx is available (passed as optional param)
            # Otherwise use binary search on bin_ends
            # For now, just use the first bin as a simple fallback
            idx = 0
        else:
            idx = 0
        targets = np.zeros(len(pace_bins))
        targets[idx] = 1.0
    
    result = torch.tensor(targets, dtype=torch.float32)
    
    # Store in cache if provided
    if cache is not None:
        cache_key = actual_pace
        # LRU eviction if cache is full (OrderedDict maintains insertion order)
        if cache_maxsize is not None and len(cache) >= cache_maxsize:
            # Remove least recently used (first item) - O(1) operation
            cache.popitem(last=False)
        cache[cache_key] = result
        # Move to end (most recently used) - O(1) operation
        cache.move_to_end(cache_key)
    
    return result


def get_gaussian_targets_adaptive(actual_pace, pace_bins, sigma_lookup_by_bin_idx, target_bin_idx, cache=None, cache_maxsize=None):
    """
    Implements Seconds-Aware Gaussian Smoothing with bin-size-adjusted sigma.
    
    The Gaussian is centered on the actual_pace (raw value) for precision, but uses
    the target bin's width to determine the adaptive sigma.
    
    Args:
        actual_pace: The actual pace value (seconds) - center of the Gaussian
        pace_bins: List of bin dicts with 'start', 'end', 'median'
        sigma_lookup_by_bin_idx: Precomputed list/array of sigmas, one per bin index
        target_bin_idx: The bin index for the target (from pace_token_to_idx lookup)
        cache: Optional OrderedDict for caching results (key: (actual_pace, sigma))
        cache_maxsize: Optional max cache size for LRU eviction
    
    Returns:
        Tensor of soft targets (probability distribution over bins)
    """
    # Direct lookup of precomputed sigma for this bin
    sigma = sigma_lookup_by_bin_idx[target_bin_idx]
    
    # Handle sigma = 0 (cross-entropy / one-hot encoding)
    if sigma == 0 or abs(sigma) < 1e-10:
        # Create one-hot target at the target_bin_idx
        targets = np.zeros(len(pace_bins))
        targets[target_bin_idx] = 1.0
        
        result = torch.tensor(targets, dtype=torch.float32)
        # Store in cache if provided
        if cache is not None:
            cache_key = (actual_pace, sigma)
            if cache_maxsize is not None and len(cache) >= cache_maxsize:
                cache.popitem(last=False)
            cache[cache_key] = result
            cache.move_to_end(cache_key)
        return result
    
    # Check cache first (for adaptive sigma, key is (actual_pace, sigma))
    if cache is not None:
        cache_key = (actual_pace, sigma)
        if cache_key in cache:
            # Move to end (most recently used) - O(1) operation
            cache.move_to_end(cache_key)
            return cache[cache_key]
    
    # Now use the standard Gaussian smoothing with this adaptive sigma
    targets = []
    sqrt2 = np.sqrt(2.0)
    for b in pace_bins:
        start = b['start']
        end = b['end']
        
        # math.erf expects a float
        val = 0.5 * (math.erf((end - actual_pace) / (sigma * sqrt2)) - 
                    math.erf((start - actual_pace) / (sigma * sqrt2)))
        targets.append(max(0, val))
    
    # Normalize to ensure it's a probability distribution
    targets = np.array(targets)
    sum_t = targets.sum()
    if sum_t > 0:
        targets = targets / sum_t
    else:
        # Fallback to one-hot if sigma is too small or something went wrong
        targets = np.zeros(len(pace_bins))
        targets[target_bin_idx] = 1.0
    
    result = torch.tensor(targets, dtype=torch.float32)
    
    # Store in cache if provided
    if cache is not None:
        cache_key = (actual_pace, sigma)
        # LRU eviction if cache is full (OrderedDict maintains insertion order)
        if cache_maxsize is not None and len(cache) >= cache_maxsize:
            # Remove least recently used (first item) - O(1) operation
            cache.popitem(last=False)
        cache[cache_key] = result
        # Move to end (most recently used) - O(1) operation
        cache.move_to_end(cache_key)
    
    return result

# --- DATASET ---
class RunTimeDataset(Dataset):
    def __init__(self, runners, vocab, pace_bins, config):
        self.examples = []
        self.vocab = vocab
        self.pace_bins = pace_bins
        self.config = config
        
        # Token layout (per race) is a fixed stride in the grammar.
        # Default: 11 tokens per race, and we predict the final token (pace) so input length is (11*n - 1).
        #
        # Original (legacy) block order placed time tokens *before* pace:
        #   <features_i> <delta_next_i> <delta_final_i> <pace_i>
        # e.g.: [features R1] [weeks_to_R2] [weeks_to_final] [pace R1] [features R2] ...
        #        [features R_final] [delta_0] [delta_0] [pace_final]
        #
        # The swapped (paper) architecture moves pace *before* the time tokens so that
        # the model sees pace for race i before the cadence gap to race i+1, improving
        # causal coherence:
        #   <features_i> <pace_i> <delta_next_i> <delta_final_i>
        # The final block's trailing [delta_0, delta_0] become redundant and are dropped
        # (drop_final_time_tokens=2), so the sequence ends on the target pace.
        #
        # swap_pace_time_tokens defaults to True; set to False only to reproduce
        # the legacy ordering.
        data_cfg = config.get('data', {}) if isinstance(config.get('data'), dict) else {}
        self.block_stride = int(data_cfg.get('block_stride', 11))
        self.swap_pace_time = bool(data_cfg.get('swap_pace_time_tokens', True))
        # Drop the last K time tokens from the final block (the redundant delta_0 sentinels),
        # so the sequence ends with the pace token. Reduces input length by K: (11*n - 1 - K).
        self.drop_final_time_tokens = int(data_cfg.get('drop_final_time_tokens', 2 if self.swap_pace_time else 0))

        if 'max_races_to_consider' in config['model']:
            self.max_len = config['model']['max_races_to_consider'] * self.block_stride - 1 - self.drop_final_time_tokens
        else:
            # If max_seq_length is explicitly set, assume it already matches the desired token order.
            self.max_len = int(config['model']['max_seq_length'])
            
        # Gaussian smoothing configuration
        training_cfg = config.get('training', {})
        self.use_adaptive_sigma = bool(training_cfg.get('use_adaptive_sigma', False))
        
        # No need for pace_to_bin_idx lookup - we get the bin directly from target_token via pace_token_to_idx!
        
        if self.use_adaptive_sigma:
            # Adaptive sigma mode: sigma = sqrt(sigma_floor^2 + (k * w)^2) (where w is target bin width)
            self.sigma_floor = float(training_cfg.get('adaptive_sigma_floor', 3.5))
            self.k = float(training_cfg.get('adaptive_sigma_k', 1.5))
            self.sigma = None  # Not used in adaptive mode
            
            # Precompute sigma for each bin (by index) - performance optimization
            self.sigma_lookup_by_bin_idx = []
            for b in pace_bins:
                bin_width = b['end'] - b['start']
                sigma = np.sqrt(self.sigma_floor**2 + (self.k * bin_width)**2)
                self.sigma_lookup_by_bin_idx.append(sigma)
            
            # Initialize ERF cache for adaptive sigma (bounded to avoid memory issues)
            # Cache key: (actual_pace, sigma) -> full soft target vector
            # Max size: 10,000 entries (should cover most common pace values)
            # Use OrderedDict for O(1) LRU operations (move_to_end is O(1))
            self._erf_cache_maxsize = training_cfg.get('erf_cache_size', 10000)
            self._erf_cache = OrderedDict()
        else:
            # Fixed sigma mode: use single sigma value for all bins
            self.sigma = float(training_cfg.get('smoothing_sigma_seconds', 10.0))
            self.sigma_floor = None
            self.k = None
            self.sigma_lookup_by_bin_idx = None
            
            # Initialize ERF cache for fixed sigma (bounded to avoid memory issues)
            # Cache key: actual_pace -> full soft target vector (sigma is fixed)
            # Max size: 10,000 entries (should cover most common pace values)
            # Use OrderedDict for O(1) LRU operations (move_to_end is O(1))
            self._erf_cache_maxsize = training_cfg.get('erf_cache_size', 10000)
            self._erf_cache = OrderedDict()
        
        # Mapping pace tokens to their index in the pace_bins list
        self.pace_token_to_idx = {b['token']: i for i, b in enumerate(pace_bins)}
        
        for r in runners:
            for ex in r.training_examples:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        seq = ex.unpadded_example_sequence

        if self.swap_pace_time:
            seq = self._swap_pace_time_and_drop_final(seq)
        
        # Next-token prediction for the FINAL pace token
        # Input: All tokens except the last one
        # Target: The last one (pace token)
        
        input_tokens = seq[:-1]
        target_token = seq[-1]
        
        # Tokenize
        input_ids = [self.vocab.get(t, self.vocab['<unk>']) for t in input_tokens]
        
        # Truncate if necessary (keep the end of the sequence)
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            
        # Pad at the back
        padding_len = self.max_len - len(input_ids)
        mask = [False] * len(input_ids) + [True] * padding_len
        input_ids = input_ids + [self.vocab['<pad>']] * padding_len
        
        # Gaussian Soft Targets
        # Center Gaussian on the precise actual_pace_seconds (more accurate than bin median)
        actual_pace = ex.actual_pace_seconds
        
        if self.use_adaptive_sigma:
            # Get target bin index directly from the token (no lookup needed!)
            target_bin_idx = self.pace_token_to_idx.get(target_token, 0)
            
            # Use adaptive sigma based on target bin width (with caching)
            soft_target = get_gaussian_targets_adaptive(
                actual_pace, 
                self.pace_bins, 
                self.sigma_lookup_by_bin_idx,
                target_bin_idx,
                cache=self._erf_cache,
                cache_maxsize=self._erf_cache_maxsize
            )
        else:
            # Use fixed sigma for all bins (with caching)
            soft_target = get_gaussian_targets(
                actual_pace, 
                self.pace_bins, 
                self.sigma,
                cache=self._erf_cache,
                cache_maxsize=self._erf_cache_maxsize
            )
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'padding_mask': torch.tensor(mask, dtype=torch.bool),
            'soft_target': soft_target,
            'actual_pace': torch.tensor(actual_pace, dtype=torch.float32)
        }

    def _swap_pace_time_and_drop_final(self, seq: list) -> list:
        """
        Original per-race block (stride=11) observed in training splits:
          [8 feature tokens][time_1][time_2][pace]
        Desired per-race block:
          [8 feature tokens][pace][time_1][time_2]
        And for the *final* block only, drop the last K time tokens (default K=2) where they are always week_delta_0.
        """
        stride = self.block_stride
        if not seq or len(seq) < stride or (len(seq) % stride) != 0:
            return seq

        nblocks = len(seq) // stride
        out: list = []

        for bi in range(nblocks):
            block = seq[bi * stride:(bi + 1) * stride]
            # Based on real data: first 8 are features, next 2 are time deltas, last is pace.
            feat = block[:8]
            t1, t2 = block[8], block[9]
            pace = block[10]

            if bi == nblocks - 1 and self.drop_final_time_tokens:
                # Only drop if they look like the expected redundant tokens; otherwise keep to avoid data corruption.
                should_drop = True
                if self.drop_final_time_tokens >= 1 and str(t2) != "week_delta_0":
                    should_drop = False
                if self.drop_final_time_tokens >= 2 and str(t1) != "week_delta_0":
                    should_drop = False

                if should_drop:
                    out.extend(feat)
                    out.append(pace)
                    continue

            out.extend(feat)
            out.append(pace)
            out.append(t1)
            out.append(t2)

        return out

# --- MODEL ---
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
        # Match positional encoding to the maximum input length we expect.
        # If using max_races_to_consider, reflect the true input length given token order.
        data_cfg = config.get('data', {}) if isinstance(config.get('data'), dict) else {}
        stride = int(data_cfg.get('block_stride', 11))
        swap = bool(data_cfg.get('swap_pace_time_tokens', True))
        drop_k = int(data_cfg.get('drop_final_time_tokens', 2 if swap else 0))
        max_len = int(m.get('max_seq_length', 512))
        if 'max_races_to_consider' in m:
            max_len = int(m['max_races_to_consider']) * stride - 1 - drop_k

        self.pos_encoder = PositionalEncoding(m['d_model'], max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m['d_model'],
            nhead=m['nhead'],
            dim_feedforward=m['dim_feedforward'],
            dropout=m['dropout'],
            batch_first=True
        )
        # Using TransformerEncoder as a Decoder by applying a causal mask
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m['num_layers'])
        # MPS does not fully support the nested tensor fast path used internally by TransformerEncoder.
        # Disable it to avoid: aten::_nested_tensor_from_mask_left_aligned (NotImplementedError on MPS).
        # (This mirrors the workaround used in Inspect_Model_Outputs.ipynb.)
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
        # Dynamic indexing
        valid_lens = (~padding_mask).sum(dim=1) - 1
        last_outputs = x[torch.arange(x.size(0)), valid_lens]
        
        logits = self.output_head(last_outputs)
        return logits

# --- TRAINING LOOP ---
def _ddp_env():
    """
    Return (is_distributed, rank, world_size, local_rank) based on torchrun env vars.
    torchrun typically sets: RANK, WORLD_SIZE, LOCAL_RANK.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def _ddp_init_if_needed(device_type: str):
    is_dist, rank, world_size, local_rank = _ddp_env()
    if not is_dist:
        return False, rank, world_size, local_rank
    if device_type != "cuda":
        raise RuntimeError("DDP requested via torchrun env, but CUDA is not available.")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def _dist_is_init() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum_float(x: float, device: torch.device) -> float:
    if not _dist_is_init():
        return float(x)
    t = torch.tensor([float(x)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_sum_int(x: int, device: torch.device) -> int:
    if not _dist_is_init():
        return int(x)
    t = torch.tensor([int(x)], device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return int(t.item())


def train_model(config_path):
    config = load_config(config_path)

    # --- Reproducibility (best-effort) ---
    seed = int(config.get('training', {}).get('random_seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Path resolution ---
    # Treat relative config paths as relative to the config file directory.
    cfg_dir = Path(config_path).resolve().parent
    if isinstance(config.get('data'), dict):
        for k in ("pace_lookup", "splits_dir"):
            if k in config["data"] and isinstance(config["data"][k], str):
                p = Path(config["data"][k])
                if not p.is_absolute():
                    config["data"][k] = str((cfg_dir / p).resolve())
    
    # Device setup: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'

    # DDP init if launched via torchrun (2-GPU-per-trial)
    is_dist, rank, world_size, local_rank = _ddp_init_if_needed(device_type)
    is_rank0 = (rank == 0)

    # Bind device to local_rank when distributed; otherwise default device
    device = torch.device(f"cuda:{local_rank}" if (device_type == "cuda" and is_dist) else device_type)
    log = print if is_rank0 else (lambda *args, **kwargs: None)
    log(f"Using device: {device} (distributed={is_dist} rank={rank}/{world_size} local_rank={local_rank})")
    
    # --- DIRECTORY SETUP ---
    save_dir_base = config['logging'].get('save_dir', 'checkpoints')
    run_name = config['logging'].get('run_name', 'default_run')
    run_dir = os.path.join(save_dir_base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config file up front (rank0 only)
    if is_rank0:
        import shutil
        shutil.copy(config_path, os.path.join(run_dir, "config_copy.yaml"))
        with open(os.path.join(run_dir, "config_resolved.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        log(f"Config copied to: {os.path.join(run_dir, 'config_copy.yaml')}")
        log(f"Config resolved copied to: {os.path.join(run_dir, 'config_resolved.yaml')}")
    
    # --- WANDB SETUP (rank0 only in DDP) ---
    if is_rank0 and config['logging'].get('use_wandb', False):
        # Prefer WANDB_API_KEY from environment; fall back to config field.
        env_key = os.environ.get("WANDB_API_KEY")
        cfg_key = config['logging'].get('wandb_api_key')
        if env_key:
            wandb.login(key=env_key)
        elif cfg_key:
            wandb.login(key=cfg_key)
        wandb.init(
            project=config['logging']['project_name'],
            name=config['logging']['run_name'],
            config=config
        )

    # 1. Load Pace Bins
    with open(config['data']['pace_lookup'], 'rb') as f:
        pace_data = pickle.load(f)
        # Assuming pace_data is a dict where we can extract this.
        # Structure from 06_Pace_Grammar_Creation.ipynb:
        # { "pace_0": { 'start': ..., 'end': ..., 'median_pace': ..., 'token': ... }, ... }
        pace_bins = []
        if isinstance(pace_data, dict):
            for token, info in pace_data.items():
                # Handle both 'median' and 'median_pace' keys just in case
                median_val = info.get('median_pace', info.get('median', 0))
                pace_bins.append({
                    'token': token,
                    'start': info['start'],
                    'end': info['end'],
                    'median': median_val
                })
            # Ensure they are sorted by median value
            pace_bins = sorted(pace_bins, key=lambda x: x['median'])
        else:
            # If it's already a list, use it as is
            pace_bins = pace_data

    # 2. Load Runners & Build Vocab
    splits_dir = Path(config['data']['splits_dir'])
    split_files = sorted(list(splits_dir.glob("*.pkl.gz")))
    
    all_runners = []
    vocab = {'<pad>': 0, '<unk>': 1}
    
    # We load a subset for testing as per the user's "test" request
    # but the logic allows for all if needed.
    num_files_to_load = config['data'].get('num_files_to_load', 5)
    
    log(f"Loading {num_files_to_load} split files and building vocabulary...")
    for i, fpath in enumerate(split_files):
        if i >= num_files_to_load: break
        log(f"  -> Loading: {fpath.name}")
        with gzip.open(fpath, 'rb') as f:
            while True:
                try:
                    batch = pickle.load(f)
                    for r in batch:
                        all_runners.append(r)
                        for ex in r.training_examples:
                            for token in ex.unpadded_example_sequence:
                                if token not in vocab:
                                    vocab[token] = len(vocab)
                except EOFError:
                    break
    
    log(f"Vocab size: {len(vocab)}")
    log(f"Number of runners: {len(all_runners)}")
    
    # Data Sanity Check
    # Computing full pace stats can be expensive on large datasets; keep it off by default.
    if is_rank0 and config['logging'].get('compute_pace_stats', False):
        all_paces = [ex.actual_pace_seconds for r in all_runners for ex in r.training_examples]
        if all_paces:
            log(f"Pace Stats: Min={min(all_paces):.1f}s, Max={max(all_paces):.1f}s, Mean={sum(all_paces)/len(all_paces):.1f}s, Unique={len(set(all_paces))}")
            if len(set(all_paces)) == 1:
                log("CRITICAL WARNING: All examples have the IDENTICAL actual_pace_seconds!")
    
    # Split Train/Val
    random.shuffle(all_runners)
    val_size = int(len(all_runners) * config['training']['val_split'])
    val_runners = all_runners[:val_size]
    train_runners = all_runners[val_size:]
    
    train_ds = RunTimeDataset(train_runners, vocab, pace_bins, config)
    val_ds = RunTimeDataset(val_runners, vocab, pace_bins, config)
    
    # Log dataset sizes (before DDP splitting)
    if is_rank0:
        train_examples_full = len(train_ds)
        val_examples_full = len(val_ds)
        total_examples = train_examples_full + val_examples_full
        batch_size = config['training']['batch_size']
        steps_per_epoch_full = (train_examples_full + batch_size - 1) // batch_size  # Ceiling division
        log(f"Dataset sizes (full): Train={train_examples_full:,} examples, Val={val_examples_full:,} examples, Total={total_examples:,} examples")
        log(f"Batch size: {batch_size}, Steps per epoch (full dataset): {steps_per_epoch_full:,}")
        if is_dist:
            log(f"  → With DDP (world_size={world_size}): Each GPU sees ~{train_examples_full // world_size:,} train examples, ~{steps_per_epoch_full // world_size:,} steps/epoch")

    # Distributed samplers (if DDP)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_dist else None
    
    # pin_memory is not supported on MPS
    use_pin_memory = config['training']['pin_memory'] and device_type == 'cuda'
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=use_pin_memory,
        persistent_workers=config['training']['num_workers'] > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if config['training']['num_workers'] > 0 else None,  # Prefetch 2 batches per worker
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=use_pin_memory,
        persistent_workers=config['training']['num_workers'] > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if config['training']['num_workers'] > 0 else None,  # Prefetch 2 batches per worker
    )
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 3. Model, Loss, Optimizer
    model = RunTimeTransformer(len(vocab), len(pace_bins), config).to(device)
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                                 weight_decay=config['training']['weight_decay'])
    
    # Scheduler: Reduce learning rate when validation improvement plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    criterion = nn.KLDivLoss(reduction='batchmean') # Since targets are soft distributions
    
    # GradScaler is primarily for CUDA. Disable if on CPU/MPS for now.
    scaler_device = 'cuda' if device_type == 'cuda' else 'cuda' 
    scaler = GradScaler(device=scaler_device, enabled=config['training']['use_amp'] and device_type == 'cuda')
    
    pace_values = torch.tensor([b['median'] for b in pace_bins], dtype=torch.float32).to(device)
    
    # Precompute bin boundaries for calibration (cached to avoid recomputing)
    bin_ends_cached = torch.tensor([b['end'] for b in pace_bins], dtype=torch.float32).to(device)
    bin_starts_cached = torch.tensor([b['start'] for b in pace_bins], dtype=torch.float32).to(device)
    
    # 3.5. Resume from checkpoint if available
    start_epoch = 0
    best_val_mae = float('inf')
    resume_checkpoint_path = config.get('training', {}).get('resume_from_checkpoint', None)
    
    # If no explicit checkpoint path, try to find latest_checkpoint.pt in run_dir
    if resume_checkpoint_path is None:
        latest_checkpoint_path = os.path.join(run_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint_path):
            resume_checkpoint_path = latest_checkpoint_path
            if is_rank0:
                log(f"Found latest checkpoint at: {resume_checkpoint_path}")
    
    # Load checkpoint if it exists
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        if is_rank0:
            log(f"Resuming from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        
        # Load model state
        if 'model_state_dict' in checkpoint:
            if is_dist:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            if is_rank0:
                log("✓ Model state loaded")
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_rank0:
                current_lr = optimizer.param_groups[0]['lr']
                log(f"✓ Optimizer state loaded (LR: {current_lr:.6f})")
        
        # Load scaler state (if available)
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            if hasattr(scaler, 'load_state_dict'):
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_rank0:
                    log("✓ GradScaler state loaded")
        
        # Resume from the next epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            if is_rank0:
                log(f"✓ Resuming from epoch {start_epoch}")
        
        # Restore best validation MAE
        if 'val_mae' in checkpoint:
            best_val_mae = checkpoint['val_mae']
            if is_rank0:
                log(f"✓ Best validation MAE: {best_val_mae:.4f}")
        
        # Restore scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_rank0:
                # Log scheduler state info for ReduceLROnPlateau
                if hasattr(scheduler, 'best') and hasattr(scheduler, 'num_bad_epochs'):
                    log(f"✓ Scheduler state loaded (best: {scheduler.best:.4f}, patience: {scheduler.num_bad_epochs}/{scheduler.patience})")
                else:
                    log("✓ Scheduler state loaded")
    else:
        if is_rank0 and resume_checkpoint_path:
            log(f"⚠️  Checkpoint not found: {resume_checkpoint_path}, starting from scratch")
    
    # 4. Loop
    global_step = 0
    for epoch in range(start_epoch, config['training']['epochs']):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss = 0
        train_mae_accum = 0
        samples_processed = 0
        
        # --- DEBUG TRACKERS ---
        batch_modes = []
        batch_entropies = []
        train_rmse_accum = 0.0  # Track squared errors for RMSE
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            ids = batch['input_ids'].to(device)
            mask = batch['padding_mask'].to(device)
            targets = batch['soft_target'].to(device)
            actual_paces = batch['actual_pace'].to(device)
            
            # Use appropriate device_type for autocast (cuda or cpu)
            act_device_type = 'cuda' if device_type == 'cuda' else 'cpu'
            
            with autocast(device_type=act_device_type, enabled=config['training']['use_amp']):
                logits = model(ids, mask)
                log_probs = torch.log_softmax(logits, dim=1)
                loss = criterion(log_probs, targets)
            
            scaler.scale(loss).backward()
            
            # Compute gradient norms if logging enabled (before stepping)
            grad_norm = None
            param_norms_dict = {}
            should_log_grads = (is_rank0 and 
                               batch_idx % config['logging'].get('log_interval', 10) == 0 and
                               config['logging'].get('log_gradients', False))
            
            if should_log_grads:
                # Unscale gradients for norm computation (scaler scales them when using AMP)
                scaler.unscale_(optimizer)
                # Compute gradient norms
                total_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        # Store per-layer norms if enabled
                        if config['logging'].get('log_per_layer_grads', False):
                            param_norms_dict[name] = param_norm.item()
                grad_norm = total_norm ** (1. / 2)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            global_step += 1
            
            # --- CALCULATE TRAIN MAE, RMSE, AND CALIBRATION ---
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                pred_paces = (probs * pace_values).sum(dim=1)
                
                # MAE
                batch_mae = torch.abs(pred_paces - actual_paces).mean().item()
                train_mae_accum += batch_mae * ids.size(0)
                
                # RMSE (accumulate squared errors)
                batch_squared_errors = (pred_paces - actual_paces) ** 2
                train_rmse_accum += batch_squared_errors.sum().item()
                
                samples_processed += ids.size(0)
                
                # Entropy: -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                batch_entropies.append(entropy)
                
                # Calibration: compute percentiles for batch (only when logging to avoid overhead)
                # Mean percentile should be ~50 if well-calibrated
                batch_percentiles = []
                should_compute_calibration = (config['logging'].get('log_batch_calibration', False) and 
                                            is_rank0 and 
                                            batch_idx % config['logging'].get('log_interval', 10) == 0)
                
                if should_compute_calibration:
                    # Vectorized percentile computation (much faster than Python loop)
                    probs_normalized = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                    cdf = torch.cumsum(probs_normalized, dim=1)
                    
                    # Use cached bin boundaries (precomputed once)
                    # Vectorized: find bin indices for all actual paces at once
                    bin_indices = torch.searchsorted(bin_ends_cached, actual_paces)
                    
                    # Handle edge cases
                    bin_indices = torch.clamp(bin_indices, 0, len(bin_ends_cached) - 1)
                    
                    # Vectorized interpolation
                    # Get CDF values at bin starts and ends
                    cdf_at_ends = torch.gather(cdf, 1, bin_indices.unsqueeze(1)).squeeze(1)
                    cdf_at_starts = torch.zeros_like(cdf_at_ends)
                    valid_start_mask = bin_indices > 0
                    if valid_start_mask.any():
                        cdf_at_starts[valid_start_mask] = torch.gather(
                            cdf[valid_start_mask], 1, (bin_indices[valid_start_mask] - 1).unsqueeze(1)
                        ).squeeze(1)
                    
                    # Get bin boundaries from cached tensors
                    bin_start_vals = bin_starts_cached[bin_indices]
                    bin_end_vals = bin_ends_cached[bin_indices]
                    
                    # Linear interpolation within bins
                    bin_widths = bin_end_vals - bin_start_vals
                    valid_width_mask = bin_widths > 1e-10
                    fractions = torch.zeros_like(actual_paces)
                    fractions[valid_width_mask] = (actual_paces[valid_width_mask] - bin_start_vals[valid_width_mask]) / bin_widths[valid_width_mask]
                    
                    # Compute CDF at actual pace
                    cdf_at_actual = cdf_at_starts + fractions * (cdf_at_ends - cdf_at_starts)
                    
                    # Convert to percentiles
                    batch_percentiles = (cdf_at_actual * 100.0).cpu().tolist()
                
                modes = torch.argmax(probs, dim=1)
                batch_modes.extend(modes.cpu().numpy())

            # --- PERIODIC LOGGING (rank0 only in DDP) ---
            if is_rank0 and batch_idx % config['logging'].get('log_interval', 10) == 0:
                if config['logging'].get('use_wandb', False):
                    # Compute batch RMSE
                    batch_rmse = torch.sqrt(batch_squared_errors.mean()).item()
                    
                    log_dict = {
                        "train/batch_loss": loss.item(),
                        "train/batch_mae": batch_mae,
                        "train/batch_rmse": batch_rmse,
                        "train/entropy": entropy,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "global_step": global_step
                    }
                    
                    # Add gradient norm if enabled
                    if config['logging'].get('log_gradients', False) and grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm
                        # Add per-layer norms if enabled (can be verbose)
                        if config['logging'].get('log_per_layer_grads', False) and param_norms_dict:
                            for name, norm_value in param_norms_dict.items():
                                log_dict[f"grad_norm/{name}"] = norm_value
                    
                    # Add calibration metrics if enabled
                    if config['logging'].get('log_batch_calibration', False) and len(batch_percentiles) > 0:
                        mean_percentile = np.mean(batch_percentiles)
                        # Mean percentile should be ~50 for well-calibrated model
                        # Deviation from 50 indicates systematic bias
                        calibration_bias = mean_percentile - 50.0
                        log_dict["train/batch_calibration_mean_percentile"] = mean_percentile
                        log_dict["train/batch_calibration_bias"] = calibration_bias
                    
                    wandb.log(log_dict)

            if is_rank0 and batch_idx == 0 and config['logging'].get('batch0_diagnostics', True):
                log(f"\n[Batch 0 Diagnostics]")
                unique_modes = len(set(modes.cpu().numpy()))
                log(f"  Unique Predicted Bins in Batch: {unique_modes} / {len(modes)}")
                log(f"  Average Entropy (sharpness): {entropy:.4f} (Low = Sharp, High = Flat)")
                log(f"  Sample Actual Paces: {batch['actual_pace'][:3].tolist()}")
                # Check if targets are diverse
                target_modes = torch.argmax(targets, dim=1)
                log(f"  Target Bin Modes: {target_modes[:5].tolist()}")

        # Epoch Stats (Calculating Train Stats) - reduce across ranks if DDP
        total_train_loss = _all_reduce_sum_float(train_loss, device)
        total_train_batches = _all_reduce_sum_int(len(train_loader), device)
        total_train_abs = _all_reduce_sum_float(train_mae_accum, device)
        total_train_rmse_squared = _all_reduce_sum_float(train_rmse_accum, device)
        total_train_count = _all_reduce_sum_int(samples_processed, device)

        avg_train_loss = total_train_loss / max(1, total_train_batches)
        avg_train_mae = total_train_abs / max(1, total_train_count)
        avg_train_rmse = np.sqrt(total_train_rmse_squared / max(1, total_train_count))
        unique_total_modes = len(set(batch_modes))
        avg_entropy = sum(batch_entropies) / len(batch_entropies)
        
        # Validation
        model.eval()
        val_abs = 0.0
        val_squared = 0.0  # For RMSE
        val_count = 0
        val_loss = 0.0
        val_percentiles = []  # For calibration metrics
        
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['padding_mask'].to(device)
                targets = batch['soft_target'].to(device)
                actual_paces = batch['actual_pace'].to(device)
                
                logits = model(ids, mask)
                log_probs = torch.log_softmax(logits, dim=1)
                batch_loss = criterion(log_probs, targets)
                val_loss += batch_loss.item()
                
                probs = torch.softmax(logits, dim=1)
                
                # Weighted Mean Prediction
                pred_paces = (probs * pace_values).sum(dim=1)
                val_abs += torch.abs(pred_paces - actual_paces).sum().item()
                val_squared += ((pred_paces - actual_paces) ** 2).sum().item()
                val_count += int(actual_paces.numel())
                
                # Compute calibration percentiles if enabled (vectorized for speed)
                if config['logging'].get('log_batch_calibration', False):
                    probs_normalized = probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
                    cdf = torch.cumsum(probs_normalized, dim=1)
                    
                    # Use cached bin boundaries (precomputed once)
                    # Vectorized: find bin indices for all actual paces at once
                    bin_indices = torch.searchsorted(bin_ends_cached, actual_paces)
                    bin_indices = torch.clamp(bin_indices, 0, len(bin_ends_cached) - 1)
                    
                    # Vectorized interpolation
                    cdf_at_ends = torch.gather(cdf, 1, bin_indices.unsqueeze(1)).squeeze(1)
                    cdf_at_starts = torch.zeros_like(cdf_at_ends)
                    valid_start_mask = bin_indices > 0
                    if valid_start_mask.any():
                        cdf_at_starts[valid_start_mask] = torch.gather(
                            cdf[valid_start_mask], 1, (bin_indices[valid_start_mask] - 1).unsqueeze(1)
                        ).squeeze(1)
                    
                    # Get bin boundaries from cached tensors
                    bin_start_vals = bin_starts_cached[bin_indices]
                    bin_end_vals = bin_ends_cached[bin_indices]
                    
                    # Linear interpolation within bins
                    bin_widths = bin_end_vals - bin_start_vals
                    valid_width_mask = bin_widths > 1e-10
                    fractions = torch.zeros_like(actual_paces)
                    fractions[valid_width_mask] = (actual_paces[valid_width_mask] - bin_start_vals[valid_width_mask]) / bin_widths[valid_width_mask]
                    
                    # Compute CDF at actual pace
                    cdf_at_actual = cdf_at_starts + fractions * (cdf_at_ends - cdf_at_starts)
                    
                    # Convert to percentiles
                    val_percentiles.extend((cdf_at_actual * 100.0).cpu().tolist())
        
        # Reduce across ranks if DDP
        total_val_abs = _all_reduce_sum_float(val_abs, device)
        total_val_squared = _all_reduce_sum_float(val_squared, device)
        total_val_count = _all_reduce_sum_int(val_count, device)
        total_val_loss = _all_reduce_sum_float(val_loss, device)
        total_val_batches = _all_reduce_sum_int(len(val_loader), device)

        avg_val_mae = total_val_abs / max(1, total_val_count)
        avg_val_rmse = np.sqrt(total_val_squared / max(1, total_val_count))
        avg_val_loss = total_val_loss / max(1, total_val_batches)
        
        # Compute calibration KS statistic and p-value if enabled
        val_ks_stat = None
        val_ks_pvalue = None
        if config['logging'].get('log_batch_calibration', False) and len(val_percentiles) > 0:
            # Convert percentiles to 0-1 range for KS test
            val_percentiles_fraction = np.array(val_percentiles) / 100.0
            # KS test: null hypothesis is that percentiles follow Uniform(0,1)
            # If well-calibrated, percentiles should be uniform
            val_ks_stat, val_ks_pvalue = kstest(val_percentiles_fraction, 'uniform')
        
        # Step the scheduler
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_rank0:
            log_msg = f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {avg_train_loss:.4f} | Train MAE: {avg_train_mae:.2f}s | Train RMSE: {avg_train_rmse:.2f}s | Val MAE: {avg_val_mae:.2f}s | Val RMSE: {avg_val_rmse:.2f}s | LR: {current_lr:.6f}"
            if val_ks_stat is not None:
                log_msg += f" | Val Calibration KS: {val_ks_stat:.4f} (p={val_ks_pvalue:.4f})"
            log(log_msg)
            log(f"  Prediction Diversity: {unique_total_modes} unique bins predicted this epoch")
            log(f"  Model Confidence (Avg Entropy): {avg_entropy:.4f}")
        
        if is_rank0 and config['logging'].get('use_wandb', False):
            log_dict = {
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_mae": avg_train_mae,
                "train/epoch_rmse": avg_train_rmse,
                "val/mae": avg_val_mae,
                "val/rmse": avg_val_rmse,
                "val/loss": avg_val_loss,
                "train/epoch_lr": current_lr,
                "diversity/unique_bins": unique_total_modes,
                "diversity/avg_entropy": avg_entropy
            }
            
            # Add validation calibration metrics if enabled
            if config['logging'].get('log_batch_calibration', False) and len(val_percentiles) > 0:
                val_mean_percentile = np.mean(val_percentiles)
                val_calibration_bias = val_mean_percentile - 50.0
                log_dict["val/calibration_mean_percentile"] = val_mean_percentile
                log_dict["val/calibration_bias"] = val_calibration_bias
                if val_ks_stat is not None:
                    log_dict["val/calibration_ks_statistic"] = val_ks_stat
                    log_dict["val/calibration_ks_pvalue"] = val_ks_pvalue
            
            wandb.log(log_dict)

        # --- VERBOSE SAMPLES (rank0 only; expensive, so off by default) ---
        if is_rank0 and config['logging'].get('verbose_samples', False):
            log(f"\n--- [Epoch {epoch+1}] Verbose Samples ---")
            n_verbose = int(config['logging'].get('num_verbose_samples', 5))
            sample_indices = random.sample(range(len(val_ds)), min(n_verbose, len(val_ds)))
            model_for_infer = model.module if is_dist else model
            for s_idx in sample_indices:
                batch_item = val_ds[s_idx]
                s_ids = batch_item['input_ids'].unsqueeze(0).to(device)
                s_mask = batch_item['padding_mask'].unsqueeze(0).to(device)
                s_actual = batch_item['actual_pace'].item()
                with torch.no_grad():
                    s_logits = model_for_infer(s_ids, s_mask)
                    s_probs = torch.softmax(s_logits, dim=1)[0]
                    mode_idx = torch.argmax(s_probs).item()
                    s_pred_mode_token = pace_bins[mode_idx]['token']
                    s_actual_token = "<unknown>"
                    for b in pace_bins:
                        if b['start'] <= s_actual < b['end']:
                            s_actual_token = b['token']
                            break
                    s_pred_mean = (s_probs * pace_values).sum().item()
                    cumsum = torch.cumsum(s_probs, dim=0)
                    median_idx = torch.searchsorted(cumsum, 0.5).item()
                    median_idx = min(median_idx, len(pace_bins) - 1)
                    s_pred_median = pace_bins[median_idx]['median']
                    s_tokens = [inv_vocab.get(i.item(), '<unk>') for i in batch_item['input_ids'] if i.item() != 0]
                log(f"Grammar: {' '.join(s_tokens)}")
                log(f"Actual Pace: {s_actual:.2f}s ({s_actual_token}) | Predicted Mean: {s_pred_mean:.2f}s | Predicted Median: {s_pred_median:.2f}s | Mode: {s_pred_mode_token}")
                log("-" * 50)

        # --- SAVE CHECKPOINTS (rank0 only; minimize I/O by default) ---
        if is_rank0:
            checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt")
            latest_path = os.path.join(run_dir, "latest_checkpoint.pt")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': (model.module.state_dict() if is_dist else model.state_dict()),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if hasattr(scaler, 'state_dict') else None,
                'vocab': vocab,
                'inv_vocab': inv_vocab,
                'config': config,
                'val_mae': avg_val_mae,
                'best_val_mae': best_val_mae
            }

            # Always keep best; keep \"latest\" periodically to reduce I/O if desired.
            latest_every = int(config.get('training', {}).get('latest_checkpoint_every_n_epochs', 1))
            if latest_every <= 1 or (epoch + 1) % latest_every == 0:
                torch.save(checkpoint, latest_path)
            save_every = int(config.get('training', {}).get('checkpoint_every_n_epochs', 0))
            save_all = bool(config.get('training', {}).get('save_epoch_checkpoints', False))
            if save_all or (save_every and (epoch + 1) % save_every == 0):
                torch.save(checkpoint, checkpoint_path)
                log(f"Checkpoint saved: {checkpoint_path}")

            # Best model logic
            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                best_path = os.path.join(run_dir, "best_model.pt")
                torch.save(checkpoint, best_path)
                log(f"New best model saved: {best_path} (Val MAE: {avg_val_mae:.2f}s)")

            # Write lightweight metrics for external tools (Optuna, dashboards, etc.)
            try:
                import json
                with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                    json.dump({"epoch": epoch + 1, "val_mae": avg_val_mae, "lr": current_lr}, f)
            except Exception:
                pass

    # DDP cleanup
    if is_dist and _dist_is_init():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='runtime_trainer_config.yaml')
    # torchrun may pass --local_rank; accept it to avoid argparse errors.
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    
    train_model(args.config)

