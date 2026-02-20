import os
import sys
import gzip
import pickle
import math
import torch
import torch.nn as nn
import numpy as np
import random
import yaml
import wandb
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.amp import GradScaler, autocast
# Removed scipy dependency
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
def get_gaussian_targets(actual_pace, pace_bins, sigma):
    """
    Implements Seconds-Aware Gaussian Smoothing using the Error Function (erf)
    as described in the RunTime White Paper.
    """
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
        # Find nearest bin
        medians = np.array([b['median'] for b in pace_bins])
        idx = np.argmin(np.abs(medians - actual_pace))
        targets = np.zeros(len(pace_bins))
        targets[idx] = 1.0
        
    return torch.tensor(targets, dtype=torch.float32)

# --- DATASET ---
class RunTimeDataset(Dataset):
    def __init__(self, runners, vocab, pace_bins, config):
        self.examples = []
        self.vocab = vocab
        self.pace_bins = pace_bins
        self.config = config
        
        # Token layout (per race) is a fixed stride in the grammar.
        # Default: 11 tokens per race, and we predict the final token (pace) so input length is (11*n - 1).
        data_cfg = config.get('data', {}) if isinstance(config.get('data'), dict) else {}
        self.block_stride = int(data_cfg.get('block_stride', 11))
        
        # Ablation settings
        self.ablation_drop_week_deltas = bool(data_cfg.get('ablation_drop_week_deltas', False))
        self.ablation_last_age_front = bool(data_cfg.get('ablation_last_age_front', False))
        self.ablation_out_stride = int(data_cfg.get('ablation_out_stride', 8))
        self.ablation_shuffle_races = bool(data_cfg.get('ablation_shuffle_races', False))
        
        # Legacy swap_pace_time support (for backward compatibility)
        self.swap_pace_time = bool(data_cfg.get('swap_pace_time_tokens', False))
        self.drop_final_time_tokens = int(data_cfg.get('drop_final_time_tokens', 2 if self.swap_pace_time else 0))

        # Calculate max_len based on ablation settings
        if self.ablation_drop_week_deltas and self.ablation_last_age_front:
            # Sequence becomes: [age_last] + [8 tokens] * (n-1 races) + [7 tokens for final block]
            # = 1 + 8*(n-1) + 7 total tokens
            # The final pace token is excluded from the final block (we predict it),
            # so the input sequence length is: 1 + 8*(n-1) + 7
            if 'max_races_to_consider' in config['model']:
                n_races = config['model']['max_races_to_consider']
                self.max_len = 1 + (self.ablation_out_stride * (n_races - 1)) + 7  # 1 + 8*(n-1) + 7
            else:
                self.max_len = int(config['model']['max_seq_length'])
        elif 'max_races_to_consider' in config['model']:
            self.max_len = config['model']['max_races_to_consider'] * self.block_stride - 1 - self.drop_final_time_tokens
        else:
            # If max_seq_length is explicitly set, assume it already matches the desired token order.
            self.max_len = int(config['model']['max_seq_length'])
            
        self.sigma = config['training']['smoothing_sigma_seconds']
        
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

        # Apply ablation transformations if enabled
        if self.ablation_drop_week_deltas and self.ablation_last_age_front:
            if self.ablation_shuffle_races:
                seq = self._apply_ablation_shuffled(seq)
            else:
                seq = self._apply_ablation(seq)
        elif self.swap_pace_time:
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
        # Use the precise actual_pace_seconds if available, else use bin median
        actual_pace = ex.actual_pace_seconds
        soft_target = get_gaussian_targets(actual_pace, self.pace_bins, self.sigma)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'padding_mask': torch.tensor(mask, dtype=torch.bool),
            'soft_target': soft_target,
            'actual_pace': torch.tensor(actual_pace, dtype=torch.float32)
        }

    def _apply_ablation_shuffled(self, seq: list) -> list:
        """
        Apply ablation transformations with shuffled historical races:
        1. Drop all week_delta_* tokens entirely
        2. Keep only the age token from the final race and move it to the front
        3. Shuffle all historical races (but keep final race fixed at the end)
        4. Each historical race block: 8 tokens [gender, conditions, humidity, temp, feels_like, wind, distance, pace]
        5. Final block: 7 tokens [gender, conditions, humidity, temp, feels_like, wind, distance] (no pace, we predict it)
        6. Final sequence: [age_at_final] + [shuffled 8-token blocks] + [7-token final block]
        
        Original 11-token block structure:
        [age, gender, conditions, humidity, temp, feels_like, wind, distance, week_delta_next, week_delta_final, pace]
        """
        stride = self.block_stride
        if not seq or len(seq) < stride or (len(seq) % stride) != 0:
            return seq

        nblocks = len(seq) // stride
        if nblocks < 2:
            # Need at least 2 blocks (one historical + final)
            return self._apply_ablation(seq)
        
        # Extract blocks
        blocks = []
        final_age_token = None
        
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

    def _apply_ablation(self, seq: list) -> list:
        """
        Apply ablation transformations:
        1. Drop all week_delta_* tokens entirely
        2. Keep only the last age token and move it to the front
        3. Each race block (except final): 8 tokens [gender, conditions, humidity, temp, feels_like, wind, distance, pace]
        4. Final block: 7 tokens [gender, conditions, humidity, temp, feels_like, wind, distance] (no pace, we predict it)
        5. Final sequence: [age_last] + [8 tokens per race] * (n-1) + [7 tokens for final block] = 1 + 8*(n-1) + 7 = 8*n
        (where n includes the final race, so n-1 full blocks + 1 final block of 7 tokens)
        
        Original 11-token block structure:
        [age, gender, conditions, humidity, temp, feels_like, wind, distance, week_delta_next, week_delta_final, pace]
        """
        stride = self.block_stride
        if not seq or len(seq) < stride or (len(seq) % stride) != 0:
            return seq

        nblocks = len(seq) // stride
        out: list = []
        last_age_token = None
        
        # Process each block: extract the last age token and build blocks without week_deltas
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
            
            # Store the age token (we'll use the last one)
            last_age_token = age_token
            
            # For all blocks except the final one: include pace (8 tokens)
            # For the final block: exclude pace (7 tokens) since we predict it
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
        ablation_drop_week_deltas = bool(data_cfg.get('ablation_drop_week_deltas', False))
        ablation_last_age_front = bool(data_cfg.get('ablation_last_age_front', False))
        ablation_out_stride = int(data_cfg.get('ablation_out_stride', 8))
        swap = bool(data_cfg.get('swap_pace_time_tokens', False))
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=use_pin_memory,
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
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            global_step += 1
            
            # --- CALCULATE TRAIN MAE ---
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                pred_paces = (probs * pace_values).sum(dim=1)
                batch_mae = torch.abs(pred_paces - actual_paces).mean().item()
                train_mae_accum += batch_mae * ids.size(0)
                samples_processed += ids.size(0)
                
                # Entropy: -sum(p * log(p))
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean().item()
                batch_entropies.append(entropy)
                
                modes = torch.argmax(probs, dim=1)
                batch_modes.extend(modes.cpu().numpy())

            # --- PERIODIC LOGGING (rank0 only in DDP) ---
            if is_rank0 and batch_idx % config['logging'].get('log_interval', 10) == 0:
                if config['logging'].get('use_wandb', False):
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_mae": batch_mae,
                        "train/entropy": entropy,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "global_step": global_step
                    })

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
        total_train_count = _all_reduce_sum_int(samples_processed, device)

        avg_train_loss = total_train_loss / max(1, total_train_batches)
        avg_train_mae = total_train_abs / max(1, total_train_count)
        unique_total_modes = len(set(batch_modes))
        avg_entropy = sum(batch_entropies) / len(batch_entropies)
        
        # Validation
        model.eval()
        val_abs = 0.0
        val_count = 0
        val_loss = 0.0
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
                val_count += int(actual_paces.numel())
        
        # Reduce across ranks if DDP
        total_val_abs = _all_reduce_sum_float(val_abs, device)
        total_val_count = _all_reduce_sum_int(val_count, device)
        total_val_loss = _all_reduce_sum_float(val_loss, device)
        total_val_batches = _all_reduce_sum_int(len(val_loader), device)

        avg_val_mae = total_val_abs / max(1, total_val_count)
        avg_val_loss = total_val_loss / max(1, total_val_batches)
        
        # Step the scheduler
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        if is_rank0:
            log(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.2f}s | LR: {current_lr:.6f}")
            log(f"  Prediction Diversity: {unique_total_modes} unique bins predicted this epoch")
            log(f"  Model Confidence (Avg Entropy): {avg_entropy:.4f}")
        
        if is_rank0 and config['logging'].get('use_wandb', False):
            wandb.log({
                "epoch": epoch + 1,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_mae": avg_train_mae,
                "val/mae": avg_val_mae,
                "val/loss": avg_val_loss,
                "train/epoch_lr": current_lr,
                "diversity/unique_bins": unique_total_modes,
                "diversity/avg_entropy": avg_entropy
            })

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

            # Always keep best; keep "latest" periodically to reduce I/O if desired.
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
    parser.add_argument('--config', type=str, default='runtime_trainer_shuffled_ablation.yaml')
    # torchrun may pass --local_rank; accept it to avoid argparse errors.
    parser.add_argument('--local_rank', type=int, default=None)
    args = parser.parse_args()
    
    train_model(args.config)

