import os
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
        
        # Calculate max_len based on n*11 - 1 logic
        if 'max_races_to_consider' in config['model']:
            self.max_len = config['model']['max_races_to_consider'] * 11 - 1
        else:
            self.max_len = config['model']['max_seq_length']
            
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
        self.pos_encoder = PositionalEncoding(m['d_model'], m['max_seq_length'])
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=m['d_model'],
            nhead=m['nhead'],
            dim_feedforward=m['dim_feedforward'],
            dropout=m['dropout'],
            batch_first=True
        )
        # Using TransformerEncoder as a Decoder by applying a causal mask
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=m['num_layers'])
        
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
def train_model(config_path):
    config = load_config(config_path)
    
    # Device setup: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'
        
    device = torch.device(device_type)
    print(f"Using device: {device}")
    
    # --- DIRECTORY SETUP ---
    save_dir_base = config['logging'].get('save_dir', 'checkpoints')
    run_name = config['logging'].get('run_name', 'default_run')
    run_dir = os.path.join(save_dir_base, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save a copy of the config file up front
    import shutil
    shutil.copy(config_path, os.path.join(run_dir, "config_copy.yaml"))
    print(f"Config copied to: {os.path.join(run_dir, 'config_copy.yaml')}")
    
    # Optional: step-based checkpointing (useful for sweeps / preemptible runs).
    # - If checkpoint_interval_steps <= 0, checkpoints are only saved at end-of-epoch (default behavior).
    # - If > 0, we update latest_checkpoint.pt every N optimizer steps so there is always something to resume from.
    checkpoint_interval_steps = int(config.get('training', {}).get('checkpoint_interval_steps', 0) or 0)
    keep_step_checkpoints = bool(config.get('training', {}).get('keep_step_checkpoints', False))
    
    # --- WANDB SETUP ---
    if config['logging'].get('use_wandb', False):
        if config['logging'].get('wandb_api_key'):
            wandb.login(key=config['logging']['wandb_api_key'])
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
    
    print(f"Loading {num_files_to_load} split files and building vocabulary...")
    for i, fpath in enumerate(split_files):
        if i >= num_files_to_load: break
        print(f"  -> Loading: {fpath.name}")
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
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of runners: {len(all_runners)}")
    
    # Data Sanity Check
    all_paces = [ex.actual_pace_seconds for r in all_runners for ex in r.training_examples]
    if all_paces:
        print(f"Pace Stats: Min={min(all_paces):.1f}s, Max={max(all_paces):.1f}s, Mean={sum(all_paces)/len(all_paces):.1f}s, Unique={len(set(all_paces))}")
        if len(set(all_paces)) == 1:
            print("CRITICAL WARNING: All examples have the IDENTICAL actual_pace_seconds!")
    
    # Split Train/Val
    random.shuffle(all_runners)
    val_size = int(len(all_runners) * config['training']['val_split'])
    val_runners = all_runners[:val_size]
    train_runners = all_runners[val_size:]
    
    train_ds = RunTimeDataset(train_runners, vocab, pace_bins, config)
    val_ds = RunTimeDataset(val_runners, vocab, pace_bins, config)
    
    # pin_memory is not supported on MPS
    use_pin_memory = config['training']['pin_memory'] and device_type == 'cuda'
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, 
                              num_workers=config['training']['num_workers'], pin_memory=use_pin_memory)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False,
                            num_workers=config['training']['num_workers'], pin_memory=use_pin_memory)
    
    inv_vocab = {v: k for k, v in vocab.items()}
    
    # 3. Model, Loss, Optimizer
    model = RunTimeTransformer(len(vocab), len(pace_bins), config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], 
                                 weight_decay=config['training']['weight_decay'])
    
    # Scheduler: Reduce learning rate when validation improvement plateaus.
    # Torch compatibility: some builds don't accept `verbose=...` here.
    import inspect
    _rlrop_sig = inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau)
    _rlrop_kwargs = dict(mode='min', factor=0.5, patience=3)
    if 'verbose' in _rlrop_sig.parameters:
        _rlrop_kwargs['verbose'] = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **_rlrop_kwargs)
    
    criterion = nn.KLDivLoss(reduction='batchmean') # Since targets are soft distributions
    
    # GradScaler is primarily for CUDA. Disable if on CPU/MPS for now.
    scaler_device = 'cuda' if device_type == 'cuda' else 'cuda' 
    scaler = GradScaler(device=scaler_device, enabled=config['training']['use_amp'] and device_type == 'cuda')
    
    pace_values = torch.tensor([b['median'] for b in pace_bins], dtype=torch.float32).to(device)
    
    # 4. Loop
    global_step = 0
    for epoch in range(config['training']['epochs']):
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

            # --- OPTIONAL STEP CHECKPOINTING ---
            if checkpoint_interval_steps > 0 and (global_step % checkpoint_interval_steps) == 0:
                latest_path = os.path.join(run_dir, "latest_checkpoint.pt")
                ckpt = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab': vocab,
                    'inv_vocab': inv_vocab,
                    'config': config,
                    # val_mae not computed mid-epoch
                }
                torch.save(ckpt, latest_path)
                if keep_step_checkpoints:
                    step_path = os.path.join(run_dir, f"checkpoint_step_{global_step}.pt")
                    torch.save(ckpt, step_path)
                print(f"[checkpoint] saved latest at step {global_step}: {latest_path}")
            
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

            # --- PERIODIC LOGGING ---
            if batch_idx % config['logging'].get('log_interval', 10) == 0:
                if config['logging'].get('use_wandb', False):
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/batch_mae": batch_mae,
                        "train/entropy": entropy,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "global_step": global_step
                    })

            if batch_idx == 0:
                print(f"\n[Batch 0 Diagnostics]")
                unique_modes = len(set(modes.cpu().numpy()))
                print(f"  Unique Predicted Bins in Batch: {unique_modes} / {len(modes)}")
                print(f"  Average Entropy (sharpness): {entropy:.4f} (Low = Sharp, High = Flat)")
                print(f"  Sample Actual Paces: {batch['actual_pace'][:3].tolist()}")
                # Check if targets are diverse
                target_modes = torch.argmax(targets, dim=1)
                print(f"  Target Bin Modes: {target_modes[:5].tolist()}")

        # Epoch Stats (Calculating Train Stats)
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae_accum / samples_processed
        unique_total_modes = len(set(batch_modes))
        avg_entropy = sum(batch_entropies) / len(batch_entropies)
        
        # Validation
        model.eval()
        val_mae = 0
        val_loss = 0
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
                val_mae += torch.abs(pred_paces - actual_paces).sum().item()
        
        avg_val_mae = val_mae / len(val_ds)
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Loss: {avg_train_loss:.4f} | Val MAE: {avg_val_mae:.2f}s | LR: {current_lr:.6f}")
        print(f"  Prediction Diversity: {unique_total_modes} unique bins predicted this epoch")
        print(f"  Model Confidence (Avg Entropy): {avg_entropy:.4f}")
        
        if config['logging'].get('use_wandb', False):
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

        # --- VERBOSE SAMPLES ---
        print(f"\n--- [Epoch {epoch+1}] Verbose Samples ---")
        # Sample 5 indices from validation dataset
        sample_indices = random.sample(range(len(val_ds)), min(5, len(val_ds)))
        for s_idx in sample_indices:
            batch_item = val_ds[s_idx]
            s_ids = batch_item['input_ids'].unsqueeze(0).to(device)
            s_mask = batch_item['padding_mask'].unsqueeze(0).to(device)
            s_actual = batch_item['actual_pace'].item()
            
            with torch.no_grad():
                s_logits = model(s_ids, s_mask)
                s_probs = torch.softmax(s_logits, dim=1)[0]
                
                # Mode Prediction (Highest Probability Bin)
                mode_idx = torch.argmax(s_probs).item()
                s_pred_mode_token = pace_bins[mode_idx]['token']
                
                # Actual Bin (Which bin does the high-precision actual pace fall into?)
                s_actual_token = "<unknown>"
                for b in pace_bins:
                    if b['start'] <= s_actual < b['end']:
                        s_actual_token = b['token']
                        break
                
                # Mean Prediction (Weighted Average)
                s_pred_mean = (s_probs * pace_values).sum().item()
                
                # Median Prediction Logic
                cumsum = torch.cumsum(s_probs, dim=0)
                median_idx = torch.searchsorted(cumsum, 0.5).item()
                median_idx = min(median_idx, len(pace_bins) - 1)
                s_pred_median = pace_bins[median_idx]['median']
                
                # Decode tokens (filtering out padding)
                s_tokens = [inv_vocab.get(i.item(), '<unk>') for i in batch_item['input_ids'] if i.item() != 0]
            
            print(f"Grammar: {' '.join(s_tokens)}")
            print(f"Actual Pace: {s_actual:.2f}s ({s_actual_token}) | Predicted Mean: {s_pred_mean:.2f}s | Predicted Median: {s_pred_median:.2f}s | Mode: {s_pred_mode_token}")
            print("-" * 50)

        # --- SAVE CHECKPOINTS ---
        checkpoint_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch+1}.pt")
        latest_path = os.path.join(run_dir, "latest_checkpoint.pt")
        
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab': vocab,
            'inv_vocab': inv_vocab,
            'config': config,
            'val_mae': avg_val_mae
        }
        
        torch.save(checkpoint, checkpoint_path)
        torch.save(checkpoint, latest_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Best model logic
        if 'best_val_mae' not in locals() or avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            best_path = os.path.join(run_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"New best model saved: {best_path} (Val MAE: {avg_val_mae:.2f}s)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='runtime_trainer_config.yaml')
    args = parser.parse_args()
    
    train_model(args.config)

