%%writefile train_ddp.py
import math
import os
import random
import psutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models, datasets
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DATA_DIR = '/kaggle/input/datasets/aryansanjeevarora/galaxy-morphology/data'

BATCH_SIZE        = 32
IMAGE_SIZE        = 224
EPOCHS            = 60
PATIENCE          = 16
LEARNING_RATE     = 3e-4
BACKBONE_LR_SCALE = 0.05

NUM_WORKERS       = 1
PIN_MEMORY        = True

CUTMIX_ALPHA = 1.0
MIXUP_ALPHA  = 0.2
CUTMIX_PROB  = 0.0
MIXUP_PROB   = 0.2

TTA_N = 16

USE_AMP = True   # Both Kaggle T4/P100 GPUs support fp16

SEED = 42

# ==========================================
# 2. Dataset
# ==========================================
class InMemoryGalaxyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images    = images
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# ==========================================
# 3. Model helpers
# ==========================================
SWIN_T_FEATURES = 768

def build_swin(num_classes, pretrained=True):
    weights = models.Swin_T_Weights.DEFAULT if pretrained else None
    model   = models.swin_t(weights=weights)
    model.head = nn.Sequential(
        nn.Linear(SWIN_T_FEATURES, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    return model

# ==========================================
# 4. Main worker (one per GPU)
# ==========================================
def main_worker(rank, world_size):
    # ---- DDP init ----
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device    = torch.device(f'cuda:{rank}')
    is_master = (rank == 0)   # Only rank-0 prints / saves

    def log(*args, **kwargs):
        if is_master:
            print(*args, **kwargs)

    # ---- Seeding ----
    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)

    # ---- Load Dataset inside the worker ----
    log("Scanning directory structure...")
    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names  = base_dataset.classes
    NUM_CLASSES  = len(class_names)

    log(f"Pre-loading images into RAM as uint8...")
    all_images, all_labels = [], []
    
    iterator = base_dataset.samples
    if is_master:
        iterator = tqdm(iterator, desc="Loading to RAM")
        
    for path, label in iterator:
        if psutil.virtual_memory().percent > 95:
            log("\n[!] WARNING: RAM > 95%, stopping early to prevent kernel crash.")
            break
        img = base_dataset.loader(path)
        all_images.append(np.array(img, dtype=np.uint8))
        all_labels.append(label)

    all_labels = np.array(all_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=SEED, stratify=all_labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )

    swin_weights  = models.Swin_T_Weights.DEFAULT
    IMAGENET_MEAN = swin_weights.transforms().mean
    IMAGENET_STD  = swin_weights.transforms().std

    # ---- Transforms ----
    train_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomChoice([
            v2.RandomRotation((0,   0)),
            v2.RandomRotation((90,  90)),
            v2.RandomRotation((180, 180)),
            v2.RandomRotation((270, 270))
        ]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    eval_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    train_dataset = InMemoryGalaxyDataset(X_train, y_train, transform=train_transforms)
    val_dataset   = InMemoryGalaxyDataset(X_val,   y_val,   transform=eval_transforms)
    test_dataset  = InMemoryGalaxyDataset(X_test,  y_test,  transform=eval_transforms)

    # ---- Samplers ----
    train_class_counts = np.bincount(y_train)
    class_weights  = 1.0 / train_class_counts
    sample_weights = torch.from_numpy(class_weights[y_train]).float()

    train_sampler = DistributedWeightedSampler(
        weights      = sample_weights,
        num_replicas = world_size,
        rank         = rank,
        num_samples  = len(y_train),
        replacement  = True,
        seed         = SEED
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    pw = NUM_WORKERS > 0
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=pw, drop_last=False
    )

    # ---- Model ----
    model = build_swin(NUM_CLASSES, pretrained=True).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # ---- Dynamic Optimizer ----
    _base = model.module
    head_params = list(_base.head.parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in _base.parameters() if id(p) not in head_param_ids]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * BACKBONE_LR_SCALE},
        {'params': head_params,     'lr': LEARNING_RATE}
    ], weight_decay=0.05)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=4, min_lr=1e-7
    )

    # ---- [FIX 1] Modern Unified PyTorch 2.3+ GradScaler ----
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    criterion_val = nn.CrossEntropyLoss()

    cutmix = v2.CutMix(num_classes=NUM_CLASSES, alpha=CUTMIX_ALPHA)
    mixup  = v2.MixUp(num_classes=NUM_CLASSES,  alpha=MIXUP_ALPHA)

    # ==========================================
    # Training loop
    # ==========================================
    best_val_acc               = 0.0
    epochs_without_improvement = 0
    save_path                  = 'best_galaxy_swint.pth'

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)

        # --- Train ---
        model.train()
        running_loss        = 0.0
        correct_train       = 0
        total_train         = 0
        total_train_samples = 0

        train_bar = tqdm(train_loader, desc=f"[GPU {rank}] Ep {epoch+1}/{EPOCHS} Train",
                         disable=not is_master)
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_n = inputs.size(0)

            r = random.random()
            if r < CUTMIX_PROB:
                inputs, labels_target = cutmix(inputs, labels)
            elif r < CUTMIX_PROB + MIXUP_PROB:
                inputs, labels_target = mixup(inputs, labels)
            else:
                labels_target = labels 

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels_target) 

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * batch_n
            total_train_samples += batch_n
            
            _, predicted = torch.max(outputs, 1)
            total_train += batch_n
            correct_train += (predicted == labels).sum().item()

            if is_master:
                train_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Aggregate train metrics across GPUs
        train_loss_t  = torch.tensor(running_loss,        device=device)
        train_samp_t  = torch.tensor(total_train_samples, device=device)
        train_corr_t  = torch.tensor(correct_train,       device=device)
        train_total_t = torch.tensor(total_train,         device=device)
        dist.all_reduce(train_loss_t,  op=dist.ReduceOp.SUM)
        dist.all_reduce(train_samp_t,  op=dist.ReduceOp.SUM)
        dist.all_reduce(train_corr_t,  op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_t, op=dist.ReduceOp.SUM)

        train_loss = (train_loss_t / train_samp_t).item()
        train_acc  = (train_corr_t / train_total_t).item() if train_total_t.item() > 0 else float('nan')

        # --- Validate ---
        model.eval()
        val_loss_sum   = 0.0
        correct_val    = 0
        total_val_samp = 0
        total_val      = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader,
                                       desc=f"[GPU {rank}] Ep {epoch+1}/{EPOCHS} Val",
                                       disable=not is_master):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_n = inputs.size(0)
                
                with torch.autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = model(inputs)
                    loss    = criterion_val(outputs, labels)

                val_loss_sum   += loss.item() * batch_n
                total_val_samp += batch_n
                _, predicted    = torch.max(outputs, 1)
                total_val      += batch_n
                correct_val    += (predicted == labels).sum().item()

        # Aggregate val metrics
        vl_t  = torch.tensor(val_loss_sum,   device=device)
        vs_t  = torch.tensor(total_val_samp, device=device)
        vc_t  = torch.tensor(correct_val,    device=device)
        vn_t  = torch.tensor(total_val,      device=device)
        dist.all_reduce(vl_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(vs_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(vc_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(vn_t, op=dist.ReduceOp.SUM)

        val_loss = (vl_t / vs_t).item()
        val_acc  = (vc_t / vn_t).item()

        val_acc_t = torch.tensor(val_acc, device=device)
        dist.broadcast(val_acc_t, src=0)
        scheduler.step(val_acc_t.item())

        lrs     = [pg['lr'] for pg in optimizer.param_groups]
        acc_str = f"{train_acc:.4f}" if not math.isnan(train_acc) else "n/a"
        log(f"Ep {epoch+1:02d} | TrLoss: {train_loss:.4f} | TrAcc: {acc_str} | "
            f"VaLoss: {val_loss:.4f} | VaAcc: {val_acc:.4f} | "
            f"LR bb: {lrs[0]:.2e} head: {lrs[1]:.2e}")

        # --- Checkpoint (rank-0 only) ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            if is_master:
                clean = model.module.state_dict()
                torch.save(clean, save_path)
                log(f"  --> Saved best model (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            log(f"  --> No improvement. ES counter: {epochs_without_improvement}/{PATIENCE}")
            if epochs_without_improvement >= PATIENCE:
                log(f"\n[!] Early stopping at epoch {epoch+1}.")
                break

    # ==========================================
    # DDP Cleanup & Standalone TTA Evaluation
    # ==========================================
    # ---- [FIX 2] Explicit device target to drop context warning ----
    dist.barrier(device_ids=[rank])   
    dist.destroy_process_group()

    if is_master:
        log("\n--- Loading best model for TTA evaluation ---")

        best_model = build_swin(NUM_CLASSES, pretrained=False).to(device)
        best_model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )
        best_model.eval()

        tta_base_transforms = v2.Compose([
            v2.ToImage(),
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ])
        tta_gpu_transforms = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomChoice([
                v2.RandomRotation((0,   0)),
                v2.RandomRotation((90,  90)),
                v2.RandomRotation((180, 180)),
                v2.RandomRotation((270, 270))
            ]),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

        tta_dataset = InMemoryGalaxyDataset(X_test, y_test, transform=tta_base_transforms)
        tta_loader  = DataLoader(
            tta_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=pw, drop_last=False
        )

        def predict_tta_gpu(model, base_tensors, n=TTA_N):
            accumulated = None
            for _ in range(n):
                aug = tta_gpu_transforms(base_tensors)
                with torch.no_grad(), torch.autocast(device_type='cuda', enabled=USE_AMP):
                    probs = torch.softmax(model(aug), dim=1)
                accumulated = probs if accumulated is None else accumulated + probs
            return (accumulated / n).argmax(dim=1)

        correct_tta, total_tta = 0, 0
        for base_tensors, labels in tqdm(tta_loader, desc=f"TTA (n={TTA_N})"):
            base_tensors = base_tensors.to(device)
            labels       = labels.to(device)
            predicted    = predict_tta_gpu(best_model, base_tensors)
            total_tta   += labels.size(0)
            correct_tta += (predicted == labels).sum().item()

        log(f"\nFinal Test Accuracy (GPU TTA n={TTA_N}): {correct_tta / total_tta * 100:.2f}%")

        # Baseline (no TTA)
        baseline_loader = DataLoader(
            InMemoryGalaxyDataset(X_test, y_test, transform=eval_transforms),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
            persistent_workers=pw, drop_last=False
        )
        correct_base, total_base = 0, 0
        with torch.no_grad():
            for inputs, labels in baseline_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autocast(device_type='cuda', enabled=USE_AMP):
                    outputs = best_model(inputs)
                _, predicted  = torch.max(outputs, 1)
                total_base   += labels.size(0)
                correct_base += (predicted == labels).sum().item()
        log(f"Final Test Accuracy (no TTA):            {correct_base / total_base * 100:.2f}%")


# ==========================================
# Custom DistributedWeightedSampler
# ==========================================
class DistributedWeightedSampler(torch.utils.data.Sampler):
    def __init__(self, weights, num_replicas, rank, num_samples, replacement=True, seed=0):
        self.weights      = weights
        self.num_replicas = num_replicas
        self.rank         = rank
        self.num_samples  = num_samples
        self.replacement  = replacement
        self.seed         = seed
        self.epoch        = 0
        self.num_samples_per_rank = math.ceil(num_samples / num_replicas)
        self.total_size           = self.num_samples_per_rank * num_replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, self.total_size,
            replacement=self.replacement, generator=g
        ).tolist()
        start = self.rank * self.num_samples_per_rank
        end   = start + self.num_samples_per_rank
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples_per_rank


# ==========================================
# Entry point
# ==========================================
if __name__ == '__main__':
    WORLD_SIZE = torch.cuda.device_count()
    if WORLD_SIZE < 2:
        raise RuntimeError(
            f"Only {WORLD_SIZE} GPU(s) detected. This script requires exactly 2 GPUs. "
            "Enable both GPUs in Kaggle: Settings → Accelerator → GPU T4 x2."
        )
    print(f"Launching DDP training on {WORLD_SIZE} GPUs.")

    mp.spawn(
        main_worker,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True
    )
