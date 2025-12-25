import argparse
import math
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from titans_pytorch.mac_vit import MemoryViT
from titans_pytorch.memory_models import MemoryMLP

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MemoryViT on ImageNet-1K.")
    parser.add_argument("--train-dir", type=Path, help="Path to ImageNet-1K training root (class subfolders).")
    parser.add_argument("--val-dir", type=Path, help="Path to ImageNet-1K validation root (class subfolders).")
    parser.add_argument("--use-hf-dataset", action="store_true", help="Load ImageNet-1K from huggingface datasets.")
    parser.add_argument("--hf-cache-dir", type=Path, help="Optional cache directory for huggingface datasets.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate for AdamW.")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--memory-lr", type=float, default=0.0, help="If >0, use a separate optimizer to update memory-only params from aux loss during eval.")
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--max-steps-per-epoch", type=int, help="Optional limit on steps for quick debugging.")
    parser.add_argument("--decorr-weight", type=float, default=0.1, help="Weight for decorrelation auxiliary loss from memory.")
    parser.add_argument("--use-amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/memory_vit_imagenet1k"))
    parser.add_argument("--save-every", type=int, default=0, help="If >0, save a checkpoint every N epochs.")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--mlp-dim", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--memory-depth", type=int, default=2, help="Depth for the NeuralMemory MLP.")
    parser.add_argument("--memory-expansion", type=float, default=4.0, help="Expansion factor for the NeuralMemory MLP.")
    parser.add_argument("--memory-chunk-size", type=int, default=4, help="Chunk size used inside NeuralMemory.")
    parser.add_argument("--memory-batch-size", type=int, help="Optional memory batch size for NeuralMemory.")
    parser.add_argument("--memory-gate-attn", action="store_true", help="Use memory output to gate attention instead of residual add.")
    parser.add_argument("--num-persist-mem-tokens", type=int, default=4, help="Persistent memory tokens appended to key/value.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", default="memory-vit-imagenet1k")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms


def _build_hf_dataloaders(args: argparse.Namespace, train_tfms, val_tfms) -> tuple[DataLoader, DataLoader]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("huggingface datasets is not installed; run `pip install datasets` or disable --use-hf-dataset.") from exc

    def transform_with(tfms):
        def _apply(batch):
            images = batch["image"]
            if isinstance(images, list):
                pixels = [tfms(img.convert("RGB") if hasattr(img, "convert") else img) for img in images]
            else:
                pixels = tfms(images.convert("RGB") if hasattr(images, "convert") else images)
            return {"pixel_values": pixels, "label": batch["label"]}
        return _apply

    def collate_fn(batch):
        if isinstance(batch, dict):
            images = batch["pixel_values"]
            labels = batch["label"]
        else:
            images = [b["pixel_values"] for b in batch]
            labels = [b["label"] for b in batch]

        if isinstance(images, list):
            images = torch.stack([img if isinstance(img, torch.Tensor) else torch.tensor(img) for img in images])
        elif isinstance(images, torch.Tensor) and images.ndim == 3:
            images = images.unsqueeze(0)

        labels = torch.tensor(labels, dtype=torch.long) if not torch.is_tensor(labels) else labels
        if labels.ndim > 1:
            labels = labels.squeeze()
        return images, labels

    train_ds = load_dataset("ILSVRC/imagenet-1k", split="train", cache_dir=args.hf_cache_dir)
    val_ds = load_dataset("ILSVRC/imagenet-1k", split="validation", cache_dir=args.hf_cache_dir)
    train_ds = train_ds.with_transform(transform_with(train_tfms))
    val_ds = val_ds.with_transform(transform_with(val_tfms))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_tfms, val_tfms = build_transforms(args.image_size)

    if args.use_hf_dataset:
        return _build_hf_dataloaders(args, train_tfms, val_tfms)

    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_tfms)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    return train_loader, val_loader


def build_model(num_classes: int, args: argparse.Namespace) -> MemoryViT:
    memory_model = MemoryMLP(
        dim=args.dim_head,
        depth=args.memory_depth,
        expansion_factor=args.memory_expansion,
    )

    model = MemoryViT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        neural_memory_model=memory_model,
        neural_memory_kwargs={
            "dim_head": args.dim_head,
            "heads": args.heads,
        },
        neural_memory_chunk_size=args.memory_chunk_size,
        neural_memory_batch_size=args.memory_batch_size,
        neural_mem_gate_attn_output=args.memory_gate_attn,
        num_persist_mem_tokens=args.num_persist_mem_tokens,
    )
    return model


def build_memory_optimizer(model: nn.Module, memory_lr: float) -> Optional[torch.optim.Optimizer]:
    if memory_lr <= 0:
        return None
    memory_module = getattr(model, "neural_memory_model", None)
    if memory_module is None:
        return None
    return torch.optim.Adam(memory_module.parameters(), lr=memory_lr)


def init_wandb(args: argparse.Namespace):
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("Weights & Biases requested but not installed. Run `pip install wandb`.") from exc

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        entity=args.wandb_entity,
        config=vars(args),
    )
    return wandb


def unpack_logits(output) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(output, tuple):
        logits = output[0]
        aux_loss = output[1] if len(output) > 1 else None
    else:
        logits, aux_loss = output, None
    return logits, aux_loss


def forward_with_loss(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
    decorr_weight: float,
    amp: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    with autocast(device_type=device.type, enabled=amp):
        output = model(images)
        logits, mem_loss = unpack_logits(output)
        loss = loss_fn(logits, targets)
        if mem_loss is not None and torch.is_tensor(mem_loss):
            loss = loss + decorr_weight * mem_loss
    return loss, logits, mem_loss


def run_sanity_check(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    loss_fn: nn.Module,
    decorr_weight: float,
    batches: int,
    amp: bool,
    wandb_run,
    memory_optimizer: Optional[torch.optim.Optimizer],
) -> None:
    model.train()
    steps = 0
    total_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, _, mem_loss = forward_with_loss(model, images, targets, loss_fn, decorr_weight, amp, device)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        steps += 1
        if wandb_run:
            wandb_run.log({"sanity/loss": loss.item(), "sanity/mem_loss": float(mem_loss.detach().item()) if mem_loss is not None else 0.0})
        if memory_optimizer is not None and mem_loss is not None and torch.is_tensor(mem_loss):
            memory_optimizer.zero_grad()
            (decorr_weight * mem_loss).backward()
            memory_optimizer.step()
        if steps >= batches:
            break
    avg_loss = total_loss / max(steps, 1)
    print(f"[sanity-check] ran {steps} batches, avg loss={avg_loss:.4f}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    loss_fn: nn.Module,
    decorr_weight: float,
    amp: bool,
    log_interval: int,
    max_steps: Optional[int],
    wandb_run,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for step, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        loss, _, mem_loss = forward_with_loss(model, images, targets, loss_fn, decorr_weight, amp, device)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        if step % log_interval == 0:
            print(f"  step {step:05d} loss={loss.item():.4f} mem_loss={float(mem_loss.detach().item()) if mem_loss is not None else 0.0:.4f}")
            if wandb_run:
                wandb_run.log({
                    "train/step_loss": loss.item(),
                    "train/step_mem_loss": float(mem_loss.detach().item()) if mem_loss is not None else 0.0,
                })

        if max_steps and step >= max_steps:
            break

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    loss_fn: nn.Module,
    decorr_weight: float,
    amp: bool,
    memory_optimizer: Optional[torch.optim.Optimizer],
) -> tuple[float, float]:
    if loader is None:
        return 0.0, 0.0

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    grad_enabled = memory_optimizer is not None
    with torch.set_grad_enabled(grad_enabled):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=amp):
                output = model(images)
                logits, mem_loss = unpack_logits(output)
                loss = loss_fn(logits, targets)
                if mem_loss is not None and torch.is_tensor(mem_loss):
                    loss = loss + decorr_weight * mem_loss

            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if memory_optimizer is not None and mem_loss is not None and torch.is_tensor(mem_loss):
                memory_optimizer.zero_grad()
                (decorr_weight * mem_loss).backward()
                memory_optimizer.step()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    best_val_acc: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not args.use_hf_dataset and (args.train_dir is None or args.val_dir is None):
        raise ValueError("When not using --use-hf-dataset, both --train-dir and --val-dir must be provided.")

    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args.num_classes, args).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    memory_optimizer = build_memory_optimizer(model, args.memory_lr)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")
    wandb_run = init_wandb(args)
    if wandb_run:
        wandb_run.watch(model, log_freq=100)

    print("Running sanity check on a couple of batches...")
    run_sanity_check(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        loss_fn=loss_fn,
        decorr_weight=args.decorr_weight,
        batches=2,
        amp=args.use_amp,
        wandb_run=wandb_run,
        memory_optimizer=memory_optimizer,
    )

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1:03d}/{args.epochs} | lr={current_lr:.6f}")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            loss_fn=loss_fn,
            decorr_weight=args.decorr_weight,
            amp=args.use_amp,
            log_interval=50,
            max_steps=args.max_steps_per_epoch,
            wandb_run=wandb_run,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            loss_fn=loss_fn,
            decorr_weight=args.decorr_weight,
            amp=args.use_amp,
            memory_optimizer=memory_optimizer,
        )

        print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if wandb_run:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, scaler, epoch + 1, best_val_acc)

        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(args.output_dir / f"epoch_{epoch + 1}.pt", model, optimizer, scaler, epoch + 1, best_val_acc)

    save_checkpoint(args.output_dir / "last.pt", model, optimizer, scaler, args.epochs, best_val_acc)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
