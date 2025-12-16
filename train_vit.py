import math
import random
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from titans_pytorch import MemoryAsContextVisionTransformer

DEPTH = 1
NEURAL_MEMORY_DEPTH = 2
EPOCHS = 50
LR = 3e-4
MEM_LR = 3e-4
LOG_FILE = "train_vit_1.log"


def set_deterministic(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def get_dataloaders(data_dir: str = "./data", batch_size: int = 32, seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    set_deterministic(seed)
    transform = transforms.ToTensor()
    train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, generator=g, num_workers=0)
    return train_loader, test_loader

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        if isinstance(output, tuple):
            logits, mem_loss = output
            loss = loss_fn(logits, labels) + mem_loss
        else:
            logits = output
            loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, memory_optimizer: Optional[torch.optim.Optimizer] = None) -> float:
    model.eval()
    correct = 0
    total = 0
    grad_enabled = memory_optimizer is not None
    with torch.set_grad_enabled(grad_enabled):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            if memory_optimizer is not None and isinstance(output, tuple):
                memory_optimizer.zero_grad()
                _, mem_loss = output
                mem_loss.backward()
                memory_optimizer.step()
    return correct / total if total else 0.0

class SimpleTransformerBlockNoMemory(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.ff2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm_attn(x)
        q = self.q_proj(y)
        k = self.k_proj(y)
        v = self.v_proj(y)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn_weights = attn_scores.softmax(dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        x = x + attn_out
        o = self.norm_ff(x)
        o = self.ff2(self.act(self.ff1(o)))
        return x + o


class SimpleCifarTransformer(nn.Module):
    def __init__(self, patch_size: int = 4, dim: int = 256, depth: int = 1):
        super().__init__()
        num_patches = (32 // patch_size) * (32 // patch_size)
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Linear(3 * patch_size * patch_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.blocks = nn.ModuleList([SimpleTransformerBlockNoMemory(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b, -1, c * p * p)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


def build_mac_vit(device: torch.device, neural_memory_depth: int) -> MemoryAsContextVisionTransformer:
    """Helper to build a MAC ViT with a configurable neural memory depth."""
    depth = max(0, min(neural_memory_depth, DEPTH))
    neural_memory_layers = tuple(range(1, depth + 1))

    return MemoryAsContextVisionTransformer(
        num_tokens = 10,
        dim = 256,
        depth = DEPTH,
        segment_len = 128,
        num_persist_mem_tokens = 4,
        num_longterm_mem_tokens = 16,
        image_size = 32,
        patch_size = 4,
        neural_memory_layers = neural_memory_layers,
    ).to(device)

def build_baseline_vit(device: torch.device) -> SimpleCifarTransformer:
    return SimpleCifarTransformer(
        patch_size = 4,
        dim = 256,
        depth = DEPTH,
    ).to(device)

def main() -> None:
    set_deterministic()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_loader, test_loader = get_dataloaders()

    with open(LOG_FILE, 'w') as log_file:

        def log(msg: str) -> None:
            log_file.write(msg + '\n')
            log_file.flush()

        loss_results: list[tuple[str, list[float]]] = []
        accuracy_results: list[tuple[str, list[float]]] = []

        configs = [
            ("vit_no_memory", lambda: build_baseline_vit(device)),
            ("mac_vit_memory", lambda: build_mac_vit(device, NEURAL_MEMORY_DEPTH)),
        ]

        for model_name, model_fn in configs:
            model = model_fn()
            param_count = sum(p.numel() for p in model.parameters())
            log(f"{model_name}: {param_count:,} parameters")

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            memory_optimizer = None  # build_memory_optimizer(model, MEM_LR)

            losses: list[float] = []
            accuracies: list[float] = []

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, train_loader, optimizer, device)
                losses.append(train_loss)

                test_acc = evaluate(model, test_loader, device, memory_optimizer=memory_optimizer)
                accuracies.append(test_acc)

                log(f"{model_name} depth={DEPTH} epoch {epoch}: train_loss={train_loss:.4f}, test_acc={test_acc:.4f}")

            loss_results.append((model_name, losses))
            accuracy_results.append((model_name, accuracies))

        def plot_metric(results: list[tuple[str, list[float]]], ylabel: str, title: str, filename: str) -> None:
            plt.figure(figsize=(6, 4))
            for name, values in results:
                plt.plot(range(1, len(values) + 1), values, label=name)
            plt.xlabel("epoch")
            plt.ylabel(ylabel)
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(filename)
            log(f"Saved {ylabel} plot to {filename}")

        plot_metric(loss_results, "train loss", f"Train loss (depth={DEPTH})", f"loss_{DEPTH}.png")
        plot_metric(accuracy_results, "accuracy", f"Test accuracy (depth={DEPTH})", f"accuracy_{DEPTH}.png")


if __name__ == "__main__":
    main()
