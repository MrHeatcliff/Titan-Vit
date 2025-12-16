# Titan-ViT
**Vision Transformer with Neural Memory (Memory-as-Context)**

This repository contains a minimal but explicit implementation of a Vision Transformer (ViT)
augmented with **Neural Memory**, following the *Memory-as-Context* paradigm.

The code is designed for **research and experimentation**, focusing on understanding how
neural memory depth interacts with transformer depth and training dynamics.

---

## 1. Overview
- Backbone: Vision Transformer (ViT)
- Memory mechanism: Neural Memory injected at configurable layers
- Training setup: Single-node, single-GPU (by default)
- Purpose: Research, ablation, and analysis — **not production-ready**

Most of the source code is adapt from https://github.com/lucidrains/titans-pytorch

---

## 2. Installation

Python ≥ 3.12 is recommended.

Install dependencies defined in `pyproject.toml`:

```bash
pip install -e .
```
## 3. Configuration
All experiment-level configurations are defined directly inside `train_vit.py`
```python
DEPTH = 1                  # Number of ViT layers
NEURAL_MEMORY_DEPTH = 2    # Number of layers equipped with neural memory
EPOCHS = 50                # Training epochs
LR = 3e-4                  # Learning rate for ViT backbone
MEM_LR = 3e-4              # Learning rate for neural memory
LOG_FILE = "train_vit_1.log"

```

## 4. Training

I using `tmux` to run on the server:

``` bash
tmux new -s vit_train
python train_vit.py 
```

Detach safely with:
``` bash
Ctrl + b, d
```

## 5. Log 
You can inspect log file:
``` bash
tail -f train_vit_1.log     # Realtime monitoring
less train_vit_1.log       # Inspect full log
tail -n 100 train_vit_1.log # View last 100 lines
```