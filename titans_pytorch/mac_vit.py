# https://arxiv.org/abs/2510.14657
# but instead of their decorr module updated with SGD, remove all projections and just return a decorrelation auxiliary loss

import torch
from torch import nn, stack, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from copy import deepcopy

from einops import rearrange, repeat, reduce, einsum, pack, unpack
from einops.layers.torch import Rearrange
from titans_pytorch.neural_memory import NeuralMemory

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# decorr loss

class DecorrelationLoss(Module):
    def __init__(
        self,
        sample_frac = 1.,
        soft_validate_num_sampled = False
    ):
        super().__init__()
        assert 0. <= sample_frac <= 1.
        self.need_sample = sample_frac < 1.
        self.sample_frac = sample_frac

        self.soft_validate_num_sampled = soft_validate_num_sampled
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens
    ):
        batch, seq_len, dim, device = *tokens.shape[-3:], tokens.device

        if self.need_sample:
            num_sampled = int(seq_len * self.sample_frac)

            assert self.soft_validate_num_sampled or num_sampled >= 2.

            if num_sampled <= 1:
                return self.zero

            tokens, packed_shape = pack([tokens], '* n d e')

            indices = torch.randn(tokens.shape[:2]).argsort(dim = -1)[..., :num_sampled, :]

            batch_arange = torch.arange(tokens.shape[0], device = tokens.device)
            batch_arange = rearrange(batch_arange, 'b -> b 1')

            tokens = tokens[batch_arange, indices]
            tokens, = unpack(tokens, packed_shape, '* n d e')

        dist = einsum(tokens, tokens, '... n d, ... n e -> ... d e') / tokens.shape[-2]
        eye = torch.eye(dim, device = device)

        loss = dist.pow(2) * (1. - eye) / ((dim - 1) * dim)

        loss = reduce(loss, '... b d e -> b', 'sum')
        return loss.mean()

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.norm(x)
        return self.net(x), normed

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        num_persist_mem_tokens = 0
    ):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.num_persist_mem_tokens = num_persist_mem_tokens
        self.has_persist_mem = num_persist_mem_tokens > 0

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if self.has_persist_mem:
            self.persistent_memory = nn.Parameter(torch.zeros(2, heads, num_persist_mem_tokens, dim_head))
        else:
            self.register_parameter('persistent_memory', None)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        normed = self.norm(x)

        qkv = self.to_qkv(normed).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.has_persist_mem:
            pmk, pmv = repeat(self.persistent_memory, 'kv h n d -> kv b h n d', b = k.shape[0])
            k = torch.cat((pmk, k), dim = -2)
            v = torch.cat((pmv, v), dim = -2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), normed

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):

        normed_inputs = []

        for attn, ff in self.layers:
            attn_out, attn_normed_inp = attn(x)
            x = attn_out + x

            ff_out, ff_normed_inp = ff(x)
            x = ff_out + x

            normed_inputs.append(attn_normed_inp)
            normed_inputs.append(ff_normed_inp)

        return self.norm(x), stack(normed_inputs)

class ViT(Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., decorr_sample_frac = 1.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        # decorrelation loss related

        self.has_decorr_loss = decorr_sample_frac > 0.

        if self.has_decorr_loss:
            self.decorr_loss = DecorrelationLoss(decorr_sample_frac)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        img,
        return_decorr_aux_loss = None
    ):
        return_decorr_aux_loss = default(return_decorr_aux_loss, self.training) and self.has_decorr_loss

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, normed_layer_inputs = self.transformer(x)

        # maybe return decor loss

        decorr_aux_loss = self.zero

        if return_decorr_aux_loss:
            decorr_aux_loss = self.decorr_loss(normed_layer_inputs)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x), decorr_aux_loss


# Memory-augmented Transformer for ViT (per-layer memory like MAC)
class MemoryViTTransformer(Module):
    """
    Transformer with per-layer NeuralMemory modules, similar to MemoryAsContextTransformer.
    Memory is integrated at each layer to retrieve context and gate/augment attention output.
    """
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout = 0.,
        neural_memory_model = None,
        neural_memory_kwargs: dict | None = None,
        neural_memory_chunk_size = 1,
        neural_memory_batch_size = None,
        neural_mem_gate_attn_output = False,
        num_persist_mem_tokens = 0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])
        self.neural_mem_gate_attn_output = neural_mem_gate_attn_output

        nm_kwargs = default(neural_memory_kwargs, {})

        for layer_idx in range(depth):
            is_first = (layer_idx == 0)

            # per-layer memory
            mem = NeuralMemory(
                dim = dim,
                chunk_size = neural_memory_chunk_size,
                batch_size = neural_memory_batch_size,
                model = deepcopy(neural_memory_model),
                **nm_kwargs
            )

            attn = Attention(
                dim,
                heads = heads,
                dim_head = dim_head,
                dropout = dropout,
                num_persist_mem_tokens = num_persist_mem_tokens
            )
            ff = FeedForward(dim, mlp_dim, dropout = dropout)

            self.layers.append(ModuleList([
                mem,
                attn,
                ff,
            ]))

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        x,
        mem_states = None,
        return_mem_states = False,
    ):
        """
        Args:
            x: (batch, seq_len, dim)
            mem_states: list of NeuralMemState for each layer, or None
            return_mem_states: if True, return updated memory states

        Returns:
            x: (batch, seq_len, dim)
            normed_layer_inputs: stack of all layer inputs for decorr loss
            next_mem_states: (optional) list of updated memory states
        """
        if mem_states is None:
            mem_states = [None] * len(self.layers)

        normed_inputs = []
        next_mem_states = []

        for layer_idx, (mem, attn, ff) in enumerate(self.layers):
            mem_state = mem_states[layer_idx]

            # retrieve from memory and gate attention
            retrieved, next_mem_state = mem(x, state = mem_state)

            if self.neural_mem_gate_attn_output:
                # memory gates the attention output
                attn_out_gates = retrieved.sigmoid()
            else:
                # memory is added to residual
                x = x + retrieved

            # attention
            attn_out, attn_normed_inp = attn(x)
            x = attn_out + x

            if self.neural_mem_gate_attn_output and exists(attn_out_gates):
                x = x * attn_out_gates

            # feedforward
            ff_out, ff_normed_inp = ff(x)
            x = ff_out + x

            normed_inputs.append(attn_normed_inp)
            normed_inputs.append(ff_normed_inp)
            next_mem_states.append(next_mem_state)

        x = self.norm(x)

        if return_mem_states:
            return x, stack(normed_inputs), next_mem_states

        return x, stack(normed_inputs)


# Memory-as-Context Vision Transformer
class MemoryViT(Module):
    """
    Vision Transformer with per-layer Neural Memory integration.
    Each transformer layer has its own NeuralMemory that:
    1. Stores patch embeddings as experiences
    2. Retrieves memory context
    3. Gates or augments attention output
    
    Designed for image classification with memory-augmented self-attention.
    """
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        decorr_sample_frac = 1.,
        # memory-specific args
        neural_memory_model = None,
        neural_memory_kwargs: dict | None = None,
        neural_memory_chunk_size = 1,
        neural_memory_batch_size = None,
        neural_mem_gate_attn_output = False,
        num_persist_mem_tokens = 0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # memory-augmented transformer
        self.transformer = MemoryViTTransformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            mlp_dim = mlp_dim,
            dropout = dropout,
            neural_memory_model = neural_memory_model,
            neural_memory_kwargs = neural_memory_kwargs,
            neural_memory_chunk_size = neural_memory_chunk_size,
            neural_memory_batch_size = neural_memory_batch_size,
            neural_mem_gate_attn_output = neural_mem_gate_attn_output,
            num_persist_mem_tokens = num_persist_mem_tokens,
        )

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

        # decorrelation loss
        self.has_decorr_loss = decorr_sample_frac > 0.
        if self.has_decorr_loss:
            self.decorr_loss = DecorrelationLoss(decorr_sample_frac)

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    def forward(
        self,
        img,
        mem_states = None,
        return_mem_states = False,
        return_decorr_aux_loss = None,
    ):
        """
        Args:
            img: (batch, channels, height, width)
            mem_states: list of memory states from previous inference, or None
            return_mem_states: if True, return memory states for next call
            return_decorr_aux_loss: if True, compute and return decorrelation loss

        Returns:
            logits: (batch, num_classes)
            decorr_aux_loss: scalar loss
            mem_states: (optional) list of updated memory states
        """
        return_decorr_aux_loss = default(return_decorr_aux_loss, self.training) and self.has_decorr_loss

        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # add cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # memory-augmented transformer
        if return_mem_states:
            x, normed_layer_inputs, next_mem_states = self.transformer(
                x,
                mem_states = mem_states,
                return_mem_states = True,
            )
        else:
            x, normed_layer_inputs = self.transformer(
                x,
                mem_states = mem_states,
                return_mem_states = False,
            )
            next_mem_states = None

        # compute decorrelation auxiliary loss
        decorr_aux_loss = self.zero
        if return_decorr_aux_loss:
            decorr_aux_loss = self.decorr_loss(normed_layer_inputs)

        # pool: cls token or mean
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        logits = self.mlp_head(x)

        if return_mem_states:
            return logits, decorr_aux_loss, next_mem_states

        return logits, decorr_aux_loss

# quick test

if __name__ == '__main__':
    decorr_loss = DecorrelationLoss(0.1)

    hiddens = torch.randn(6, 2, 512, 256)

    decorr_loss(hiddens)
    decorr_loss(hiddens[0])

    decorr_loss = DecorrelationLoss(0.0001, soft_validate_num_sampled = True)
    out = decorr_loss(hiddens)
    assert out.item() == 0
