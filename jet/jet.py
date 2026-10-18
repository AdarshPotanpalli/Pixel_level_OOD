from contextlib import contextmanager
from typing import Optional
from typing import Tuple

import einops
import torch as t
import torch.nn as nn
import torch.nn.init as init
from torch import autocast


def get_channel_coupling(layers, channels):
    w = t.zeros((layers, channels, channels))
    generator = t.Generator()
    generator.manual_seed(0)

    for i in range(layers):
        p = t.randperm(channels, generator=generator)
        row_idx = t.ones(channels, dtype=t.long) * i
        col_idx = t.arange(channels)
        w[row_idx, p, col_idx] = 1.0
    return w


def get_spatial_coupling(layers, patch_size, width, height, proj_coupling_types):
    nh = height // patch_size
    nw = width // patch_size
    n = nh * nw
    w = t.zeros((layers, n, n))

    for i in range(layers):
        if i >= len(proj_coupling_types):
            kind = "checkerboard"
        else:
            kind = proj_coupling_types[i]

        if kind.startswith("vstripes"):
            idx1 = t.arange(0, n, 2)
            idx2 = t.arange(1, n, 2)
        elif kind.startswith("hstripes"):
            idx1 = t.nonzero((t.arange(n) // nw) % 2 == 0, as_tuple=True)[0]
            idx2 = t.nonzero((t.arange(n) // nw) % 2 == 1, as_tuple=True)[0]
        elif kind.startswith("checkerboard"):
            vals = t.arange(n).reshape(nh, nw) + t.arange(nh).reshape(nh, 1)
            idx1 = t.nonzero((vals.flatten() % 2) == 0, as_tuple=True)[0]
            idx2 = t.nonzero((vals.flatten() % 2) == 1, as_tuple=True)[0]

        if kind.endswith("-inv"):
            idx1, idx2 = idx2, idx1

        w[i, idx1, t.arange(n//2)] = 1
        w[i, idx2, t.arange(n//2, n)] = 1
    return w


@contextmanager
def precision(precision):
    prev = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision(precision)
        yield
    finally:
        t.set_float32_matmul_precision(prev)


class Jet(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        patch_dim: int = 48,
        n_patches: int = 256,
        img_dims: Tuple[int, int] = (64, 64),
        coupling_layers: int = 32,
        block_depth: int = 2,
        block_width: int = 512,
        num_heads: int = 8,
        scale_factor: float = 2.0,
        coupling_types: Tuple[str] = (
            "channels", "channels",
            "channels", "channels",
            "spatial",
        ),
        spatial_coupling_projs: Tuple[str] = (
            "checkerboard", "checkerboard-inv",
            "vstripes", "vstripes-inv",
            "hstripes", "hstripes-inv",
        )
    ):
        super().__init__()
        self.patch_size = patch_size
        self.img_height, self.img_width = img_dims
        self.coupling_layers = coupling_layers

        coupling_types *= (coupling_layers // len(coupling_types) + 1)
        self.coupling_types = coupling_types[:coupling_layers]

        self.blocks = nn.ModuleList(
            [
                Coupling(
                    patch_dim=patch_dim,
                    depth=block_depth,
                    n_patches=n_patches,
                    width=block_width,
                    num_heads=num_heads,
                    scale_factor=scale_factor,
                )
                for _ in range(coupling_layers)
            ]
        )

        channels_indices = [i for i, k in enumerate(self.coupling_types) if k == "channels"]
        spatial_indices = [i for i, k in enumerate(self.coupling_types) if k == "spatial"]

        n_channels = len(channels_indices)
        n_spatial = len(spatial_indices)

        channel_projs = get_channel_coupling(n_channels, patch_dim)
        spatial_projs = get_spatial_coupling(
            n_spatial,
            patch_size,
            self.img_height,
            self.img_width,
            spatial_coupling_projs,
        )

        self.register_buffer("channel_projections", channel_projs)
        self.register_buffer("spatial_projections", spatial_projs)

        channel_proj_map = {i: idx for idx, i in enumerate(channels_indices)}
        spatial_proj_map = {i: idx for idx, i in enumerate(spatial_indices)}

        self.channel_indices = [channel_proj_map.get(i, -1) for i in range(coupling_layers)]
        self.spatial_indices = [spatial_proj_map.get(i, -1) for i in range(coupling_layers)]
        self.kind_flags = [k == "channels" for k in self.coupling_types]

    def patchify(self, x: t.Tensor):
        return einops.rearrange(
            x,
            "b (h hp) (w wp) c -> b (h w) (hp wp c)",
            hp=self.patch_size,
            wp=self.patch_size,
        )

    def unpatchify(self, x: t.Tensor):
        return einops.rearrange(
            x,
            "b (h w) (hp wp c) -> b (h hp) (w wp) c",
            h=self.img_height // self.patch_size,
            w=self.img_width // self.patch_size,
            hp=self.patch_size,
            wp=self.patch_size,
        )

    def forward(self, x: t.Tensor):
        x = self.patchify(x)
        logdet = t.zeros((x.shape[0], 1), device=x.device)

        for i in range(len(self.blocks)):
            block = self.blocks[i]
            is_channel = self.kind_flags[i]
            channel_idx = self.channel_indices[i]
            spatial_idx = self.spatial_indices[i]

            channel_proj = self.channel_projections[channel_idx] if channel_idx >= 0 else None
            spatial_proj = self.spatial_projections[spatial_idx] if spatial_idx >= 0 else None

            x, layer_logdet = block(x, is_channel, channel_proj, spatial_proj)
            logdet += layer_logdet

        return x, logdet

    def inverse(self, x: t.Tensor):
        logdet = t.zeros((x.shape[0], 1), device=x.device)

        for i in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[i]
            is_channel = self.kind_flags[i]

            channel_idx = self.channel_indices[i]
            spatial_idx = self.spatial_indices[i]

            channel_proj = self.channel_projections[channel_idx] if channel_idx >= 0 else None
            spatial_proj = self.spatial_projections[spatial_idx] if spatial_idx >= 0 else None

            x, layer_logdet = block.inverse(x, is_channel, channel_proj, spatial_proj)
            logdet += layer_logdet

        return self.unpatchify(x), logdet


class Coupling(nn.Module):
    def __init__(
        self,
        patch_dim: int = 48,
        depth: int = 1,
        n_patches: int = 256,
        width: int = 256,
        num_heads: int = 4,
        scale_factor: float = 2.0,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.dnn = DNN(
            out_dim=patch_dim,
            n_patches=n_patches,
            depth=depth,
            width=width,
            num_heads=num_heads,
        )

    @autocast(enabled=False, device_type="cuda")
    def _split_channels(
        self, x: t.Tensor, channel_proj: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        with precision("highest"):
            x = t.einsum("ntk,km->ntm", x, channel_proj)
        x1, x2 = t.chunk(x, 2, dim=-1)
        return x1, x2

    @autocast(enabled=False, device_type="cuda")
    def _merge_channels(
        self, x1: t.Tensor, x2: t.Tensor, channel_proj: t.Tensor
    ) -> t.Tensor:
        x = t.cat([x1, x2], dim=-1)
        with precision("highest"):
            x = t.einsum("ntk,km->ntm", x, channel_proj.T)
        return x

    @autocast(enabled=False, device_type="cuda")
    def _split_spatial(
        self, x: t.Tensor, spatial_proj: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        with precision("highest"):
            x = t.einsum("ntk,tm->nmk", x, spatial_proj)
        x1, x2 = t.chunk(x, 2, dim=-2)

        x1 = einops.rearrange(x1, "... n (s c) -> ... (n s) c", s=2)
        x2 = einops.rearrange(x2, "... n (s c) -> ... (n s) c", s=2)
        return x1, x2

    @autocast(enabled=False, device_type="cuda")
    def _merge_spatial(
        self, x1: t.Tensor, x2: t.Tensor, spatial_proj: t.Tensor
    ) -> t.Tensor:
        x1 = einops.rearrange(x1, "... (n s) c -> ... n (s c)", s=2)
        x2 = einops.rearrange(x2, "... (n s) c -> ... n (s c)", s=2)

        x = t.cat([x1, x2], dim=-2)
        with precision("highest"):
            x = t.einsum("ntk,tm->nmk", x, spatial_proj.T)
        return x

    def _setup(
        self,
        x: t.Tensor,
        is_channel: bool,
        channel_proj: t.Tensor,
        spatial_proj: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        if is_channel:
            x1, x2 = self._split_channels(x, channel_proj)
        else:
            x1, x2 = self._split_spatial(x, spatial_proj)

        bias, raw_scale = self.dnn(x1)
        scale = t.sigmoid(raw_scale) * self.scale_factor
        logdet = t.nn.functional.logsigmoid(raw_scale) + t.log(
            t.tensor(self.scale_factor, device=x.device)
        )
        logdet = logdet.sum(dim=tuple(range(1, logdet.dim()))).unsqueeze(-1)
        return x1, x2, bias, scale, logdet

    def forward(
        self,
        x: t.Tensor,
        is_channel: bool,
        channel_proj: t.Tensor,
        spatial_proj: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        x1, x2, bias, scale, logdet = self._setup(x, is_channel, channel_proj, spatial_proj)
        x2 = (x2 + bias) * scale

        if is_channel:
            x = self._merge_channels(x1, x2, channel_proj)
        else:
            x = self._merge_spatial(x1, x2, spatial_proj)

        return x, logdet

    def inverse(
        self,
        x: t.Tensor,
        is_channel: bool,
        channel_proj: t.Tensor,
        spatial_proj: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        x1, x2, bias, scale, logdet = self._setup(x, is_channel, channel_proj, spatial_proj)
        x2 = (x2 / scale) - bias

        if is_channel:
            x = self._merge_channels(x1, x2, channel_proj)
        else:
            x = self._merge_spatial(x1, x2, spatial_proj)

        return x, -logdet


class DNN(nn.Module):
    def __init__(
        self,
        out_dim: int = 32,
        depth: int = 1,
        n_patches: int = 256,
        width: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.depth = depth
        self.width = width
        self.num_heads = num_heads
        self.init_proj = nn.Linear(out_dim // 2, width)

        self.pos_emb = nn.Parameter(t.empty(1, n_patches, width, dtype=t.float32))
        init.normal_(self.pos_emb, mean=0.0, std=1 / width ** 0.5)

        self.encoder = Encoder(width, depth, num_heads=num_heads)
        self.final_proj = nn.Linear(width, out_dim)

        init.zeros_(self.final_proj.weight)
        init.zeros_(self.final_proj.bias)

    def forward(self, x: t.Tensor):
        x = self.init_proj(x) + self.pos_emb
        x = self.encoder(x)
        x = self.final_proj(x)
        bias, scale = t.chunk(x, 2, dim=-1)
        return bias, scale


class Encoder(nn.Module):
    def __init__(
        self,
        width: int,
        depth: int,
        mlp_dim: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth

        layers = []
        for _ in range(self.depth):
            layers.append(Encoder1DBlock(width, mlp_dim, num_heads, dropout))

        self.blocks = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(width)

    def forward(self, x: t.Tensor):
        x = self.blocks(x)
        return self.ln(x)


class Encoder1DBlock(nn.Module):
    def __init__(
        self,
        width: int,
        mlp_dim: Optional[int] = None,
        num_heads: int = 12,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.mha = nn.MultiheadAttention(
            width,
            num_heads=num_heads,
            batch_first=True,
        )
        init.xavier_uniform_(self.mha.in_proj_weight)
        init.xavier_uniform_(self.mha.out_proj.weight)

        self.ln2 = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(width, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x: t.Tensor):
        y = self.ln1(x)
        y, _ = self.mha(y, y, y, need_weights=False)
        y = self.dropout(y)
        x = x + y
        y = self.ln2(x)
        y = self.mlp(y)
        y = self.dropout(y)
        return x + y


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, mlp_dim or 4 * input_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(mlp_dim or 4 * input_dim, input_dim)

        init.xavier_uniform_(self.in_proj.weight)
        init.xavier_uniform_(self.out_proj.weight)
        init.normal_(self.in_proj.bias, std=1e-6)
        init.normal_(self.out_proj.bias, std=1e-6)

    def forward(self, x: t.Tensor):
        x = self.in_proj(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.out_proj(x)