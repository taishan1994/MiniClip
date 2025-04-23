import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from typing import Optional, Callable, Union, Tuple
from contextlib import nullcontext
from dataclasses import dataclass
from utils import freeze_batch_norm_2d, to_2tuple
from transform import image_transform
from tokenizer import tokenize


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, hidden_z=None):
        '''
        x: (N, L, C)
        hidden_z: (C,)
        '''
        self.hidden_z = hidden_z
        orig_type = x.dtype

        if hidden_z is None:
            x = F.layer_norm(x, self.normalized_shape,
                             self.weight, self.bias, self.eps)
        else:
            assert len(self.normalized_shape) == 1
            # [TODO] weighted layer norm
            remaining_index = torch.where(hidden_z != 0)[0]
            compressed_input = torch.index_select(
                x, dim=-1, index=remaining_index)
            compressed_weight = self.weight[remaining_index]
            compressed_bias = self.bias[remaining_index]
            normalized_shape = len(remaining_index)
            normed_input = F.layer_norm(
                compressed_input, [normalized_shape], compressed_weight, compressed_bias, self.eps)
            x = x.new_zeros(x.shape)
            x[..., remaining_index] = normed_input.to(orig_type)

        return x.to(orig_type)

    def prune(self):
        if self.hidden_z is None:
            return self
        hidden_z = self.hidden_z
        assert len(self.normalized_shape) == 1
        remaining_index = torch.where(hidden_z != 0)[0]
        compressed_weight = self.weight[remaining_index]
        compressed_bias = self.bias[remaining_index]
        # m = self
        m = LayerNorm(remaining_index.shape[0]).to(self.weight.device)
        m.normalized_shape = (len(remaining_index),)
        m.weight.data = compressed_weight.contiguous()
        m.bias.data = compressed_bias.contiguous()
        return m

    def prune_mul_hidden(self):
        if self.hidden_z is None:
            return self
        hidden_z = self.hidden_z
        assert len(self.normalized_shape) == 1
        remaining_index = torch.where(hidden_z != 0)[0]
        compressed_weight = self.weight[remaining_index] * \
                            hidden_z[remaining_index]
        compressed_bias = self.bias[remaining_index] * \
                          hidden_z[remaining_index]
        m = self
        m.normalized_shape = (len(remaining_index),)
        m.weight.data = compressed_weight.contiguous()
        m.bias.data = compressed_bias.contiguous()
        return m


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Mlp(nn.Module):
    def __init__(self, d_model, mlp_width, act_layer=nn.GELU, scale_fc=False):
        super().__init__()
        self.d_model = d_model
        self.mlp_width = mlp_width
        self.c_fc = nn.Linear(d_model, mlp_width)
        assert not scale_fc
        # self.ln = LayerNorm(mlp_width) if scale_fc else nn.Identity()
        self.act_layer = act_layer
        self.scale_fc = scale_fc
        self.gelu = act_layer()
        self.c_proj = nn.Linear(mlp_width, d_model)

    def forward(self, x, hidden_z=None, intermediate_z=None):
        '''
        x: (N, L, C)
        intermediate_z: (mlp_width,) or (1, 1, mlp_width)
        hidden_z: (embed_dim,) or (1, 1, embed_dim)
        '''
        self.hidden_z = hidden_z
        self.intermediate_z = intermediate_z

        x = self.c_fc(x)
        x = self.gelu(x)
        if intermediate_z is not None:
            x = torch.mul(x, intermediate_z)
        x = self.c_proj(x)
        if hidden_z is not None:
            x = torch.mul(x, hidden_z)
        return x

    def prune(self):
        device = self.c_fc.weight.device
        if self.hidden_z is None:
            self.hidden_z = torch.ones(
                (self.d_model,), dtype=torch.bool, device=device)
        if self.intermediate_z is None:
            self.intermediate_z = torch.ones(
                (self.mlp_width,), dtype=torch.bool, device=device)
        hidden_r = torch.where(self.hidden_z != 0)[0]
        intermediate_r = torch.where(self.intermediate_z != 0)[0]
        d_model = len(hidden_r)
        mlp_width = len(intermediate_r)
        # m = self
        m = copy.deepcopy(self)
        m.c_fc = nn.Linear(hidden_r.shape[0], intermediate_r.shape[0])
        m.c_proj = nn.Linear(intermediate_r.shape[0], hidden_r.shape[0])
        m.d_model = d_model
        m.mlp_width = mlp_width
        m.c_fc.weight = nn.Parameter(
            (self.c_fc.weight[intermediate_r][:, hidden_r]).contiguous())
        m.c_fc.bias = nn.Parameter(
            (self.c_fc.bias[intermediate_r]).contiguous())

        m.c_proj.weight = nn.Parameter(((self.c_proj.weight *
                                         self.intermediate_z.view(1, -1) * self.hidden_z.view(-1, 1))[hidden_r][:,
                                        intermediate_r]).contiguous())
        m.c_proj.bias = nn.Parameter(
            ((self.c_proj.bias * self.hidden_z)[hidden_r]).contiguous())
        return m


class MultiheadAttention(nn.MultiheadAttention):
    def prune(self):
        device = self.in_proj_weight.device
        if self.hidden_z is None:
            self.hidden_z = torch.ones(
                (self.embed_dim,), dtype=torch.bool, device=device)
        if self.head_z is None:
            self.head_z = torch.ones(
                (self.num_heads,), dtype=torch.bool, device=device)
        hidden_r = torch.where(self.hidden_z != 0)[0]
        head_r = torch.where(self.head_z != 0)[0]
        d_model = len(hidden_r)
        d_head = len(head_r)
        org_num_heads = self.num_heads
        org_head_dim = self.head_dim
        org_embed_dim = self.embed_dim
        mod = self
        mod.use_naive_compute = True
        mod.embed_dim = d_model
        mod.head_dim = self.head_dim
        mod.num_heads = d_head
        inter_dim = d_head * self.head_dim
        mod.in_proj_weight = nn.Parameter(self.in_proj_weight.view(
            3, org_num_heads, org_head_dim, org_embed_dim)[:, head_r][..., hidden_r].reshape(-1, d_model))
        if self.in_proj_bias is not None:
            mod.in_proj_bias = nn.Parameter(self.in_proj_bias.view(
                3, org_num_heads, org_head_dim)[:, head_r].reshape(-1))
        mod.out_proj.weight = nn.Parameter(
            ((self.out_proj.weight * self.hidden_z.view(-1, 1)).
             view(org_embed_dim, org_num_heads, org_head_dim) * self.head_z.view(1, org_num_heads, 1))[hidden_r][:,
            head_r].reshape(d_model, -1)
        )
        if self.out_proj.bias is not None:
            mod.out_proj.bias = nn.Parameter(
                (self.out_proj.bias * self.hidden_z.view(-1, )).
                    view(org_embed_dim)[hidden_r].reshape(-1)
            )
        return mod


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = MultiheadAttention(d_model, n_head)
        assert not scale_attn
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = Mlp(d_model, mlp_width, act_layer, scale_fc)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                  *,
                  head_z: Optional[torch.Tensor] = None,
                  hidden_z: Optional[torch.Tensor] = None,
                  ):

        self.attn.head_z = head_z
        self.attn.hidden_z = hidden_z

        if (head_z is None and hidden_z is None and
                not getattr(self.attn, 'use_naive_compute', False)):
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        else:
            # the following code does not support `attn_mask`
            # x: (length, batch_size, embed_dim)
            n_head = self.attn.num_heads
            length, batch_size, d_model = x.shape
            ws = self.attn.in_proj_weight.chunk(3)
            bs = self.attn.in_proj_bias.chunk(3)
            dim_per_head = len(ws[0]) // n_head
            # (length, batch_size, n_head * dim_per_head)
            q, k, v = [F.linear(x, w, b) for w, b in zip(ws, bs)]
            # (batch_size * n_head, length, d_head)
            q = q.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            k = k.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            v = v.reshape(length, batch_size * n_head, -1).transpose(0, 1)
            scale = dim_per_head ** -0.5
            q *= scale
            # (batch_size * n_head, length, length)
            sim = q @ k.transpose(1, 2)
            if attn_mask is not None:
                sim += attn_mask
            sim = torch.softmax(sim, -1)
            # (batch_size * n_head, length, head_dim)
            out = sim @ v
            if head_z is not None:
                out = out.view(batch_size, n_head, length, dim_per_head)
                # head_z: (1, n_head, 1, 1)
                out *= head_z.view(1, -1, 1, 1)
                out = out.view(batch_size * n_head, length, dim_per_head)
            out = out.transpose(0, 1).reshape(length, batch_size, -1)
            out = F.linear(out, self.attn.out_proj.weight,
                           self.attn.out_proj.bias)
            if hidden_z is not None:
                out = torch.mul(out, hidden_z)
            return out

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None):

        self.hidden_z = hidden_z
        self.heads_z = heads_z
        self.mha_z = mha_z
        self.intermediate_z = intermediate_z
        self.ffn_z = ffn_z

        # x: (length, batch_size, embed_dim) e.g. 50, 128, 768 for vision
        if self.attention is not None:
            attn_out = self.attention(self.ln_1(x, hidden_z=hidden_z),
                                      attn_mask=attn_mask,
                                      head_z=heads_z, hidden_z=hidden_z)
            if mha_z is not None:  # a number
                attn_out = attn_out.mul(mha_z)
            x = x + attn_out
        if self.mlp is not None:
            ln_2_out = self.ln_2(x, hidden_z=hidden_z)

            mlp_out = self.mlp(ln_2_out,
                               intermediate_z=intermediate_z,
                               hidden_z=hidden_z)
            if ffn_z is not None:  # a number
                mlp_out = mlp_out.mul(ffn_z)
            x = x + mlp_out
        return x

    def prune(self):
        mod = self
        if (self.mha_z is not None and self.mha_z.item() == 0) or (self.heads_z).sum() == 0:
            mod.ln_1 = None
            mod.attn = None
            mod.attention = None
        else:
            mod.ln_1 = mod.ln_1.prune()
            mod.attn = mod.attn.prune()
            if self.mha_z is not None:
                mod.attn.out_proj.weight.data *= self.mha_z
                mod.attn.out_proj.bias.data *= self.mha_z

        if self.ffn_z is not None and self.ffn_z.item() == 0:
            mod.ln_2 = None
            mod.mlp = None
        else:
            mod.ln_2 = mod.ln_2.prune()
            mod.mlp = mod.mlp.prune()
            if self.ffn_z is not None:
                mod.mlp.c_proj.weight.data *= self.ffn_z
                mod.mlp.c_proj.bias.data *= self.ffn_z
        return mod


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        assert width % heads == 0
        self.head_dim = width // heads
        self.num_heads = heads
        self.mlp_ratio = mlp_ratio

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None):

        return self.infer_blocks(x, attn_mask,
                                 hidden_z=hidden_z,
                                 heads_z=heads_z,
                                 mha_z=mha_z,
                                 intermediate_z=intermediate_z,
                                 ffn_z=ffn_z)

    def infer_blocks(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, block_idxs=None,
                     hidden_z: Optional[torch.Tensor] = None,
                     heads_z: Optional[torch.Tensor] = None,
                     mha_z: Optional[torch.Tensor] = None,
                     intermediate_z: Optional[torch.Tensor] = None,
                     ffn_z: Optional[torch.Tensor] = None):

        num_layers = self.layers
        if hidden_z is not None:
            assert hidden_z.shape == (self.width,)
        if heads_z is not None:
            if heads_z.ndim == 5:
                heads_z = heads_z.view(num_layers, self.num_heads)
            assert heads_z.shape in [(num_layers, self.num_heads), (self.num_heads,)], (
                heads_z.shape, (num_layers, self.num_heads))
        if mha_z is not None:
            assert mha_z.shape == (num_layers,), mha_z.shape
        if intermediate_z is not None:
            if intermediate_z.ndim == 4:
                intermediate_z = intermediate_z.view(num_layers, -1)
            assert intermediate_z.shape in [
                (num_layers, self.mlp_ratio * self.width), (self.mlp_ratio * self.width,)], intermediate_z.shape
        if ffn_z is not None:
            assert ffn_z.shape == (num_layers,), ffn_z.shape

        def _get_zi(z, i, ndim=2):
            if z is None:
                return None
            if z.ndim == ndim:
                return z[i]
            return z

        block_idxs = block_idxs or list(range(self.layers))
        for i in block_idxs:
            r = self.resblocks[i]

            x = r(x, attn_mask=attn_mask,
                  hidden_z=hidden_z,
                  heads_z=_get_zi(heads_z, i),
                  mha_z=_get_zi(mha_z, i, ndim=1),
                  intermediate_z=_get_zi(intermediate_z, i),
                  ffn_z=_get_zi(ffn_z, i, ndim=1))

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def extra_repr(self):
        return f'grad_checkpointing={self.grad_checkpointing}'

    def prune(self):
        mod = self
        for i in range(len(self.resblocks)):
            self.resblocks[i] = self.resblocks[i].prune()
        return mod


class LogitScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, dummy):
        return self.logit_scale


class TextEncoder(nn.Module):
    def __init__(self, embed_dim, text_cfg, quick_gelu):
        super().__init__()

        act_layer = QuickGELU if quick_gelu else nn.GELU
        self.context_length = text_cfg.context_length

        if text_cfg.layers > 0:
            self.transformer = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
        else:
            self.transformer = None

        self.text_projection = None
        if text_cfg.layers > 0:
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(
                text_cfg.vocab_size, text_cfg.width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, text_cfg.width))
            self.ln_final = LayerNorm(text_cfg.width)

            self.text_projection = nn.Parameter(
                torch.empty(text_cfg.width, embed_dim))
            self.register_buffer(
                'attn_mask', self.build_attention_mask(), persistent=False)
        else:
            self.token_embedding = None
        self.init_parameters()

        self.l0_module = None

        self.mask = None

    def init_parameters(self):
        if self.transformer is not None:
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)

            proj_std = (self.transformer.width ** -0.5) * \
                       ((2 * self.transformer.layers) ** -0.5)
            attn_std = self.transformer.width ** -0.5
            fc_std = (2 * self.transformer.width) ** -0.5
            for block in self.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            if self.text_projection is not None:
                nn.init.normal_(self.text_projection,
                                std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, normalized=False,
                    hidden_z: Optional[torch.Tensor] = None,
                    heads_z: Optional[torch.Tensor] = None,
                    mha_z: Optional[torch.Tensor] = None,
                    intermediate_z: Optional[torch.Tensor] = None,
                    ffn_z: Optional[torch.Tensor] = None,
                    embed_dim_z: Optional[torch.Tensor] = None,
                    ):
        self.hidden_z = hidden_z
        self.embed_dim_z = embed_dim_z

        text = text.to(self.token_embedding.weight.device)
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        if hidden_z is not None:
            x = torch.mul(x, hidden_z)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask,
                             hidden_z=hidden_z,
                             heads_z=heads_z,
                             mha_z=mha_z,
                             intermediate_z=intermediate_z,
                             ffn_z=ffn_z)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x, hidden_z)

        # if hidden_z is not None:
        #     x = torch.mul(x, hidden_z)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.get_proj_feature(x)
        if embed_dim_z is not None:
            x = x.mul(embed_dim_z)
        if normalized:
            x = F.normalize(x, dim=-1)

        return x

    def get_proj_feature(self, x):
        return x @ self.text_projection

    def forward(self, text, normalized=False):
        mask = dict()

        if self.l0_module is not None:
            mask = self.l0_module.forward()

        self.mask = mask

        return self.encode_text(text, normalized=normalized, **mask)


class VisualTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU,
            teacher_width: int = -1,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.embed_dim = width
        self.layers = layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer)
        self.head_dim = width // heads

        self.ln_post = LayerNorm(width)
        # image proj
        if teacher_width > 0:
            self.proj = nn.Parameter(torch.empty(
                teacher_width, output_dim), requires_grad=False)
        else:
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor,
                hidden_z: Optional[torch.Tensor] = None,
                heads_z: Optional[torch.Tensor] = None,
                mha_z: Optional[torch.Tensor] = None,
                intermediate_z: Optional[torch.Tensor] = None,
                ffn_z: Optional[torch.Tensor] = None,
                embed_dim_z: Optional[torch.Tensor] = None):

        self.hidden_z = hidden_z
        self.embed_dim_z = embed_dim_z

        x = x.to(self.conv1.weight.device)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # the first token is the class token.
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, 1 + grid ** 2, width]
        x = x + self.positional_embedding.to(x.dtype)  # 128, 50, 768

        if hidden_z is not None:
            x = torch.mul(x, hidden_z)
        x = self.ln_pre(x, hidden_z=hidden_z)

        x = x.permute(1, 0, 2)  # NLD -> LND 50, 128, 768
        x = self.transformer(x,
                             hidden_z=hidden_z,
                             heads_z=heads_z,
                             mha_z=mha_z,
                             intermediate_z=intermediate_z,
                             ffn_z=ffn_z)

        x = x.permute(1, 0, 2)  # LND -> NLD

        # select class token
        x = self.ln_post(x[:, 0, :], hidden_z=hidden_z)

        if self.proj is not None:
            x = self.get_proj_feature(x)

        return x

    def get_proj_feature(self, x):
        if self.proj is not None:
            x = x @ self.proj
        return x

    def extra_repr(self):
        return 'image_size={}, output_dim={}'.format(self.image_size, self.output_dim)


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    teacher_width: int = -1
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    # use (imagenet) pretrained weights for named model
    timm_model_pretrained: bool = False
    # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_pool: str = 'avg'
    # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj: str = 'linear'


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    teacher_width: int = -1
    heads: int = 8
    layers: int = 12


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim, vision_cfg, quick_gelu):
        super().__init__()
        act_layer = QuickGELU if quick_gelu else nn.GELU

        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
            teacher_width=vision_cfg.teacher_width,
        )
        self.init_parameters()

        self.l0_module = None

        self.mask = None

    def init_parameters(self):
        if hasattr(self.visual, 'init_parameters'):
            self.visual.init_parameters()

    def forward(self, image, normalized=False,
                **mask):

        if self.l0_module is not None:
            mask = self.l0_module.forward()

        self.mask = mask

        image_features = self.visual(image, **mask)

        embed_dim_z = mask.get('embed_dim_z', None)
        if embed_dim_z is not None:
            image_features = image_features.mul(embed_dim_z)

        if normalized:
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def prune(self):
        self.visual = self.visual.prune()
        return self


class MiniClip(nn.Module):
    def __init__(self, cfg_path):
        super(MiniClip, self).__init__()

        # autocast context
        self.image_autocast = nullcontext
        self.text_autocast = nullcontext
        self.logit_autocast = nullcontext

        with open(cfg_path, "r") as fp:
            cfg = json.loads(fp.read())

        emb_dim = cfg["embed_dim"]
        text_cfg = CLIPTextCfg(**cfg["text_cfg"])
        vision_cfg = CLIPVisionCfg(**cfg["vision_cfg"])
        quick_gelu = True
        self.text_encoder = TextEncoder(emb_dim, text_cfg, quick_gelu)
        self.image_encoder = ImageEncoder(emb_dim, vision_cfg, quick_gelu)

        self.logit_scale = LogitScale()

    def encode_image(self, image, normalized=False):
        with self.image_autocast():
            return self.image_encoder(image, normalized=normalized)

    def encode_text(self, text, normalized=False):
        with self.text_autocast():
            return self.text_encoder(text, normalized=normalized)

    def forward(self, image, text, normalized=True):
        image_features = text_features = None
        if image is not None:
            with self.image_autocast():
                image_features = self.image_encoder(
                    image, normalized=normalized)
        if text is not None:
            with self.text_autocast():
                text_features = self.text_encoder(text, normalized=normalized)
        with self.logit_autocast():
            logit_scale = self.logit_scale(torch.tensor(0))
        return image_features, text_features, logit_scale.exp()


if __name__ == '__main__':
    # text_embed_dim = 512
    # with open("model_configs/TinyCLIP-ViT-40M-32-Text-19M.json", "r") as fp:
    #     cfg = json.loads(fp.read())
    #
    # text_cfg = CLIPTextCfg(**cfg["text_cfg"])
    # quick_gelu = True
    # text_model = TextEncoder(512, text_cfg, quick_gelu)
    # for k,v in text_model.named_parameters():
    #     print(k, v.shape)
    #
    # image_embed_dim = 512
    # image_cfg = CLIPVisionCfg(**cfg["vision_cfg"])
    # vimage_model = ImageEncoder(image_embed_dim, image_cfg, quick_gelu)
    # for k,v in vimage_model.named_parameters():
    #     print(k, v.shape)

    cfg_path = "model_configs/TinyCLIP-ViT-40M-32-Text-19M.json"
    clip = MiniClip(cfg_path)

    for k, v in clip.named_parameters():
        print(k, v.shape)

    state_dict = torch.load("/data/gongoubo/MiniClip/model_hub/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt",
                            map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict["state_dict"].items():
        if "visual" in k:
            new_state_dict[k.replace("module", "image_encoder")] = v
        elif "logit_scale" in k:
            new_state_dict[k.replace("module", "logit_scale")] = v
        else:
            new_state_dict[k.replace("module", "text_encoder")] = v

    clip.load_state_dict(new_state_dict, strict=True)

    img_path = "data/dog.png"
    text = ["a dog", "a cat", "a fish", "a pig"]

    image = Image.open(img_path).convert("RGB")
    val_processor = image_transform(clip.image_encoder.visual.image_size, is_train=False)

    image_input = val_processor(image).unsqueeze(0)
    text_input = tokenize(text)

    print(image_input.shape)
    print(text_input.shape)
    img_feature = clip.encode_image(image_input, normalized=True)
    text_feature = clip.encode_text(text_input, normalized=True)

    img_feature = img_feature.detach().cpu().numpy()
    text_feature = text_feature.detach().cpu().numpy()
    print(text_feature @ img_feature.T)
