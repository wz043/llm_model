import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 2048
    vocab_size: int = 50000
    # GQA
    dim: int = 1152
    n_heads: int = 12
    groups: int = 6
    head_dim: int = dim // n_heads
    n_layers: int = 24
    attn_drop: float = 0
    proj_drop: float = 0
    # RoPE
    qk_rope_head_dim: int = dim // n_heads
    base: float = 10000.0
    # FFN
    ffn_drop:float = 0


class Embedding(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.emb = nn.Embedding(self.vocab_size,self.dim)

    def forward(self,x):
        x = self.emb(x)
        return x


def apply_rotary_emb(x,args:ModelArgs):
    device = x.device
    bsz,seq_len,heads,dim = x.shape
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_reverse = torch.zeros_like(x,device=device)
    x_reverse[..., 0::2] = -x_odd
    x_reverse[..., 1::2] = x_even
    i = torch.arange(0, dim, 2,device=device).repeat_interleave(2)
    base_theta = 1.0 / (args.base ** (i / dim)).to(device)
    pos = torch.arange(seq_len,device=device)
    angles = torch.outer(pos, base_theta)
    cos_theta = torch.cos(angles).reshape(1,seq_len,1,dim).to(device)
    sin_theta = torch.sin(angles).reshape(1,seq_len,1,dim).to(device)
    return x*cos_theta + x_reverse*sin_theta


def repeat_kv(x, groups: int) -> torch.Tensor:
    bs, seq, n_kv_heads, head_dim = x.shape
    if groups == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seq, n_kv_heads, groups, head_dim)
        .reshape(bs, seq, n_kv_heads * groups, head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,x:torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)


class GQA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.groups = args.groups
        self.n_heads = args.n_heads
        self.kv_heads = self.n_heads // self.groups
        self.head_dim = self.dim // self.n_heads
        self.q = nn.Linear(self.dim, self.dim)
        self.kv = nn.Linear(self.dim, (self.kv_heads * self.head_dim) * 2)
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(args.attn_drop)
        self.wo_drop = nn.Dropout(args.proj_drop)
        self.wo = nn.Linear(self.dim, self.dim)

    def forward(self, x, mask, pad_mask):
        bsz, seqlen, feature = x.shape
        q = self.q(x).reshape(bsz, seqlen, self.n_heads, feature // self.n_heads)
        kv = self.kv(x).reshape(bsz, seqlen, 2, self.kv_heads, feature // self.n_heads).permute(2, 0, 1, 3, 4)
        rq, rk, v = apply_rotary_emb(q, self.args), apply_rotary_emb(kv[0], self.args), kv[1]
        attn = torch.einsum("bshd,bthd->bsht", rq, repeat_kv(rk, self.groups)) * self.scale  # rq @ rk.transpose(-2, -1) * self.scale + mask
        if mask is not None:
            attn = attn + mask.unsqueeze(1) + pad_mask.unsqueeze(2)
        scores = attn.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores = self.attn_drop(scores)
        x = torch.einsum("bsht,bthd->bshd", scores, repeat_kv(v, self.groups))
        x = self.wo(x.flatten(2))
        return self.wo_drop(x)


class FFN(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.w1 = nn.Linear(self.dim, self.dim * 4)
        self.w2 = nn.Linear(self.dim * 4, self.dim)
        self.w3 = nn.Linear(self.dim, self.dim * 4)
        self.ffn_drop = nn.Dropout(args.ffn_drop)

    def forward(self, x):
        return self.ffn_drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.attn = GQA(args)
        self.ffn = FFN(args)
        self.attn_norm = RMSNorm(self.dim)
        self.ffn_norm = RMSNorm(self.dim)

    def forward(self, x, mask, pad_mask):
        x = x + self.attn(self.attn_norm(x), mask, pad_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.emb = Embedding(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(args))
        self.norm = RMSNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens , pad_mask, **kwargs):
        seqlen = tokens.size(1)
        h = self.emb(tokens)
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, mask, pad_mask)
        h = self.norm(h)
        logits = self.head(h)
        return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    y = torch.randint(0, 2, (2, 128, 128))
    model = Transformer(args)
    print(model(x, y).size())
    print(model(x, y))
    print(torch.cuda.memory_summary())
    current_mem = torch.cuda.memory_allocated() / 1024 ** 2
    print(current_mem, "MB")
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params / 10 ** 6, "M")
