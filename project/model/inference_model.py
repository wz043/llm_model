import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    max_batch_size:int = 8
    max_seq_len:int = 2048
    vocab_size:int = 151936
    #GQA
    dim:int = 1152
    n_heads:int = 8
    groups:int = 4
    head_dim:int = dim // n_heads
    n_layers:int = 21
    attn_drop:int = 0.
    proj_drop:int = 0.
    #RoPE
    qk_rope_head_dim:int = dim // n_heads
    base:int = 10000.0
    # FFN
    ffn_drop:int = 0.

class Embedding(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.dim = args.dim
        self.emb = nn.Embedding(self.vocab_size,self.dim)

    def forward(self,x):
        x = self.emb(x)
        return x

def precompute_freqs_cis(args:ModelArgs):
    freqs = 1.0 / (args.base ** (torch.arange(0, args.qk_rope_head_dim, 2, dtype=torch.float32) / args.qk_rope_head_dim))
    seqlen = args.max_seq_len
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

def apply_rotary_emb(x,freqs):
    dtype = x.dtype
    #x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    bsz,seq,heads,heads_dim = x.shape
    x = torch.view_as_complex(x.float().view(bsz,seq,heads, -1, 2))
    freqs_cis = freqs.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

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
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class GQA(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.groups = args.groups
        self.n_heads = args.n_heads
        self.kv_heads = self.n_heads//self.groups
        self.head_dim = self.dim // self.n_heads
        self.q = nn.Linear(self.dim , self.dim)
        self.kv = nn.Linear(self.dim , (self.kv_heads*self.head_dim)*2)
        self.scale = self.head_dim ** -0.5
        self.attn_drop = nn.Dropout(args.attn_drop)
        self.wo_drop = nn.Dropout(args.proj_drop)
        self.wo = nn.Linear(self.dim,self.dim)
        self.register_buffer("k_cache",torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads//self.groups, self.head_dim),persistent=False)
        self.register_buffer("v_cache",torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads//self.groups, self.head_dim),persistent=False)

    def forward(self,x, start_pos, freqs_cis, mask):
        bsz , seqlen , feature = x.shape
        end_pos = start_pos + seqlen
        q = self.q(x).reshape(bsz ,seqlen ,self.n_heads ,feature // self.n_heads)
        kv = self.kv(x).reshape(bsz ,seqlen ,2 ,self.kv_heads ,feature // self.n_heads).permute(2,0,1,3,4)
        rq ,rk ,v =apply_rotary_emb(q,freqs_cis),apply_rotary_emb(kv[0],freqs_cis),kv[1]
        self.k_cache[:bsz, start_pos:end_pos] = rk
        self.v_cache[:bsz, start_pos:end_pos] = v
        attn = torch.einsum("bshd,bthd->bsht",rq,repeat_kv(self.k_cache[:bsz, :end_pos],self.groups)) * self.scale#rq @ rk.transpose(-2, -1) * self.scale + mask
        if mask is not None:
            attn += mask.unsqueeze(1)
        scores = attn.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores = self.attn_drop(scores)
        x = torch.einsum("bsht,bthd->bshd",scores, repeat_kv(self.v_cache[:bsz, :end_pos],self.groups))
        x = self.wo(x.flatten(2))
        return self.wo_drop(x)


class FFN(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.w1 = nn.Linear(self.dim,self.dim*4)
        self.w2 = nn.Linear(self.dim*4,self.dim)
        self.w3 = nn.Linear(self.dim,self.dim*4)
        self.ffn_drop = nn.Dropout(args.ffn_drop)

    def forward(self,x):
        return self.ffn_drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Block(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.attn = GQA(args)
        self.ffn = FFN(args)
        self.attn_norm = RMSNorm(self.dim)
        self.ffn_norm = RMSNorm(self.dim)

    def forward(self,x,start_pos, freqs_cis, mask):
        x = x + self.attn(self.attn_norm(x),start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class GMT_Zero(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.emb = Embedding(args)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(args))
        self.norm = RMSNorm(self.dim)
        self.head = nn.Linear(self.dim,self.vocab_size)
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, token, start_pos:int = 0):
        seqlen = token.size(1)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        h = self.emb(token)
        mask = None
        if seqlen > 1 :
            mask = torch.full((seqlen, seqlen), float("-inf"), device=token.device).triu_(1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=token.device),mask])
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)#[:, -1]
        logits = self.head(h)
        return logits

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = GMT_Zero(args)
    print(x)
    print(model(x))
    print(model(x).size())
    print(torch.cuda.memory_summary())
    current_mem = torch.cuda.memory_allocated() / 1024 ** 2
    print(current_mem,"MB")
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params/10**6,"M")
    print("Model keys:", model.state_dict().keys())
