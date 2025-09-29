import argparse
import math
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------

@dataclass
class Config:
    # model
    vocab_size: int = None  # deduzido do dataset
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128

    # training
    batch_size: int = 64
    lr: float = 3e-4
    epochs: int = 10

    # IO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path: str = 'ckpt.pth'


# -----------------------------
# Tokenizer (char-level)
# -----------------------------

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi.get(c, self.stoi[' ']) for c in s]

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)


# -----------------------------
# Dataset
# -----------------------------

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        # data: list of token ids
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(1, (self.data.size(0) - 1) // self.block_size)

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = self.data[start:end]
        y = self.data[start+1:end+1]
        # pad if necessary
        if x.size(0) < self.block_size:
            pad = torch.zeros(self.block_size - x.size(0), dtype=torch.long)
            x = torch.cat([x, pad])
            y = torch.cat([y, pad])
        return x, y


# -----------------------------
# Model components
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        # causal mask (1 means allowed), saved as buffer
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)  # (B, nh, T, hd)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # (B, nh, T, hd)
        out = out.transpose(1,2).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, hidden_mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_mult * n_embd),
            nn.GELU(),
            nn.Linear(hidden_mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        assert config.vocab_size is not None
        self.vocab_size = config.vocab_size
        C = config.n_embd
        self.token_emb = nn.Embedding(self.vocab_size, C)
        self.pos_emb = nn.Embedding(config.block_size, C)
        self.blocks = nn.ModuleList([
            TransformerBlock(C, config.n_head, config.block_size) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(C)
        self.head = nn.Linear(C, self.vocab_size, bias=False)
        self.block_size = config.block_size

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        tok = self.token_emb(idx)  # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                minv = v[:, -1].unsqueeze(1)
                logits = torch.where(logits < minv, torch.full_like(logits, -1e10), logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------
# Training / Utils
# -----------------------------

def get_batch(data_tensor, batch_size, block_size, device):
    # sample random windows
    n = data_tensor.size(0) - 1
    ix = torch.randint(0, n - block_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


def train_loop(model, data_tensor, config: Config):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    device = config.device
    iters_per_epoch = max(1, (data_tensor.size(0) - 1) // config.batch_size)

    for epoch in range(config.epochs):
        pbar = tqdm(range(iters_per_epoch), desc=f"Epoch {epoch+1}/{config.epochs}")
        total_loss = 0.0
        for it in pbar:
            xb, yb = get_batch(data_tensor, config.batch_size, config.block_size, device)
            logits = model(xb)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), yb.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            if (it+1) % 10 == 0:
                pbar.set_postfix({'loss': total_loss / (it+1)})

        # salvar checkpoint ao final de cada época
        save = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.__dict__,
        }
        torch.save(save, config.ckpt_path)
        print(f"Salvo checkpoint em {config.ckpt_path} (época {epoch+1})")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','generate'], default='train')
    parser.add_argument('--data_path', type=str, default='data.txt')
    parser.add_argument('--checkpoint', type=str, default='ckpt.pth')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=40)

    args = parser.parse_args()

    if args.mode == 'train':
        assert os.path.exists(args.data_path), f"Arquivo de dados nao existe: {args.data_path}"
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
        data_ids = tokenizer.encode(text)
        data_tensor = torch.tensor(data_ids, dtype=torch.long)

        config = Config()
        config.vocab_size = tokenizer.vocab_size
        config.n_embd = args.n_embd
        config.n_head = args.n_head
        config.n_layer = args.n_layer
        config.block_size = args.block_size
        config.batch_size = args.batch_size
        config.lr = 3e-4
        config.epochs = args.epochs
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config.ckpt_path = args.checkpoint

        model = MiniGPT(config).to(config.device)
        print(f"Vocab size: {config.vocab_size}, device: {config.device}")
        train_loop(model, data_tensor, config)

    elif args.mode == 'generate':
        # precisa carregar checkpoint e tokenizer (neste script, tokenizer vem do arquivo data.txt)
        assert os.path.exists(args.data_path), "Forneça data_path para carregar tokenizer (ex: data.txt)"
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = CharTokenizer(text)

        # carregar checkpoint
        assert os.path.exists(args.checkpoint), f"Checkpoint não encontrado: {args.checkpoint}"
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        cfg_dict = ckpt.get('config', {})
        cfg = Config()
        cfg.__dict__.update(cfg_dict)
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        cfg.vocab_size = tokenizer.vocab_size
        model = MiniGPT(cfg).to(cfg.device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        prompt = args.prompt
        if prompt == '':
            prompt = input('Prompt: ')

        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(cfg.device)
        out_ids = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

        # Decodifica os IDs para texto
        text_out = tokenizer.decode(out_ids[0].tolist())

        # Corta o texto no marcador [FIM] para não continuar para a próxima pergunta
        if "[FIM]" in text_out:
            text_out = text_out.split("[FIM]")[0]

        print('\n=== Geração ===\n')
        print(text_out)


if __name__ == '__main__':
    main()
