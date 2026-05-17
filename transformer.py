"""
Windows版 Transformer 语言模型训练脚本
适配 RTX 4060 Ti 16GB / i7-12700 / 32GB RAM
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# ---------------------- 超参数（针对 16GB 显存优化）----------------------
VOCAB_SIZE = 50000          # 保持不变，已足够
EMBED_DIM  = 512            # ↑ 从 384 提升，表示力更强
NUM_HEADS  = 8              # 保持 8 头，每头 64 维
HEAD_DIM   = 64
NUM_LAYERS = 12             # ↑ 从 8 增加到 12，更深的理解
BLOCK_SIZE = 128            # 可尝试 256，但先保持 128 以稳定
BATCH_SIZE = 48             # ↑ 从 32 提升，利用更大显存
LEARNING_RATE = 3e-4
EPOCHS = 5                  # 更多 epoch，配合学习率衰减

# ---------------------- 1. 位置编码 ----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]


# ---------------------- 2. 多头注意力 ----------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_O(attn_output)
        return output


# ---------------------- 3. Transformer 块 ----------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.ln1(x + attn_out)          # 残差 + LN
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)            # 残差 + LN
        return x


# ---------------------- 4. MiniGPT 解码器 ----------------------
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=block_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ff_dim=d_model * 4)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.block_size = block_size
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(block_size, block_size)).view(1, block_size, block_size)
        )

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        mask = self.causal_mask[:, :seq_len, :seq_len]
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_final(x)
        logits = self.lm_head(x)   # (batch, seq_len, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            input_cond = input_ids[:, -self.block_size:]
            logits = self.forward(input_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ---------------------- 5. 简单分词器 ----------------------
class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        self.pad_idx = 0
        self.unk_idx = 1
        self.bos_idx = 2
        self.eos_idx = 3

    def build_vocab(self, texts):
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        vocab = [word for word, _ in sorted_words[:self.vocab_size - 4]]

        self.word2idx = {word: i + 4 for i, word in enumerate(vocab)}
        for i, token in enumerate(self.special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word

    def encode(self, text):
        words = ['<bos>'] + text.lower().split() + ['<eos>']
        return [self.word2idx.get(w, self.unk_idx) for w in words]

    def decode(self, token_ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in token_ids])


# ---------------------- 6. 数据集 ----------------------
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text)
            for i in range(0, len(tokens) - 1, block_size):
                chunk = tokens[i:i + block_size + 1]
                if len(chunk) > 1:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    x_list, y_list = zip(*batch)
    max_len = min(max(len(x) for x in x_list), BLOCK_SIZE)
    padded_x, padded_y = [], []
    for x, y in zip(x_list, y_list):
        if len(x) < max_len:
            pad_len = max_len - len(x)
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)])
        else:
            x = x[:max_len]
            y = y[:max_len]
        padded_x.append(x)
        padded_y.append(y)
    return torch.stack(padded_x), torch.stack(padded_y)


# ---------------------- 7. 训练函数 ----------------------
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

def train(model, dataloader, optimizer, epochs, save_prefix="model"):
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    total_start = time.time()  # 总训练开始时间

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                logits = model(inputs)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1), ignore_index=0)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - total_start

        print(f"=== Epoch {epoch+1} 完成 ===")
        print(f"    平均损失: {avg_loss:.4f}")
        print(f"    本 epoch 耗时: {epoch_time/60:.2f} 分钟")
        print(f"    已训练总时长: {total_elapsed/60:.2f} 分钟")
        scheduler.step()  # 更新学习率

        # 保存模型
        save_path = f"{save_prefix}_epoch{epoch+1}.pt"
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'vocab_size': VOCAB_SIZE,
            'embed_dim': EMBED_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'block_size': BLOCK_SIZE,
        }, save_path)
        print(f"    模型已保存到 {save_path}\n")


# ---------------------- 8. 主程序 ----------------------
if __name__ == "__main__":
    # ---- 读取你的 100M 词数据 ----
    print("正在读取训练数据...")
    with open("combined_train.txt", "r", encoding="utf-8") as f:
        all_texts = [line.strip() for line in f if line.strip()]
    print(f"读取到 {len(all_texts)} 行文本")

    # ---- 构建词表 ----
    print("构建词表...")
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(all_texts)
    print(f"词表大小: {len(tokenizer.word2idx)}")

    # ---- 创建数据集和数据加载器 ----
    dataset = TextDataset(all_texts, tokenizer, BLOCK_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_fn, drop_last=True,
                            pin_memory=True, num_workers=2)   # Windows 下 num_workers>0 可能需 if __name__ 保护

    # ---- 初始化模型 ----
    print("初始化模型...")
    model = MiniGPT(
        vocab_size=VOCAB_SIZE,
        d_model=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练 ----
    print("开始训练...")
    train(model, dataloader, optimizer, EPOCHS)

    # ---- 保存模型 ----
    save_path = "./mini_gpt_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'num_heads': NUM_HEADS,
        'num_layers': NUM_LAYERS,
        'block_size': BLOCK_SIZE,
    }, save_path)
    print(f"模型已保存到 {save_path}")

    # ---- 保存分词器 ----
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer.word2idx, f)
    print("分词器已保存")