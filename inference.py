"""
Windows版 交互式生成脚本
加载训练好的模型，在终端里进行对话
"""

import torch
import pickle
from transformer import MiniGPT, SimpleTokenizer   # 确保 transformer.py 在相同文件夹

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推理使用设备: {DEVICE}")

# -------------------- 1. 加载分词器 --------------------
try:
    with open("tokenizer.pkl", "rb") as f:
        word2idx = pickle.load(f)
    tokenizer = SimpleTokenizer(vocab_size=len(word2idx))
    tokenizer.word2idx = word2idx
    tokenizer.idx2word = {v: k for k, v in word2idx.items()}
    print(f"分词器加载成功，词表大小: {len(tokenizer.word2idx)}")
except FileNotFoundError:
    print("错误：找不到 tokenizer.pkl，请先运行 transformer.py 训练并保存分词器。")
    exit()

# -------------------- 2. 加载模型 --------------------
try:
    checkpoint = torch.load("model_epoch5.pt", map_location=DEVICE, weights_only=False)
    model = MiniGPT(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['embed_dim'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
        block_size=checkpoint['block_size']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功，可以开始生成。")
except FileNotFoundError:
    print("错误：找不到 mini_gpt_model.pt，请先运行 transformer.py 训练并保存模型。")
    exit()

# -------------------- 3. 交互循环 --------------------
print("\n" + "=" * 50)
print("输入一句话（英文），模型会接着写下去。")
print("输入 'exit' 退出程序。")
print("=" * 50 + "\n")

while True:
    try:
        prompt = input(">>> ").strip()
    except (KeyboardInterrupt, EOFError):
        break
    if prompt.lower() == "exit":
        break
    if not prompt:
        continue

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=100,
            temperature=0.8
        )

    full_text = tokenizer.decode(output_ids[0].cpu().tolist())
    full_text = full_text.replace('<bos>', '').replace('<eos>', '').strip()
    print(full_text)
    print("-" * 50)