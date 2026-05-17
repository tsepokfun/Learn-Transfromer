# 🧠 MiniGPT: A Transformer Language Model from Scratch

> A minimal, educational implementation of a GPT-style Transformer for language modeling, built from scratch with PyTorch.

Trained on a large text corpus to generate English text.

**🎯 Goal:** Understand the internals of the Transformer architecture (Q/K/V, multi-head attention, residual connections, causal masking) by writing and training it yourself.

---

## 📂 Files

* `transformer.py` — Training script (model definition, training loop, checkpoint saving)
* `inference.py` — Interactive text generation script
* `README.md` — This file

---

## 📊 Model Overview

| Specification | Details |
| --- | --- |
| **Architecture** | Decoder-only Transformer (GPT-like) |
| **Parameters** | ~89 Million |
| **Vocabulary** | 50,000 words (whitespace tokenizer) |
| **Layers** | 12 |
| **Dimensions** | Embed dim: 512, Heads: 8, Context length: 128 tokens |
| **Training Data** | 1.36 million lines (~77 million words, mainly English Wikipedia with some noise) |
| **Hardware** | NVIDIA RTX 4060 Ti 16GB |
| **Training Time** | ~9.6 hours (5 epochs) |
| **Performance** | Final average loss: **3.21** (Perplexity ~24.7) |

---

## ⚙️ How It Works

This project implements the core components of the Transformer architecture from the ground up:

* **Embeddings:** Word embeddings + sinusoidal position encoding
* **Attention:** Multi-head scaled dot-product attention (Q, K, V projections)
* **Masking:** Causal (look-ahead) mask to prevent peeking at future tokens
* **Architecture:** Residual connections + layer normalization
* **FFN:** Feed-forward networks (`GELU` activation)
* **Loss:** Cross-entropy loss with `ignore_index` for padding
* **Optimization:** `AdamW` optimizer with gradient clipping and mixed precision training
* **Generation:** Autoregressive generation with temperature sampling

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install torch numpy

```

### 2. Prepare your data

Put your training text in `corpus.txt` (one sentence or paragraph per line).
*Note: I used a cleaned subset of Wikitext-103 and some additional multi‑language text.*

### 3. Train the model

```bash
python transformer.py

```

The script will:

1. Build a 50k word vocabulary (saved as `tokenizer.pkl`)
2. Train for 5 epochs, saving a checkpoint after each epoch (`model_epochX.pt`)
3. Save the final model as `mini_gpt_model.pt`

> 💡 **Training Tip:** Training uses CUDA if available. On an RTX 4060 Ti with 16GB VRAM, one epoch takes about 115 minutes.

### 4. Generate text

```bash
python inference.py

```

Type a prompt and the model will continue writing.

**Example:**

```text
>>> hello
hello was released in the uk on 14 november 2012 ...

```

---

## ⚠️ Known Limitations

* **Whitespace tokenizer:** Produces many `<unk>` tokens for rare or non‑English words. Switching to BPE/WordPiece would eliminate this.
* **Factual hallucinations:** The model often generates plausible‑sounding but incorrect information.
* **No chat/instruction tuning:** It was trained on raw Wikipedia text, so it behaves like a document generator, not a helpful assistant.
* **Small context window:** Only 128 tokens, so long‑range coherence is limited.

---

## 🔮 Future Improvements

* [ ] Replace the tokenizer with a BPE model (e.g., HuggingFace tokenizers) to remove `<unk>`.
* [ ] Add KV‑cache for faster autoregressive decoding.
* [ ] Implement Top‑K / Top‑P sampling for better generation control.
* [ ] Fine‑tune on dialogue data (`User:` / `Assistant:` format) to create a chatbot.

---

## 💡 Why This Project Exists

I built this to truly understand how Transformers work — not just reading about them, but coding every piece and watching the loss drop from 10.5 to 3.2.

If you're learning Transformers, feel free to read the code, ask questions, or fork and experiment!

## 📄 License

MIT
