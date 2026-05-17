from collections import Counter

with open("combined_train.txt", "r", encoding="utf-8") as f:
    text = f.read()

words = text.lower().split()
counter = Counter(words)
total_words = len(words)

# 累积覆盖率
sorted_words = sorted(counter.items(), key=lambda x: x[1], reverse=True)
cum = 0
for i, (word, freq) in enumerate(sorted_words):
    cum += freq
    if i+1 in [5000, 10000, 20000, 30000, 50000]:
        print(f"前 {i+1} 个词覆盖率: {cum/total_words*100:.2f}%")