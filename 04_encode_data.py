import json

with open("math_theorems_final.txt", "r", encoding="utf-8") as f:
    text_cleaned = f.read()

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)

stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}

import torch

data = torch.tensor([stoi[ch] for ch in text_cleaned], dtype=torch.long)

n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"입력 {context.tolist()} -> 예측할 다음 글자 {target.item()}")

def get_batch(data, block_size, batch_size):
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

torch.manual_seed(42)
xb,yb=get_batch(train_data,block_size=8, batch_size=4)

torch.save({"train_data": train_data, "val_data": val_data}, "encoded_data.pt")
print("저장 완료")
