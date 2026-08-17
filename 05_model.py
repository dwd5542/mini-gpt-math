import torch
import torch.nn as nn
import json
import torch.nn.functional as F

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}
vocab_size = len(stoi)

loaded = torch.load("encoded_data.pt")
train_data = loaded["train_data"]
val_data = loaded["val_data"]

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

n_embd = 64
n_head = 4
n_layer = 6
block_size = 8

token_embedding_table = nn.Embedding(vocab_size, n_embd)

xb, yb = get_batch(train_data, block_size=8, batch_size=4)
emb = token_embedding_table(xb)

position_embedding_table = nn.Embedding(block_size, n_embd)

positions = torch.arange(block_size)
pos_emb = position_embedding_table(positions)

x = emb + pos_emb

head_size=16

# key=nn.Linear(n_embd,head_size,bias=False)
# query=nn.Linear(n_embd,head_size,bias=False)
# value=nn.Linear(n_embd,head_size,bias=False)

# k=key(x)
# q=query(x)
# v=value(x)

# wei=q@k.transpose(-2,-1)

# tril=torch.tril(torch.ones(block_size,block_size))

# wei_masked=wei.masked_fill(tril==0,float("-inf"))

# wei_softmax=F.softmax(wei_masked, dim=-1)

# out=wei_softmax@v
# print(out.shape)

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        wei=q@k.transpose(-2,-1)*(k.shape[-1]**-0.5)
        wei=wei.masked_fill(self.tril[:T,:T]==0, float("-inf"))
        wei=F.softmax(wei, dim=-1)
        v=self.value(x)
        out=wei@v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1)

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
        )

    def forward(self,x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)

    def forward(self,x):
        x=x+self.sa(self.ln1(x))
        x=x+self.ffwd(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self,vocab_size,n_embd,block_size,n_head,n_layer):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size, n_embd)
        self.blocks=nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)

    def forward(self,idx,targets=None):
        B,T=idx.shape
        tok_emb=self.token_embedding_table(idx)
        pos_emb=self.position_embedding_table(torch.arange(T))
        x=tok_emb+pos_emb
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)

        if targets is None:
            loss=None
        else:
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)

        return logits,loss

# mha = MultiHeadAttention(num_heads=4, head_size=4)
# out = mha(x)
# print(out.shape)

# ff=FeedForward(n_embd=16)
# out2=ff(out)
# print(out2.shape)

# block = Block(n_embd=16, n_head=4)
# out3 = block(x)
# print(out3.shape)

model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
logits, loss = model(xb, yb)
# print(logits.shape)
# print(loss)

# import math

# losses = []
# for _ in range(20):
#     xb, yb = get_batch(train_data, block_size, batch_size=4)
#     _, loss = model(xb, yb)
#     losses.append(loss.item())

# losses = torch.tensor(losses)
# print(f"평균 loss: {losses.mean():.4f}")
# print(f"표준편차: {losses.std():.4f}")
# print(f"이론적 바닥값 log(vocab_size): {math.log(vocab_size):.4f}")

optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)

max_iters=16000
eval_interval=400
eval_iters=50

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split,data in [("train",train_data),("val",val_data)]:
        losses=torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb,yb=get_batch(data,block_size,batch_size=4)
            _,loss=model(xb,yb)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

# for iter in range(max_iters):
#     if iter%eval_interval==0:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#     xb, yb = get_batch(train_data, block_size, batch_size=4)
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

import json
import os

# configs = [
#     {"n_embd": 16, "n_head": 4, "n_layer": 4},
#     {"n_embd": 16, "n_head": 4, "n_layer": 6},
#     {"n_embd": 64, "n_head": 4, "n_layer": 4},
#     {"n_embd": 64, "n_head": 4, "n_layer": 6},
# ]

# results_file = "capacity_experiment_results.json"
# all_results = []

# for config in configs:
#     n_embd = config["n_embd"]
#     n_head = config["n_head"]
#     n_layer = config["n_layer"]

#     model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     print(f"=== 시작: n_embd={n_embd}, n_layer={n_layer} ===")

#     for iter in range(max_iters):
#         if iter % eval_interval == 0:
#             losses = estimate_loss()
#             print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

#         xb, yb = get_batch(train_data, block_size, batch_size=4)
#         logits, loss = model(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#     final_eval = estimate_loss()
#     result = {
#         "n_embd": n_embd,
#         "n_head": n_head,
#         "n_layer": n_layer,
#         "block_size": block_size,
#         "max_iters": max_iters,
#         "final_train_loss": final_eval["train"].item(),
#         "final_val_loss": final_eval["val"].item(),
#     }
#     all_results.append(result)
#     print(f"=== 완료: {result} ===\n")

# with open(results_file, "w") as f:
#     json.dump(all_results, f, indent=2)

# print("전체 실험 결과 저장 완료 →", results_file)

##########################################################

# configs = [
#     {"n_embd": 64, "n_head": 4, "n_layer": 6, "block_size": 16},
#     {"n_embd": 64, "n_head": 4, "n_layer": 6, "block_size": 32},
# ]

# results_file = "capacity_experiment_results.json"
# if os.path.exists(results_file):
#     with open(results_file, "r") as f:
#         all_results = json.load(f)
# else:
#     all_results = []

# for config in configs:
#     n_embd = config["n_embd"]
#     n_head = config["n_head"]
#     n_layer = config["n_layer"]
#     block_size = config["block_size"]

#     model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     print(f"=== 시작: n_embd={n_embd}, n_layer={n_layer}, block_size={block_size} ===")

#     for iter in range(max_iters):
#         if iter % eval_interval == 0:
#             losses = estimate_loss()
#             print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

#         xb, yb = get_batch(train_data, block_size, batch_size=4)
#         logits, loss = model(xb, yb)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()

#     final_eval = estimate_loss()
#     result = {
#         "n_embd": n_embd,
#         "n_head": n_head,
#         "n_layer": n_layer,
#         "block_size": block_size,
#         "max_iters": max_iters,
#         "final_train_loss": final_eval["train"].item(),
#         "final_val_loss": final_eval["val"].item(),
#     }
#     all_results.append(result)
#     print(f"=== 완료: {result} ===\n")

# with open(results_file, "w") as f:
#     json.dump(all_results, f, indent=2)

# print("전체 실험 결과 저장 완료 →", results_file)

##########################################################

# n_embd = 64
# n_head = 4
# n_layer = 6
# block_size = 64

# model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# print(f"=== 시작: n_embd={n_embd}, n_layer={n_layer}, block_size={block_size} ===")

# for iter in range(max_iters):
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

#     xb, yb = get_batch(train_data, block_size, batch_size=4)
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# final_eval = estimate_loss()
# result = {
#     "n_embd": n_embd,
#     "n_head": n_head,
#     "n_layer": n_layer,
#     "block_size": block_size,
#     "max_iters": max_iters,
#     "final_train_loss": final_eval["train"].item(),
#     "final_val_loss": final_eval["val"].item(),
# }
# print(f"=== 완료: {result} ===")

# results_file = "capacity_experiment_results.json"
# with open(results_file, "r") as f:
#     all_results = json.load(f)
# all_results.append(result)
# with open(results_file, "w") as f:
#     json.dump(all_results, f, indent=2)

# print("저장 완료 →", results_file)

##########################################################

n_embd = 64
n_head = 4
n_layer = 6
block_size = 32

model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"=== 최종 모델 학습 시작: n_embd={n_embd}, n_layer={n_layer}, block_size={block_size} ===")

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch(train_data, block_size, batch_size=4)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_eval = estimate_loss()
print(f"=== 학습 완료: train {final_eval['train']:.4f}, val {final_eval['val']:.4f} ===")

torch.save(model.state_dict(), "gpt_model.pt")

config = {
    "vocab_size": vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
}
with open("gpt_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("모델 가중치 저장 완료 → gpt_model.pt")
print("모델 설정 저장 완료 → gpt_config.json")