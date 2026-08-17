# %%
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data=torch.load("encoded_data_bpe.pt")

n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_batch(data, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x, y

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
        pos_emb=self.position_embedding_table(torch.arange(T,device=idx.device))
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

# %%
vocab_size=787
n_embd=64
n_head=4
n_layer=6
block_size=32
max_iters=32000
eval_interval=400
eval_iters=50
batch_size=4

model=GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size,n_head=n_head,n_layer=n_layer)
model = model.to(device)
optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"=== BPE 모델 학습 시작: vocab_size={vocab_size}, block_size={block_size} ===")
train_losses = []
val_losses = []
steps_logged = []

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        steps_logged.append(iter)
        print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb,yb=get_batch(train_data,block_size,batch_size=batch_size)
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_eval=estimate_loss()
print(f"=== 학습 완료: train {final_eval['train']:.4f}, val {final_eval['val']:.4f} ===")

torch.save(model.state_dict(),"gpt_model_bpe.pt")

config = {
    "vocab_size": vocab_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
}
with open("gpt_config_bpe.json", "w") as f:
    import json
    json.dump(config, f, indent=2)

print("모델 저장 완료 → gpt_model_bpe.pt, gpt_config_bpe.json")
# %%
with open("math_theorems_final.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

n_chars = len(full_text)
n_val_chars = n_chars - int(0.9 * n_chars)

val_text = full_text[-n_val_chars:]

print("val 구간 글자 수:", len(val_text))
print("val_data 토큰 수:", len(val_data))

chars_per_token_val = len(val_text) / len(val_data)
print("val 기준 글자/토큰 비율:", chars_per_token_val)
import json

with open("bpe_vocab.json", "r", encoding="utf-8") as f:
    bpe_data = json.load(f)

merges = [tuple(pair) for pair in bpe_data["merges"]]
bpe_stoi = bpe_data["stoi"]

def tokenize_word(word, merges):
    tokens = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens

global_cache = {}

def encode_bpe_corpus(text, merges, stoi, cache, unk_token="▯"):
    words = text.split()
    ids = []
    for word in words:
        if word not in cache:
            cache[word] = tokenize_word(word, merges)
        tokens = cache[word]
        for tok in tokens:
            if tok in stoi:
                ids.append(stoi[tok])
            else:
                ids.append(stoi[unk_token])
    return ids

n_chunks = 10
chunk_size = len(val_text) // n_chunks

ratios = []
for i in range(n_chunks):
    chunk = val_text[i*chunk_size : (i+1)*chunk_size]
    chunk_ids = encode_bpe_corpus(chunk, merges, bpe_stoi,global_cache, unk_token="▯")
    ratio = len(chunk) / len(chunk_ids)
    ratios.append(ratio)

ratios = torch.tensor(ratios)
print("각 조각별 비율:", ratios)
print("평균:", ratios.mean().item())
print("표준편차:", ratios.std().item())
# %%
gaps = [v - t for v, t in zip(val_losses, train_losses)]

n = len(gaps)
first_half = torch.tensor(gaps[:n//2])
second_half = torch.tensor(gaps[n//2:])

print("전체 eval 횟수:", n)
print("초반 절반 격차 평균:", first_half.mean().item(), "표준편차:", first_half.std().item())
print("후반 절반 격차 평균:", second_half.mean().item(), "표준편차:", second_half.std().item())
# %%

@torch.no_grad()
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def decode_bpe(ids, itos):
    tokens = [itos[i] for i in ids]
    text = "".join(tokens)
    text = text.replace("</w>", " ")
    return text.strip()

start_word = "The"
start_tokens = tokenize_word(start_word, merges)
start_ids = [bpe_stoi[tok] for tok in start_tokens]

idx = torch.tensor([start_ids], dtype=torch.long).to(device)

generated = generate(model, idx, max_new_tokens=100)
generated_ids = generated[0].tolist()

bpe_itos = {int(k): v for k, v in bpe_data["itos"].items()}

print(decode_bpe(generated_ids, bpe_itos))
# %%
print("32000스텝 실행 최종 - train:", train_losses[-1], "val:", val_losses[-1])