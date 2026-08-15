import torch 
import torch.nn as nn
from torch.nn import functional as F
import json

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

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}

with open("gpt_config.json", "r") as f:
    config = json.load(f)

n_embd = config["n_embd"]
n_head = config["n_head"]
n_layer = config["n_layer"]
block_size = config["block_size"]
vocab_size = config["vocab_size"]

model = GPTModel(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
model.load_state_dict(torch.load("gpt_model.pt"))
model.eval()

start_char="T"
idx=torch.tensor([[stoi[start_char]]],dtype=torch.long)

@torch.no_grad()
def generate(model,idx,max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-block_size:]
        logits,_=model(idx_cond)
        logits=logits[:,-1,:]
        probs=F.softmax(logits,dim=1)
        idx_next=torch.multinomial(probs,num_samples=1)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx

for i in range(5):
    idx = torch.tensor([[stoi[start_char]]], dtype=torch.long)
    generated = generate(model, idx, max_new_tokens=150)
    generated_ids = generated[0].tolist()
    generated_text = "".join([itos[i] for i in generated_ids])
    print(f"--- 생성 {i+1} ---")
    print(generated_text)
    print()