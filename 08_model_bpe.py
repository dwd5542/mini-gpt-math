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
def estimate_loss(model,train_data,val_data,block_size,eval_iters):
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
        losses = estimate_loss(model, train_data, val_data, block_size, eval_iters)
        train_losses.append(losses['train'].item())
        val_losses.append(losses['val'].item())
        steps_logged.append(iter)
        print(f"  step {iter}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb,yb=get_batch(train_data,block_size,batch_size=batch_size)
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

final_eval= estimate_loss(model, train_data, val_data, block_size, eval_iters)
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
@torch.no_grad()
def generate_top_k(model,idx,max_new_tokens,k):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-block_size:]
        logits,_=model(idx_cond)
        logits=logits[:,-1,:]

        top_vals,top_idx=torch.topk(logits,k)
        probs=F.softmax(top_vals,dim=-1)
        sampled=torch.multinomial(probs,num_samples=1)
        idx_next=top_idx.gather(-1,sampled)
        idx=torch.cat((idx,idx_next),dim=1)
    return idx

@torch.no_grad()
def generate_top_p(model,idx,max_new_tokens,p):
    for _ in range(max_new_tokens):
        idx_cond=idx[:,-block_size:]
        logits,_=model(idx_cond)
        logits=logits[:,-1,:]

        probs=F.softmax(logits,dim=-1)
        sorted_probs,sorted_idx=torch.sort(probs,descending=True, dim=-1)
        cumulative_probs=torch.cumsum(sorted_probs,dim=-1)

        cutoff=cumulative_probs>p
        cutoff[...,1:]=cutoff[...,:-1].clone()
        cutoff[...,0]=False
        sorted_probs[cutoff]=0.0

        sorted_probs=sorted_probs/sorted_probs.sum(dim=-1,keepdim=True)
        sampled=torch.multinomial(sorted_probs,num_samples=1)
        idx_next=sorted_idx.gather(-1,sampled)

        idx=torch.cat((idx,idx_next),dim=1)
    return idx

start_word="The"
start_tokens=tokenize_word(start_word, merges)
start_ids=[bpe_stoi[tok] for tok in start_tokens]

for name, fn in [
    ("기존 (순수 multinomial)", lambda: generate(model, torch.tensor([start_ids]).to(device), 80)),
    ("top-k=10", lambda: generate_top_k(model, torch.tensor([start_ids]).to(device), 80, k=10)),
    ("top-p=0.9", lambda: generate_top_p(model, torch.tensor([start_ids]).to(device), 80, p=0.9)),
]:
    result=fn()
    print(f"--- {name} ---")
    print(decode_bpe(result[0].tolist(), bpe_itos))
    print()
# %%
from collections import defaultdict

with open("math_theorems_final.txt", "r", encoding="utf-8") as f:
    corpus_text = f.read()

word_freq = defaultdict(int)
for word in corpus_text.split():
    word_freq[word] += 1

def measure_valid_word_ratio(generate_fn, name, word_freq, start_words, n_trials_per_word=5):
    ratios = []
    for start_word in start_words:
        start_tokens = tokenize_word(start_word, merges)
        start_ids = [bpe_stoi[tok] for tok in start_tokens]
        for i in range(n_trials_per_word):
            idx = torch.tensor([start_ids]).to(device)
            result = generate_fn(idx)
            decoded = decode_bpe(result[0].tolist(), bpe_itos)
            words = decoded.split()
            if len(words) == 0:
                continue
            valid = sum(1 for w in words if w.strip(".,;:()[]") in word_freq)
            ratios.append(valid / len(words))
            print(f"{name} {i} :",decoded)
    ratios = torch.tensor(ratios)
    return ratios.mean().item(), ratios.std().item()

start_words = ["The", "Let", "If", "Suppose", "For"]

results = {}
results["기존"] = measure_valid_word_ratio(
    lambda idx: generate(model, idx, 80),"기존", word_freq, start_words)
results["top-k=10"] = measure_valid_word_ratio(
    lambda idx: generate_top_k(model, idx, 80, k=10),"top-k=10", word_freq, start_words)
results["top-p=0.9"] = measure_valid_word_ratio(
    lambda idx: generate_top_p(model, idx, 80, p=0.9), "top-p=0.9", word_freq, start_words)

for name, (mean, std) in results.items():
    print(f"{name}: 평균 {mean:.4f}, 표준편차 {std:.4f}")
# %%
normal={}

mean, std = measure_valid_word_ratio(
    lambda idx: generate(model, idx, 80),
    f"normal", word_freq, start_words
)
normal["normal"] = (mean, std)
print(f"normal : 평균 {mean:.4f}, 표준편차 {std:.4f}")


k_values = [5, 10, 20, 30]
k_results = {}

for k in k_values:
    mean, std = measure_valid_word_ratio(
        lambda idx, k=k: generate_top_k(model, idx, 80, k=k),
        f"top-k={k}", word_freq, start_words
    )
    k_results[k] = (mean, std)
    print(f"k={k}: 평균 {mean:.4f}, 표준편차 {std:.4f}")

p_values = [0.7, 0.8, 0.9, 0.95]
p_results = {}

for p in p_values:
    mean, std = measure_valid_word_ratio(
        lambda idx, p=p: generate_top_p(model, idx, 80, p=p),
        f"top-p={p}", word_freq, start_words
    )
    p_results[p] = (mean, std)
    print(f"p={p}: 평균 {mean:.4f}, 표준편차 {std:.4f}")

print(normal)
print(k_results)
print(p_results)

# %%
def measure_diversity(generate_fn,start_ids,n_trials=5):
    texts=[]
    for _ in range(n_trials):
        idx=torch.tensor([start_ids]).to(device)
        result=generate_fn(idx)
        decoded=decode_bpe(result[0].tolist(),bpe_itos)
        texts.append(set(decoded.split()))

    distances=[]
    for i in range(len(texts)):
        for j in range(i+1,len(texts)):
            union=texts[i]|texts[j]
            intersection=texts[i]&texts[j]
            jaccard_sim=len(intersection)/len(union) if union else 0
            distances.append(1-jaccard_sim)

    return torch.tensor(distances).mean().item()

diversity_results = {}

for k in [5, 10, 20, 30]:
    div = measure_diversity(
        lambda idx, k=k: generate_top_k(model, idx, 80, k=k),
        start_ids, n_trials=5
    )
    diversity_results[f"top-k={k}"] = div
    print(f"top-k={k}: 다양성 {div:.4f}")

for p in [0.7, 0.8, 0.9, 0.95]:
    div = measure_diversity(
        lambda idx, p=p: generate_top_p(model, idx, 80, p=p),
        start_ids, n_trials=5
    )
    diversity_results[f"top-p={p}"] = div
    print(f"top-p={p}: 다양성 {div:.4f}")

div_baseline = measure_diversity(lambda idx: generate(model, idx, 80), start_ids, n_trials=5)
diversity_results["기존"] = div_baseline
print(f"기존: 다양성 {div_baseline:.4f}")
# %%
def measure_diversity_repeated(generate_fn,start_words,n_trials=5,n_repeats=5):
    all_diversities=[]
    for start_word in start_words:
        start_tokens=tokenize_word(start_word,merges)
        start_ids=[bpe_stoi[tok] for tok in start_tokens]
        for _ in range(n_repeats):
            div=measure_diversity(generate_fn,start_ids,n_trials)
            all_diversities.append(div)
    return torch.tensor(all_diversities)

diversity_results_repeated = {}

div_baseline_samples = measure_diversity_repeated(
    lambda idx: generate(model, idx, 80), start_words, n_trials=5, n_repeats=5
)
diversity_results_repeated["기존"] = div_baseline_samples

for k in k_values:
    samples = measure_diversity_repeated(
        lambda idx, k=k: generate_top_k(model, idx, 80, k=k), start_words, n_trials=5, n_repeats=5
    )
    diversity_results_repeated[f"top-k={k}"] = samples

for p in p_values:
    samples = measure_diversity_repeated(
        lambda idx, p=p: generate_top_p(model, idx, 80, p=p), start_words, n_trials=5, n_repeats=5
    )
    diversity_results_repeated[f"top-p={p}"] = samples

for name, samples in diversity_results_repeated.items():
    mean = samples.mean().item()
    std = samples.std().item()
    se = std / (len(samples) ** 0.5)
    print(f"{name}: 평균 {mean:.4f}, SE {se:.4f} (n={len(samples)})")

# %%
import json

checkpoints = [500, 1000, 2000]
encoded_by_cp = {}

for cp in checkpoints:
    data_cp = torch.load(f"encoded_data_bpe_{cp}.pt")
    n_cp = int(0.9 * len(data_cp))

    with open(f"bpe_vocab_{cp}.json", "r", encoding="utf-8") as f:
        bpe_data_cp = json.load(f)
    vocab_size_cp = len(bpe_data_cp["stoi"])

    encoded_by_cp[cp] = {
        "vocab_size": vocab_size_cp,
        "train_data": data_cp[:n_cp],
        "val_data": data_cp[n_cp:],
    }
    print(f"[{cp}] vocab_size={vocab_size_cp}, train={len(encoded_by_cp[cp]['train_data'])}, val={len(encoded_by_cp[cp]['val_data'])}")

# %%

n_embd = 64
n_head = 4
n_layer = 6
block_size = 32
max_iters = 32000
eval_interval = 400
eval_iters = 50
batch_size = 4

all_results = {}


for cp in checkpoints:
    print(f"\n=== {cp} 병합 모델 학습 시작 ===")

    train_data_cp = encoded_by_cp[cp]["train_data"]
    val_data_cp = encoded_by_cp[cp]["val_data"]
    vocab_size_cp = encoded_by_cp[cp]["vocab_size"]

    model = GPTModel(vocab_size=vocab_size_cp, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_losses, val_losses, steps_logged = [], [], []

    for it in range(max_iters):
        if it % eval_interval == 0:
            losses = estimate_loss(model, train_data_cp, val_data_cp, block_size, eval_iters)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            steps_logged.append(it)
            print(f"  [{cp}] step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

        xb, yb = get_batch(train_data_cp, block_size, batch_size=batch_size)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    final_eval = estimate_loss(model, train_data_cp, val_data_cp, block_size, eval_iters)
    print(f"=== [{cp}] 학습 완료: train {final_eval['train']:.4f}, val {final_eval['val']:.4f} ===")

    torch.save(model.state_dict(), f"gpt_model_bpe_{cp}.pt")

    all_results[cp] = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "steps": steps_logged,
        "final_train": final_eval['train'].item(),
        "final_val": final_eval['val'].item(),
        "vocab_size": vocab_size_cp,
    }
# %%
compression_ratios = {500: 2.490, 1000: 2.946, 2000: 3.498}

print(f"{'체크포인트':<12} {'토큰당 val loss':<18} {'압축률':<10} {'글자당 val loss':<15}")
for cp in checkpoints:
    r = all_results[cp]
    char_loss = r['final_val'] / compression_ratios[cp]
    print(f"{cp:<12} {r['final_val']:<18.4f} {compression_ratios[cp]:<10} {char_loss:<15.4f}")

# %%
n_repeats = 2
all_results_repeated = {}

for cp in checkpoints:
    for rep in range(n_repeats):
        print(f"\n=== {cp} 병합 모델, 반복 {rep+1}/{n_repeats} 학습 시작 ===")

        train_data_cp = encoded_by_cp[cp]["train_data"]
        val_data_cp = encoded_by_cp[cp]["val_data"]
        vocab_size_cp = encoded_by_cp[cp]["vocab_size"]

        model = GPTModel(vocab_size=vocab_size_cp, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

        for it in range(max_iters):
            if it % eval_interval == 0:
                losses = estimate_loss(model, train_data_cp, val_data_cp, block_size, eval_iters)
                if it % 4000 == 0:
                    print(f"  [{cp}-{rep}] step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

            xb, yb = get_batch(train_data_cp, block_size, batch_size=batch_size)
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        final_eval = estimate_loss(model, train_data_cp, val_data_cp, block_size, eval_iters)
        print(f"=== [{cp}-{rep}] 완료: train {final_eval['train']:.4f}, val {final_eval['val']:.4f} ===")

        all_results_repeated[(cp, rep)] = {
            "final_train": final_eval['train'].item(),
            "final_val": final_eval['val'].item(),
        }
# %%
compression_ratios = {500: 2.490, 1000: 2.946, 2000: 3.498}

print(f"{'체크포인트':<12} {'토큰당 val loss':<18} {'압축률':<10} {'글자당 val loss':<15}")
for cp in checkpoints:
    for rep in range(n_repeats):
        r = all_results_repeated[(cp,rep)]
        char_loss = r['final_val'] / compression_ratios[cp]
        print(f"{cp:<12} {r['final_val']:<18.4f} {compression_ratios[cp]:<10} {char_loss:<15.4f}")

# %%
import torch

grouped = {500: [1.3518, 1.3488], 1000: [1.3354, 1.3092], 2000: [1.2805, 1.2915]}

stats = {}
for cp, vals in grouped.items():
    t = torch.tensor(vals)
    mean = t.mean().item()
    std = t.std().item()
    se = std / (len(vals) ** 0.5)
    stats[cp] = (mean, se)
    print(f"{cp}: 평균 {mean:.4f}, SE {se:.4f}")
# %%
