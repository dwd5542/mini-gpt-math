# %%
from collections import defaultdict

with open("math_theorems_final.txt", "r", encoding="utf-8") as f:
    text = f.read()

words=text.split()

word_freq=defaultdict(int)
for word in words:
    word_freq[word]+=1

# print("고유 단어 수:", len(word_freq))
# print("가장 흔한 단어 5개:", sorted(word_freq.items(), key=lambda x: -x[1])[:5])

vocab={}
for word,freq in word_freq.items():
    chars=tuple(word)+("</w>",)
    vocab[chars]=freq

# print("변환 예시:", list(vocab.items())[:3])

# %%
def get_pair_freqs(vocab):
    pair_freqs=defaultdict(int)
    for word,freq in vocab.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            pair_freqs[pair]+=freq
    return pair_freqs

def merge_vocab(pair,vocab):
    new_vocab={}
    for word,freq in vocab.items():
        new_word=[]
        i=0
        while i<len(word):
            if i<len(word)-1 and word[i]==pair[0] and word[i+1]==pair[1]:
                new_word.append(word[i]+word[i+1])
                i+=2
            else:
                new_word.append(word[i])
                i+=1
        new_vocab[tuple(new_word)]=freq
    return new_vocab

def get_vocab_size(vocab):
    tokens=set()
    for word in vocab:
        tokens.update(word)
    return len(tokens),tokens

# pair_freqs=get_pair_freqs(vocab)
# # print("고유 쌍 개수:", len(pair_freqs))
# # print("가장 흔한 쌍 5개:", sorted(pair_freqs.items(), key=lambda x: -x[1])[:5])

# best_pair = max(pair_freqs, key=pair_freqs.get)
# print("이번에 병합할 쌍:", best_pair)

# vocab = merge_vocab(best_pair, vocab)
# print("병합 후 가장 흔한 단어 5개:", sorted(vocab.items(), key=lambda x: -x[1])[:5])

# %%
num_merges=500
merges=[]

for i in range(num_merges):
    pair_freqs=get_pair_freqs(vocab)
    if not pair_freqs:
        print("no pair to merge")
        break

    best_pair=max(pair_freqs,key=pair_freqs.get)
    vocab=merge_vocab(best_pair,vocab)
    merges.append(best_pair)

    if i%50==0:
        print(f"{i}번째 병합: {best_pair}")

print("총 병합 횟수:", len(merges))
final_size, final_tokens = get_vocab_size(vocab)
print("최종 vocab_size:", final_size)

# initial_chars = set()
# for word in word_freq:
#     initial_chars.update(tuple(word) + ("</w>",))

# final_size, final_tokens = get_vocab_size(vocab)

# missing = initial_chars - final_tokens
# print("사라진 초기 글자:", missing)

# %%
def tokenize_word(word, merges):
    tokens=list(word)+["</w>"]
    for pair in merges:
        i=0
        new_tokens=[]
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i]==pair[0] and tokens[i+1]==pair[1]:
                new_tokens.append(tokens[i]+tokens[i+1])
                i+=2
            else:
                new_tokens.append(tokens[i])
                i+=1
        tokens=new_tokens
    return tokens

# print(tokenize_word("theory", merges))
# print(tokenize_word("mathematics", merges))
# print(tokenize_word("xyzzyplugh", merges))

fianl_size,final_tokens=get_vocab_size(vocab)

bpe_vocab=sorted(final_tokens)
bpe_stoi={tok:i for i,tok in enumerate(bpe_vocab)}
bpe_itos={i:tok for i,tok in enumerate(bpe_vocab)}

print("BPE vocab_size:", len(bpe_vocab))
print("예시 몇 개:", bpe_vocab[:5])

# %%
def encode_bpe(text,merges,stoi,unk_token='▯'):
    words=text.split()
    ids=[]
    for word in words:
        tokens=tokenize_word(word,merges)
        for tok in tokens:
            if tok in stoi:
                ids.append(stoi[tok])
            else:
                ids.append(stoi.get(unk_token,-1))
    return ids

test_text="theory mathematics xyzzyplugh"
ids = encode_bpe(test_text, merges, bpe_stoi)
print("정수 인코딩 결과:", ids)
print("길이:", len(ids))

decoded_tokens = [bpe_itos[i] for i in ids]
print("다시 디코딩한 토큰들:", decoded_tokens)

# %%
import json 

merges_serializable=[list(pair) for pair in merges]

bpe_data={
    "merges": merges_serializable,
    "stoi": bpe_stoi,
    "itos": {str(k):v for k,v in bpe_itos.items()}
}

with open("bpe_vocab.json", "w", encoding="utf-8") as f:
    json.dump(bpe_data, f, ensure_ascii=False, indent=2)

print("BPE vocab 저장 완료 → bpe_vocab.json")
print("vocab_size:", len(bpe_stoi))

# %%

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

with open("math_theorems_final.txt","r",encoding="utf-8") as f:
    full_text=f.read()

all_ids=encode_bpe_corpus(full_text,merges,bpe_stoi,global_cache)
print("전체 토큰 개수:", len(all_ids))

import torch
data=torch.tensor(all_ids,dtype=torch.long)

n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

torch.save(data,"encoded_data_bpe.pt")
print("train_data length:", len(train_data))
print("val_data length:", len(val_data))

# %%
