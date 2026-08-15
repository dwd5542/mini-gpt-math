
import unicodedata


with open("math_theorems_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)

print(f"전체 글자 수: {len(text)}")
print(f"고유 글자 종류 수 (vocab_size): {vocab_size}")
print(chars)

from collections import Counter

char_counts = Counter(text)
sorted_counts = sorted(char_counts.items(), key=lambda x: x[1])

# for ch, count in sorted_counts[:100]:
#     print(repr(ch), count)

# print(sorted_counts[60:150])

# for threshold in [1, 2, 5, 10, 50, 100]:
#     rare = [ch for ch, cnt in char_counts.items() if cnt < threshold]
#     print(f"threshold={threshold}: 제거될 글자 종류 = {len(rare)}개, 남는 vocab_size = {vocab_size - len(rare)}")

math_whitelist = ['∏', '⇔', '≅', '≺', '∑', '⊇', '∥', '↔', '↦', '⇒',
                   '∃', '⊃', '⊥', '⌊', '⌋', '⊕', '⊗', '∉', '∀',
                   'ℝ', 'ℤ', 'ℚ',
                   'ℏ', 'ℜ', 'ℰ', 'ℵ','⇕', '𝜏']

sm_symbols = set(ch for ch in chars if unicodedata.category(ch) == 'Sm')

threshold = 5
final_vocab = set(ch for ch, cnt in char_counts.items() if cnt >= threshold) | set(math_whitelist) | sm_symbols
removed = set(chars) - final_vocab

print(f"최종 vocab_size: {len(final_vocab)}")
print(f"UNK로 바뀔 글자 수: {len(removed)}")
print(sorted(removed))

UNK_TOKEN = '▯'
assert UNK_TOKEN not in final_vocab, "UNK 후보가 이미 vocab에 있음! 다른 걸 골라야 함"
print("UNK 토큰으로 안전하게 사용 가능:", UNK_TOKEN)

final_vocab.add(UNK_TOKEN)

text_cleaned=''.join(ch if ch in final_vocab else UNK_TOKEN for ch in text)

num_replaced=sum(1 for ch in text if ch not in final_vocab)
print(f"전체 글자 수: {len(text_cleaned)}")
print(f"UNK로 바뀐 글자 개수: {num_replaced}")
print(f"최종 vocab_size (UNK 포함): {len(final_vocab)}")

sorted_vocab = sorted(final_vocab)

stoi = {ch: i for i, ch in enumerate(sorted_vocab)}
itos = {i: ch for i, ch in enumerate(sorted_vocab)}

print(stoi['a'])
print(itos[stoi['a']])

import json

with open("math_theorems_final.txt", "w", encoding="utf-8") as f:
    f.write(text_cleaned)

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)

print("저장 완료")