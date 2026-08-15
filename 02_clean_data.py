import re

text = open("math_theorems.txt", encoding="utf-8").read()

print("전체 글자 수:", len(text))
print("'{\\displaystyle' 등장 횟수:", text.count("{\\displaystyle"))
print("'== References ==' 등장 횟수:", text.count("== References =="))

pattern = re.compile(r"(\n\s*){3,}\{\\displaystyle.*?\}\n", re.DOTALL)
matches = pattern.findall(text)

matches_full = re.findall(r"(?:\n\s*){3,}\{\\displaystyle.*?\}\n", text, re.DOTALL)
sum1=sum(len(m) for m in matches_full)

print("매칭 개수:", len(matches_full))
print("총 글자 수:", sum1)
print("전체 대비 비율: {:.2%}".format(sum1 / len(text)))

print("--- 샘플 5개 ---")
for m in matches_full[:5]:
    print(repr(m))
    print("=====")

lines = text.split("\n")
lengths = [len(line) for line in lines]

import numpy as np
lengths = np.array(lengths)

indented_lines = [line for line in lines if re.match(r"^ {2,}\S", line)]
print("2칸 이상 들여쓰기로 시작하는 줄 수:", len(indented_lines))

print("--- 무작위로 중간 지점에서 10개 샘플 ---")
mid = len(indented_lines) // 2
for l in indented_lines[mid:mid+10]:
    print(repr(l))

normal_looking = [line for line in indented_lines if len(line.strip()) > 15 and " " in line.strip()]
print("15자 넘고 공백(단어 구분)도 있는 줄 수:", len(normal_looking))

for l in normal_looking[:10]:
    print(repr(l))

cleaned_lines = [line for line in lines if not re.match(r"^ {2,}\S", line)]
cleaned_text = "\n".join(cleaned_lines)

print("정리 전 글자 수:", len(text))
print("정리 후 글자 수:", len(cleaned_text))
print("제거된 비율: {:.2%}".format((len(text) - len(cleaned_text)) / len(text)))


final_text = re.sub(r"[ \t]{2,}", "", cleaned_text)
final_text = re.sub(r"\n{3,}", "\n\n", final_text)

print("정리 전:", len(cleaned_text))
print("정리 후:", len(final_text))

print("제거된 비율: {:.2%}".format((len(text) - len(final_text)) / len(text)))

idx = final_text.find("bipartite graphs")
print(repr(final_text[idx:idx+150]))

print("최종 글자 수:", len(final_text))
print("남은 displaystyle 개수:", final_text.count("{\\displaystyle"))
print("연속 3줄바꿈 이상 남은 개수:", len(re.findall(r"\n{3,}", final_text)))

with open("math_theorems_cleaned.txt", "w", encoding="utf-8") as f:
    f.write(final_text)