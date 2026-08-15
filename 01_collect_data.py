import wikipediaapi
import time

wiki=wikipediaapi.Wikipedia(
    user_agent="mini-gpt-math-progject (learning purposes)",
    language="en"
)

C="Category:Mathematical theorems"

def get_article_titles(category_title):
    page=wiki.page(category_title)
    members=page.categorymembers
    titles=[title for title, p in members.items() if p.ns==0]
    return titles

def get_subcategory_titles(category_title):
    page=wiki.page(category_title)
    members=page.categorymembers
    subcats=[title for title,p in members.items() if p.ns==14]
    return subcats

subcats=get_subcategory_titles(C)
all_titles=set(get_article_titles(C))

for subcat in subcats:
    all_titles.update(get_article_titles(subcat))

print(f'total {len(all_titles)}th documents title collect')

with open("math_theorems.txt","w",encoding="utf-8") as f:
    for i,title in enumerate(all_titles):
        try:
            page=wiki.page(title)
            text=page.text
            f.write(text)
            f.write("\n\n=== END OF DOCUMENT ===\n\n")
        except Exception as e:
            print(f'fail: {title} ({e})')
        if i%50==0:
            print(f"{i}/{len(all_titles)} progressing")

        time.sleep(0.1)

print("end")