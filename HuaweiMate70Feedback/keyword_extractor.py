# keyword_extractor.py
import jieba.analyse
import pandas as pd

def extract_keywords(csv_path, top_k=100):
    df = pd.read_csv(csv_path)
    all_text = " ".join(df["content"].dropna().astype(str).tolist())
    keywords = jieba.analyse.extract_tags(all_text, topK=top_k, withWeight=True)
    with open("data/top_keywords.txt", "w", encoding="utf-8") as f:
        for word, weight in keywords:
            f.write(f"{word}\t{weight:.4f}\n")
    print("关键词提取完成")

if __name__ == "__main__":
    extract_keywords("data/comments.csv")
