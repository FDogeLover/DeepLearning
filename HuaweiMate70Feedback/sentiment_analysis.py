# sentiment_analysis.py
from snownlp import SnowNLP
import pandas as pd

def analyze_sentiment(csv_path):
    df = pd.read_csv(csv_path)
    df["sentiment"] = df["content"].apply(lambda x: SnowNLP(str(x)).sentiments)
    df["label"] = df["sentiment"].apply(lambda s: "正面" if s > 0.6 else "负面" if s < 0.4 else "中性")
    df.to_csv("data/comments_with_sentiment.csv", index=False, encoding="utf-8-sig")
    print("情感分析完成")

if __name__ == "__main__":
    analyze_sentiment("data/comments.csv")
