# visualize.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def draw_wordcloud(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        text = " ".join([line.split("\t")[0] for line in f.readlines()])
    wc = WordCloud(font_path="msyh.ttc", width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("关键词词云")
    plt.show()

def draw_sentiment_chart(csv_file):
    df = pd.read_csv(csv_file)
    df["label"].value_counts().plot(kind="bar", color=["green", "red", "gray"])
    plt.title("情感分布")
    plt.xticks(rotation=0)
    plt.ylabel("数量")
    plt.show()

if __name__ == "__main__":
    draw_wordcloud("data/top_keywords.txt")
    draw_sentiment_chart("data/comments_with_sentiment.csv")
