import re
import pandas as pd
from snownlp import SnowNLP
from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 显示中文
matplotlib.rcParams['axes.unicode_minus'] = False             # 正常显示负号

# 中文设计维度关键词词典
dimension_keywords = {
    "相机": ["相机", "拍照", "像素", "照相", "摄影"],
    "屏幕显示": ["屏幕", "显示", "高清", "清晰", "色彩", "饱和度"],
    "外观设计": ["外观", "设计", "时尚", "大气", "配色"],
    "性能": ["性能", "速度", "流畅", "卡顿", "运行", "多任务"],
    "配置": ["内存", "存储", "配置"],
    "电池续航": ["电池", "续航", "待机", "充电"],
    "手感": ["手感", "质感", "丝滑"],
   # "物流": ["物流", "快递", "顺丰", "发货", "到货"],
    "品牌价值感": ["国货", "国产", "华为", "爱国"]
}


# 读取 txt 文件并解析成结构化数据
def parse_feedback_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    feedback_list = []

    for block in blocks:
        id_match = re.search(r'评价\s*(\d+)', block)
        model_match = re.search(r'型号:\s*(.+)', block)
        text_match = re.search(r'内容:\s*(.+)', block, re.DOTALL)

        if id_match and model_match and text_match:
            feedback_list.append({
                "id": int(id_match.group(1)),
                "model": model_match.group(1).strip(),
                "text": text_match.group(1).strip()
            })

    return feedback_list


# 使用 SnowNLP 进行中文情感分析，返回"正向/中性/负向"
def classify_sentiment(text, threshold_pos=0.7, threshold_neg=0.4):
    score = SnowNLP(text).sentiments
    if score >= threshold_pos:
        return "正向"
    elif score <= threshold_neg:
        return "负向"
    else:
        return "中性"


# 情感分析 + 设计维度识别
def analyze_feedback(feedback_data):
    records = []
    for entry in feedback_data:
        text = entry["text"]
        dimensions_found = set()

        for dim, keywords in dimension_keywords.items():
            if any(kw in text for kw in keywords):
                dimensions_found.add(dim)

        sentiment = classify_sentiment(text)

        for dim in dimensions_found:
            records.append({
                "评价编号": entry["id"],
                "型号": entry["model"],
                "设计维度": dim,
                "情感": sentiment,
                "原始内容": text
            })

    return pd.DataFrame(records)


# 生成情感柱状图
def plot_sentiment_bar(df, filename="sentiment_bar.png"):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 设置中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False             # 正常显示负号

    summary = df.groupby(["设计维度", "情感"]).size().unstack(fill_value=0)
    summary.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set3")
    plt.title("各设计维度情感分布")
    plt.ylabel("评价数量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"已保存情感柱状图：{filename}")



# 生成维度词云图
def plot_wordcloud(df, filename="dimension_wordcloud.png"):
    words = " ".join(df["设计维度"].tolist())
    wordcloud = WordCloud(font_path="msyh.ttc", background_color="white", width=800, height=400).generate(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("高频设计维度词云")
    plt.savefig(filename)
    print(f"已保存词云图：{filename}")


# 汇总满意度表
def summarize_sentiment(df):
    summary = df.groupby(["设计维度", "情感"]).size().unstack(fill_value=0)
    summary["满意率"] = (summary.get("正向", 0) / summary.sum(axis=1) * 100).round(1).astype(str) + "%"
    return summary


# 主程序入口
if __name__ == "__main__":
    feedbacks = parse_feedback_file("cleaned_reviews.txt")
    df_result = analyze_feedback(feedbacks)

    print("\n【情感分析结果】：\n")
    print(df_result)

    df_result.to_csv("分析结果_逐条.csv", index=False)
    summarize_sentiment(df_result).to_csv("维度满意度统计.csv")

    plot_sentiment_bar(df_result)
    plot_wordcloud(df_result)

    print("\n所有分析完成，结果已保存为 CSV + 图像文件。")
