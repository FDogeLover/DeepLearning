import re


def clean_review(text):
    """清洗评价内容：去除过短/包含**的无效评价"""
    # 过滤条件：至少10个字符且不包含"**"模式
    return len(text) >= 10 and '**' not in text


def extract_reviews(file_path, min_review_length=10):
    """
    从文件提取并清洗评价数据

    参数:
        file_path: 输入文件路径
        min_review_length: 有效评价的最小长度

    返回:
        list: 清洗后的评价数据 [{'model':..., 'review':...}]
    """
    results = []
    current_model = None

    # 优化后的正则表达式
    model_pattern = re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日·)(.+?)(\t|$)')
    review_pattern = re.compile(r'参数\d+_文本\s+text\s+(.+?)\s+是')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 提取型号
            if '参数' in line and '_文本' in line and '年' in line and '月' in line:
                model_match = model_pattern.search(line)
                if model_match:
                    current_model = model_match.group(2).strip('\t ')

            # 提取评价
            elif '参数' in line and '_文本' in line and 'text' in line:
                review_match = review_pattern.search(line)
                if review_match and current_model:
                    review_text = review_match.group(1).strip()

                    # 初次过滤
                    if not review_text.startswith(('有用(', '商家回复', '更多')):
                        # 二次过滤（长度和内容）
                        if clean_review(review_text):
                            results.append({
                                'model': current_model,
                                'review': review_text
                            })

    return results


# 使用示例
input_file = '数据.txt'
output_file = 'cleaned_reviews.txt'

print(f"正在处理文件: {input_file}")
extracted_data = extract_reviews(input_file)

# 打印统计信息
print(f"共提取到 {len(extracted_data)} 条有效评价")

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, item in enumerate(extracted_data, 1):
        f.write(f"评价 {idx}:\n")
        f.write(f"型号: {item['model']}\n")
        f.write(f"内容: {item['review']}\n\n")

print(f"清洗后的结果已保存到: {output_file}")

# 打印前5条样本
print("\n样本数据预览:")
for item in extracted_data[:5]:
    print(f"型号: {item['model']}")
    print(f"内容: {item['review'][:50]}...\n")