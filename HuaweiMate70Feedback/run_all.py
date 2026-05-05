import requests
import csv
import time

def get_comments(product_id, max_pages=10, delay=1):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Referer": f"https://item.jd.com/{product_id}.html"
    }

    all_comments = []

    for page in range(max_pages):
        print(f"正在爬取第 {page + 1} 页评论...")
        url = f"https://club.jd.com/comment/productPageComments.action?productId={product_id}&score=0&sortType=5&page={page}&pageSize=10"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            data = response.json()
            comments = data['comments']
            for comment in comments:
                all_comments.append(comment['content'].strip())
            time.sleep(delay)
        except Exception as e:
            print(f"第 {page + 1} 页获取失败: {e}")
            break

    # 保存为 CSV
    with open("jd_comments.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["评论内容"])
        for comment in all_comments:
            writer.writerow([comment])

    print(f"爬取完毕，共 {len(all_comments)} 条评论，保存为 jd_comments.csv")

if __name__ == "__main__":
    get_comments(product_id="100156822392", max_pages=10)
