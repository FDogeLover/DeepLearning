import requests
import json
import time

url = "https://api.m.jd.com/client.action"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
    "Referer": "https://item.jd.com/100156822392.html",
    "Origin": "https://item.jd.com",
    "Content-Type": "application/x-www-form-urlencoded",
    "Cookie": "你的完整Cookie，建议使用一个已登录账号的Cookie",
}

params = {
    "functionId": "getCommentListWithCard",
    "appid": "item-v3",
    "body": json.dumps({
        "skuId": "100156822392",
        "page": 0,
        "pageSize": 10,
        "sortType": 5,
        "filterType": 0
    })
}

response = requests.post(url, headers=headers, data=params)

try:
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print("请求失败或返回内容无法解析：", e)
    print("状态码：", response.status_code)
    print("返回内容：", response.text)
