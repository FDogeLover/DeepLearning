# config.py
with open("cookie.txt", "r", encoding="utf-8") as f:
    cookie_string = f.read().strip()
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    ),
    "Referer": "https://item.jd.com/100156822392.html",  # 改成目标商品链接
    "Origin": "https://item.jd.com",
    "Content-Type": "application/x-www-form-urlencoded",
    "Cookie": (
        "shshshfp=6f197...; "
        "shshshfpa=17b3...; "
        "shshshfpb=BApXSxU-c0fNAAfHi8KjpPP6_-txQ1kr_BgQYETtp9xJ1MsPkB462; "
        "areaId=15_1213_3038_0;"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Connection": "keep-alive",
    "Host": "api.m.jd.com",
    "X-Referer-Page": "https://item.jd.com/100156822392.html",
    "X-Rp-Client": "h5_1.0.0",
}



