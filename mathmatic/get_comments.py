import requests
import pandas as pd
import json
from fake_useragent import UserAgent

# ———— 配置 ————
COOKIE = '你的 Cookie 字符串...'
SEARCH_URL = 'https://index.baidu.com/api/SearchApi/index'
PTBK_URL_TEMPLATE = 'https://index.baidu.com/Interface/ptbk?uniqid={uniqid}'

HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Cookie': COOKIE,
    'User-Agent': UserAgent().random,
    # 其余 header 可选……
}

# ———— 辅助函数 ————
def parse_search_params():
    return {
        'area': '0',
        'word': json.dumps([[{"name": "华为mate70", "wordType": 1}]]),
        'startDate': '2025-01-01',
        'endDate': '2025-05-13',
    }

def decrypt(ptbk: str, encrypted_list: list[str]) -> str:
    if not ptbk or len(ptbk) % 2 != 0:
        return ""
    n = len(ptbk) // 2
    mapping = {ptbk[i]: ptbk[i + n] for i in range(n)}
    try:
        return ''.join(mapping[ch] for ch in encrypted_list)
    except KeyError as e:
        print(f"[decrypt] 未找到映射字符: {e}")
        return ""

def fill_zero(val: str) -> int:
    if not val:
        return 0
    try:
        return int(val)
    except ValueError:
        return 0

# ———— 主流程 ————
def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    # 1) 获取加密指数数据
    resp = session.get(SEARCH_URL, params=parse_search_params())
    resp.raise_for_status()
    enc_data = resp.json().get('data', {})
    uniqid = enc_data.get('uniqid', '')
    if not uniqid:
        print("未获取到 uniqid，退出")
        return

    # 2) 获取解密用的 ptbk
    ptbk_resp = session.get(PTBK_URL_TEMPLATE.format(uniqid=uniqid))
    ptbk_resp.raise_for_status()
    ptbk = ptbk_resp.json().get('data', '')

    # 3) 逐条解密并收集
    frames = []
    for ui in enc_data.get('userIndexes', []):
        word = ui['word'][0]['name']
        date_range = pd.date_range(ui['all']['startDate'], ui['all']['endDate']).strftime('%Y-%m-%d')
        all_enc = ui['all']['data']
        pc_enc = ui['pc']['data']
        wise_enc = ui['wise']['data']

        all_vals = [fill_zero(x) for x in decrypt(ptbk, all_enc).split(',')]
        pc_vals = [fill_zero(x) for x in decrypt(ptbk, pc_enc).split(',')]
        wise_vals = [fill_zero(x) for x in decrypt(ptbk, wise_enc).split(',')]

        df = pd.DataFrame({
            '关键词': word,
            '日期': date_range,
            '全部': all_vals,
            '电脑端': pc_vals,
            '移动端': wise_vals
        })
        frames.append(df)

    # 4) 合并 & 写文件
    result = pd.concat(frames, ignore_index=True)
    result.to_csv('result.csv', index=False, encoding='utf-8-sig')
    print("已生成 result.csv，行数：", len(result))

if __name__ == '__main__':
    main()
