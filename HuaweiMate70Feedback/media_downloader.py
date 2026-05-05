# media_downloader.py
import os, requests
from tqdm import tqdm
import pandas as pd

def download_images(df, save_dir="data/images"):
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for images in tqdm(df["images"]):
        for url in images if isinstance(images, list) else []:
            try:
                ext = url.split(".")[-1]
                img_data = requests.get("https:" + url).content
                with open(f"{save_dir}/{count}.{ext}", "wb") as f:
                    f.write(img_data)
                count += 1
            except:
                continue

def download_videos(df, save_dir="data/videos"):
    os.makedirs(save_dir, exist_ok=True)
    for i, url in enumerate(df["video"]):
        if url:
            try:
                video_data = requests.get("https:" + url).content
                with open(f"{save_dir}/video_{i}.mp4", "wb") as f:
                    f.write(video_data)
            except:
                continue

if __name__ == "__main__":
    df = pd.read_csv("data/comments.csv")
    download_images(df)
    download_videos(df)
