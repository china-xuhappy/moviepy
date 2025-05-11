# import cv2
# import imagehash
# from PIL import Image
#
def get_video_hashes(path, frame_gap=30):
    cap = cv2.VideoCapture(path)
    hashes = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_gap)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hash_val = imagehash.phash(img)
        hashes.append(hash_val)
    cap.release()
    return hashes

def compare_hashes(hashes1, hashes2, threshold=10):
    matched = 0
    total = min(len(hashes1), len(hashes2))
    for h1, h2 in zip(hashes1, hashes2):
        if h1 - h2 <= threshold:  # 汉明距离容忍差异
            matched += 1
    return matched / total if total > 0 else 0

import argparse
import cv2
import imagehash
import numpy as np
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel
import torch
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# 加载 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def extract_key_frames(video_path, step=30, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % step == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(pil_image)
        frame_count += 1
    cap.release()
    return frames


def get_phash_similarity(frames1, frames2):
    return compare_hashes(frames1, frames2)


def get_clip_similarity(frames1, frames2):
    images1 = clip_processor(images=frames1, return_tensors="pt").pixel_values
    images2 = clip_processor(images=frames2, return_tensors="pt").pixel_values
    with torch.no_grad():
        emb1 = clip_model.get_image_features(images1).cpu().numpy()
        emb2 = clip_model.get_image_features(images2).cpu().numpy()
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    sim = np.mean(np.dot(emb1, emb2.T).diagonal())
    return sim


def extract_text_features(frames):
    texts = []
    for f in frames:
        text = pytesseract.image_to_string(f, lang='eng')
        texts.append(text.strip())
    return texts


def text_similarity(texts1, texts2):
    matches = sum(t1 == t2 for t1, t2 in zip(texts1, texts2))
    return matches / max(len(texts1), len(texts2))


def compare_videos(video1, video2):
    frames1 = extract_key_frames(video1)
    frames2 = extract_key_frames(video2)

    print("👉 正在计算感知哈希相似度...")
    phash_sim = get_phash_similarity(frames1, frames2)

    print("👉 正在计算 CLIP 语义相似度...")
    clip_sim = get_clip_similarity(frames1, frames2)

    print("👉 正在进行 OCR 字幕检测...")
    texts1 = extract_text_features(frames1)
    texts2 = extract_text_features(frames2)
    text_sim = text_similarity(texts1, texts2)

    print("\n📊 综合相似度检测结果：")
    print(f"🧠 感知哈希（视觉）：{phash_sim:.2%}")
    print(f"🗣️ 语义相似度（CLIP）：{clip_sim:.2%}")
    print(f"🔤 OCR 文本相似度：{text_sim:.2%}")


# hashes1 = get_video_hashes("./materials/material.mp4")
# hashes2 = get_video_hashes("./previews/preview_7_template_video.mp4")
#
# similarity = compare_hashes(hashes1, hashes2)
# print(f"相似度（容差模式）: {similarity:.2%}")
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="视频相似度比对工具")
    # parser.add_argument("video1", help="原视频路径")
    # parser.add_argument("video2", help="对比视频路径")
    # args = parser.parse_args()

    compare_videos("./materials/material.mp4", "./previews/preview_7_template_video.mp4")