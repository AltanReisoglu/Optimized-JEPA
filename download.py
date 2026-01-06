import os
from datasets import load_dataset
from yt_dlp import YoutubeDL

# =========================
# Ayarlar
# =========================
DATASET_NAME = "friedrichor/MSR-VTT"
CONFIG = "train_9k"        # <-- ZORUNLU
SAVE_DIR = "assets"
MAX_VIDEOS = 50

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Dataset yükle
# =========================
print("📦 MSR-VTT (train_9k) yükleniyor...")
dataset = load_dataset(DATASET_NAME, CONFIG, split="train")

print(f"🎬 Toplam video sayısı: {len(dataset)}")
print(f"⬇️  İlk {MAX_VIDEOS} video indirilecek")

# =========================
# yt-dlp ayarları
# =========================
ydl_opts = {
    "format": "mp4/best",
    "outtmpl": os.path.join(SAVE_DIR, "%(id)s.mp4"),  # dataset id = dosya adı
    "quiet": False,
    "no_warnings": True,
}

# =========================
# Video indirme
# =========================
with YoutubeDL(ydl_opts) as ydl:
    for i, item in enumerate(dataset):
        if i >= MAX_VIDEOS:
            break

        dataset_id = item["id"]   # MSR-VTT id
        url = item["url"]

        output_path = os.path.join(SAVE_DIR, f"{dataset_id}.mp4")
        if os.path.exists(output_path):
            print(f"⏩ Zaten var: {dataset_id}")
            continue

        print(f"⬇️  [{i+1}/{MAX_VIDEOS}] İndiriliyor → {dataset_id}")

        try:
            ydl.download([url])
        except Exception as e:
            print(f"❌ Hata ({dataset_id}): {e}")

print("✅ İlk 50 video indirildi.")
