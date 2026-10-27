"""
Create downscaled copies of the images (short side = RESIZE_SHORT_SIDE,
aspect ratio preserved, Lanczos filter). Run once.

File names and folder structure match the originals, so the CSV files do
not need to change. Decoding 2048x1536 TIFs every epoch slows training
considerably; since images are resized anyway, the downscaled copies give
the same results with much faster loading.

The original images are NOT modified.

Usage (from the project folder):
    python preresize_images.py
    python preresize_images.py --overwrite     # regenerate existing files too
"""

import argparse
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm

import config


def resize_short_side(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    if min(w, h) <= short:
        return img  # never upscale
    scale = short / min(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


def save_like_original(img: Image.Image, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        img.save(path, compression="tiff_lzw")
    elif ext in (".jpg", ".jpeg"):
        img.save(path, quality=95, subsampling=0)
    else:
        img.save(path)


def process(file_list, src_dir, dst_dir, short, overwrite):
    n_done = n_skip = n_err = 0
    for rel in tqdm(file_list, desc=os.path.basename(dst_dir)):
        rel = str(rel).replace("\\", "/")
        src = os.path.join(src_dir, rel)
        dst = os.path.join(dst_dir, rel)
        if os.path.exists(dst) and not overwrite:
            n_skip += 1
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                save_like_original(resize_short_side(im, short), dst)
            n_done += 1
        except Exception as e:
            print(f"ERROR {src}: {e}")
            n_err += 1
    print(f"{dst_dir}: created {n_done}, already present {n_skip}, errors {n_err}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    short = config.RESIZE_SHORT_SIDE

    df = pd.read_csv(config.CSV_FILE, delimiter=";", encoding="latin-1")
    df.columns = [c.replace("\u00ef\u00bb\u00bf", "") for c in df.columns]  # strip UTF-8 BOM
    process(df[df.columns[0]].tolist(), config.ORIGINAL_IMG_DIR, config.RESIZED_IMG_DIR, short, args.overwrite)

    if os.path.exists(config.OOD_CSV_FILE):
        ood = pd.read_csv(config.OOD_CSV_FILE, sep=";")
        ood.columns = [c.strip().replace("\u00ef\u00bb\u00bf", "") for c in ood.columns]
        process(ood[ood.columns[0]].tolist(), config.ORIGINAL_OOD_IMG_DIR, config.RESIZED_OOD_IMG_DIR, short, args.overwrite)
    else:
        print(f"OOD CSV not found ({config.OOD_CSV_FILE}): skipping OOD images.")


if __name__ == "__main__":
    main()
