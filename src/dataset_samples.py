import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageStat
from torch.utils.data import Dataset
from torchvision import transforms

import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_sample_id(filename: str) -> str:
    """Extract the sample ID from a filename, stripping magnification,
    polarization tokens and trailing suffixes."""
    basename = os.path.basename(filename)
    name = re.sub(r"\.(png|jpg|jpeg|tif|tiff)$", "", basename, flags=re.IGNORECASE)
    name = re.sub(r"[._!-]\d+$|[._!]+$", "", name)

    token_pattern = r"(?:[._-]|\b)(?:2[,._]5-?X?|4X|10X|20X|40X|50X|210X|PPL|XPL|PL|PX)(?=[._-]|\b)"
    cleaned = re.sub(token_pattern, "", name, flags=re.IGNORECASE)
    cleaned = re.sub(token_pattern, "", cleaned, flags=re.IGNORECASE)

    return cleaned.strip("-_ .")


def extract_polarization(filename: str) -> str:
    """Extract the polarization (PPL/XPL) from a filename. PL -> PPL, PX -> XPL."""
    basename = os.path.basename(filename)
    name = re.sub(r"\.(png|jpg|jpeg|tif|tiff)$", "", basename, flags=re.IGNORECASE)
    match = re.search(r"(?:[._-]|\b)(PPL|XPL|PX|PL)(?=[._-]|\b)", name, flags=re.IGNORECASE)
    if not match:
        return "UNKNOWN"
    token = match.group(1).upper()
    if token in ("PPL", "PL"):
        return "PPL"
    if token in ("XPL", "PX"):
        return "XPL"
    return "UNKNOWN"


def load_dataset_csv(csv_file: str = None) -> pd.DataFrame:
    """Read the training CSV (IMAGE_ID;CLASS) and add sample_id and polarization."""
    df = pd.read_csv(csv_file or config.CSV_FILE, delimiter=";", encoding="latin-1")
    df.columns = [c.replace("\u00ef\u00bb\u00bf", "") for c in df.columns]  # strip UTF-8 BOM
    col_img, col_label = df.columns[0], df.columns[1]
    df = df.dropna(subset=[col_label]).copy()
    df[col_label] = df[col_label].astype(int)
    df["sample_id"] = df[col_img].apply(extract_sample_id)
    df["polarization"] = df[col_img].apply(extract_polarization)
    return df


class FabricsDataset(Dataset):
    """Images and 0-based fabric labels from a dataframe whose first two
    columns are the image path (relative to img_dir) and the 1-based class."""

    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.copy().reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        col_img = self.df.columns[0]
        col_label = self.df.columns[1]

        if "sample_id" not in self.df.columns:
            self.df["sample_id"] = self.df[col_img].apply(extract_sample_id)
        if "polarization" not in self.df.columns:
            self.df["polarization"] = self.df[col_img].apply(extract_polarization)

        self.image_paths = self.df[col_img].apply(
            lambda x: os.path.join(img_dir, str(x).replace("\\", "/"))
        )
        self.labels = self.df[col_label]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths.iloc[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            label = int(self.labels.iloc[idx]) - 1  # labels in the CSV are 1-based
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, config.CROP_SIZE, config.CROP_SIZE), -1


# -- Transform building blocks --------------------------------------------------

class RandomRot90:
    """Random rotation by 0/90/180/270 degrees.

    Thin sections have no canonical orientation, and unlike arbitrary-angle
    rotations this introduces neither black corners nor interpolation.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        return img if k == 0 else img.rotate(90 * k, expand=True)


class PadToSquare:
    """Pad the image to a square without distorting or cropping it.

    By default the padding uses the image's MEAN colour rather than black:
    in XPL a black band could be mistaken for minerals at extinction.
    """

    def __init__(self, fill: str = "mean"):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        side = max(w, h)
        if self.fill == "mean":
            colour = tuple(int(round(c)) for c in ImageStat.Stat(img).mean[:3])
        else:
            colour = (0, 0, 0)
        canvas = Image.new("RGB", (side, side), colour)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def _resize_short_side(img: Image.Image, short: int, min_side: int) -> Image.Image:
    w, h = img.size
    scale = short / min(w, h)
    new = (max(min_side, round(w * scale)), max(min_side, round(h * scale)))
    return img.resize(new, Image.BICUBIC)


def _square_crops(img: Image.Image, n: int) -> list:
    """Return n SQUARE crops (side = short side) spread along the long side.

    With n=2 on a 4:3 or 16:9 image the two crops overlap in the middle and
    together cover the entire surface.
    """
    w, h = img.size
    side = min(w, h)
    if n <= 1:
        x0 = (w - side) // 2
        return [img.crop((x0, 0, x0 + side, side))]
    xs = np.linspace(0, w - side, n).round().astype(int)
    ys = np.linspace(0, h - side, n).round().astype(int) if h > w else [0] * n
    if w >= h:
        return [img.crop((int(x), 0, int(x) + side, side)) for x in xs]
    return [img.crop((0, int(y), side, int(y) + side)) for y in ys]


class MultiViewEval:
    """Evaluation transform: PIL image -> tensor [K, 3, C, C] with the views
    selected by config.VIEW_MODE.

      whole   -> the WHOLE image (padded to square, or squashed if
                 WHOLE_VIEW='squash'): global context, nothing cropped
      halves  -> N_HALVES square crops covering the full surface: fine
                 detail at the short-side scale

    MultiCropWrapper (model.py) averages logits and features over the K views,
    so each image still yields a single prediction.
    """

    def __init__(self, short_side, crop, mode, n_halves=2, whole_view="pad"):
        self.short_side = short_side
        self.crop = crop
        self.mode = mode
        self.n_halves = n_halves
        self.whole_view = whole_view
        self.pad = PadToSquare()
        self.finish = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])

    def __call__(self, img: Image.Image) -> torch.Tensor:
        views = []
        if "whole" in self.mode:
            base = self.pad(img) if self.whole_view == "pad" else img
            views.append(base.resize((self.crop, self.crop), Image.BICUBIC))
        if "halves" in self.mode:
            resized = _resize_short_side(img, self.short_side, self.crop)
            views += [c.resize((self.crop, self.crop), Image.BICUBIC)
                      for c in _square_crops(resized, self.n_halves)]
        if not views:
            raise ValueError(f"VIEW_MODE='{self.mode}' produces no views.")
        return torch.stack([self.finish(v) for v in views])


class RandomView:
    """Training transform consistent with MultiViewEval.

    Each call randomly returns either the whole image (with probability
    P_WHOLE_TRAIN) or a square crop at a random position, at the same scale
    used for evaluation, so the model learns both points of view.
    """

    def __init__(self, short_side_choices, crop, mode, p_whole, whole_view="pad"):
        self.short_side_choices = short_side_choices
        self.crop = crop
        self.mode = mode
        self.p_whole = p_whole
        self.whole_view = whole_view
        self.pad = PadToSquare()

    def __call__(self, img: Image.Image) -> Image.Image:
        use_whole = ("whole" in self.mode) and (
            "halves" not in self.mode or random.random() < self.p_whole
        )
        if use_whole:
            base = self.pad(img) if self.whole_view == "pad" else img
            return base.resize((self.crop, self.crop), Image.BICUBIC)

        # random.choice is kept even with a single scale: it draws from the
        # random stream, so removing it would change the training results.
        short = random.choice(self.short_side_choices)
        resized = _resize_short_side(img, short, self.crop)
        w, h = resized.size
        side = min(w, h)
        x = random.randint(0, w - side)
        y = random.randint(0, h - side)
        return resized.crop((x, y, x + side, y + side)).resize((self.crop, self.crop), Image.BICUBIC)


def get_transforms():
    """Return (train_transform, val_transform) according to config.VIEW_MODE."""
    C = config.CROP_SIZE
    S = config.RESIZE_SHORT_SIDE
    mode = config.VIEW_MODE

    train_steps = [
        RandomView([S], C, mode, config.P_WHOLE_TRAIN, config.WHOLE_VIEW),
        RandomRot90(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    val_transform = MultiViewEval(S, C, mode, config.N_HALVES, config.WHOLE_VIEW)

    if config.COLOR_JITTER and any(v > 0 for v in config.COLOR_JITTER.values()):
        train_steps.append(transforms.ColorJitter(**config.COLOR_JITTER))
    if config.GAUSSIAN_BLUR_P > 0:
        train_steps.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=(5, 5))], p=config.GAUSSIAN_BLUR_P))
    train_steps += [
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ]
    return transforms.Compose(train_steps), val_transform


def describe_preprocessing() -> str:
    mode = config.VIEW_MODE
    n = (1 if "whole" in mode else 0) + (config.N_HALVES if "halves" in mode else 0)
    return (f"{mode} ({n} views), short side {config.RESIZE_SHORT_SIDE}, "
            f"whole view: {config.WHOLE_VIEW}")


def collate_fn(batch):
    """Drop images that failed to load (label == -1)."""
    batch = [(x, y) for x, y in batch if y >= 0]
    if not batch:
        return torch.empty(0, 3, config.CROP_SIZE, config.CROP_SIZE), torch.empty(0, dtype=torch.int64)
    return torch.utils.data.default_collate(batch)


# -- Split (leave-one-fabric-out experiment) ------------------------------------

def split_by_samples(df: pd.DataFrame, seed: int = config.SEED, rare_threshold: int = 8):
    """Stratified train/val/test split at the SAMPLE level, so that images of
    the same sample never end up in different splits.

    Classes with a single sample go to train only; rare classes
    (<= rare_threshold samples) get one sample in test and one in val;
    the others are split ~70/15/15.
    """
    col_img = df.columns[0]
    col_label = df.columns[1]

    df["sample_id"] = df[col_img].apply(extract_sample_id)
    df["polarization"] = df[col_img].apply(extract_polarization)

    all_sample_ids = df["sample_id"].unique()
    sample_id_to_class = df.groupby("sample_id")[col_label].first().to_dict()

    sample_ids_by_class = defaultdict(list)
    for s_id in all_sample_ids:
        sample_ids_by_class[sample_id_to_class[s_id]].append(s_id)

    train_sample_ids, val_sample_ids, test_sample_ids = set(), set(), set()
    rng = np.random.default_rng(seed)

    for cls in sorted(sample_ids_by_class.keys()):
        unique_groups = np.array(sample_ids_by_class[cls])
        rng.shuffle(unique_groups)
        n_groups = len(unique_groups)

        if n_groups == 1:
            train_sample_ids.add(unique_groups[0])
        elif n_groups <= rare_threshold:
            test_sample_ids.add(unique_groups[0])
            remaining = unique_groups[1:]
            if len(remaining) >= 2:
                val_sample_ids.add(remaining[0])
                train_sample_ids.update(remaining[1:])
            else:
                train_sample_ids.update(remaining)
        else:
            n_test = max(1, int(0.15 * n_groups))
            n_val = max(1, int(0.15 * n_groups))
            test_sample_ids.update(unique_groups[:n_test])
            val_sample_ids.update(unique_groups[n_test:n_test + n_val])
            train_sample_ids.update(unique_groups[n_test + n_val:])

    assert not (train_sample_ids & val_sample_ids)
    assert not (train_sample_ids & test_sample_ids)
    assert not (val_sample_ids & test_sample_ids)

    train_df = df[df["sample_id"].isin(train_sample_ids)].copy()
    val_df = df[df["sample_id"].isin(val_sample_ids)].copy()
    test_df = df[df["sample_id"].isin(test_sample_ids)].copy()
    return train_df, val_df, test_df
