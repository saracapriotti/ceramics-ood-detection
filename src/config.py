import os

# -- Architecture ---------------------------------------------------------------
#   "dvit"      DynamicViT (the model of the paper)
#   "resnet50"  torchvision ResNet50
#   "vit"       timm ViT, here DeiT-small (see TIMM_MODEL in the presets at the end)
# Pick it from the shell, without editing this file:
#     ARCH=resnet50 python run_cv.py --name whole_halves
# Each architecture writes to its own checkpoints_<arch>/ and outputs_<arch>/
# folders (the D-ViT keeps the plain checkpoints/ and outputs/), so runs never
# overwrite each other. Use the SAME ARCH for all the scripts of one pipeline.
ARCH       = os.environ.get("ARCH", "dvit").lower()
TIMM_MODEL = "vit_base_patch16_224"   # only used when ARCH == "vit"
_SUFFIX    = "" if ARCH == "dvit" else f"_{ARCH}"

# -- Paths ----------------------------------------------------------------------
# By default everything is resolved relative to the repository root (the
# parent of src/). DATASET_DIR and DYNAMICVIT_REPO can be overridden with the
# environment variables of the same name.
REPO_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR     = os.environ.get("DATASET_DIR", os.path.join(REPO_DIR, "data"))
DYNAMICVIT_REPO = os.environ.get("DYNAMICVIT_REPO", os.path.join(REPO_DIR, "DynamicViT"))

CSV_FILE        = os.path.join(DATASET_DIR, "dataset.csv")
OOD_CSV_FILE    = os.path.join(DATASET_DIR, "OOD-DATASET.csv")
CHECKPOINT_DIR  = os.path.join(REPO_DIR, "checkpoints" + _SUFFIX)
OUTPUT_DIR      = os.path.join(REPO_DIR, "outputs" + _SUFFIX)
PRETRAINED_PATH = os.path.join(REPO_DIR, "pretrained", "dynamic-vit_lv-m_r0.7.pth")

# -- Reproducibility ------------------------------------------------------------
SEED = 100

# -- Data loading ---------------------------------------------------------------
BATCH_SIZE  = 64
NUM_WORKERS = 4

# -- Preprocessing --------------------------------------------------------------
# VIEW_MODE:
#   "whole+halves" -> whole image + 2 square halves (3 views), used in the paper:
#                     the whole view preserves the overall distribution of
#                     inclusions, the halves provide detail; together they
#                     cover the full surface
#   "halves"       -> the 2 halves only
#   "whole"        -> the whole image only
VIEW_MODE     = "whole+halves"
N_HALVES      = 2
WHOLE_VIEW    = "pad"   # "pad" (letterbox to square) or "squash"
P_WHOLE_TRAIN = 0.34    # probability of sampling the whole view during training

RESIZE_SHORT_SIDE = 448
CROP_SIZE         = 224
EVAL_CHUNK_SIZE   = 128  # views processed per forward pass at evaluation

COLOR_JITTER    = dict(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
GAUSSIAN_BLUR_P = 0.5

# Pre-resized images (see preresize_images.py): same results as the original
# 2048x1536 TIFs, much faster loading.
ORIGINAL_IMG_DIR     = os.path.join(DATASET_DIR, "final_dataset")
ORIGINAL_OOD_IMG_DIR = os.path.join(DATASET_DIR, "OOD-DATASET")
RESIZED_IMG_DIR      = os.path.join(DATASET_DIR, f"final_dataset_short{RESIZE_SHORT_SIDE}")
RESIZED_OOD_IMG_DIR  = os.path.join(DATASET_DIR, f"OOD-DATASET_short{RESIZE_SHORT_SIDE}")
USE_PRERESIZED       = True  # set to False if preresize_images.py has not been run

if USE_PRERESIZED:
    IMG_DIR, OOD_IMG_DIR = RESIZED_IMG_DIR, RESIZED_OOD_IMG_DIR
else:
    IMG_DIR, OOD_IMG_DIR = ORIGINAL_IMG_DIR, ORIGINAL_OOD_IMG_DIR

# -- ImageNet normalisation -----------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# -- Model architecture (DynamicViT, LV-ViT-M backbone) -------------------------
PATCH_SIZE   = 16
EMBED_DIM    = 512
DEPTH        = 20
NUM_HEADS    = 8
MLP_RATIO    = 3.0
P_EMB        = "4_2"
SKIP_LAM     = 2.0
RETURN_DENSE = True
MIX_TOKEN    = True

BASE_KEEP_RATE = 0.8
PRUNING_LOC    = [7, 12, 17]
TOKEN_RATIO    = [BASE_KEEP_RATE, BASE_KEEP_RATE ** 2, BASE_KEEP_RATE ** 3]

# -- Training -------------------------------------------------------------------
NUM_EPOCHS    = 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 0.01
PATIENCE      = 25   # equal to NUM_EPOCHS: early stopping effectively disabled

LABEL_SMOOTHING  = 0.1
LR_WARMUP_EPOCHS = 2
LR_MIN_FACTOR    = 0.01
LAMBDA_RATIO     = 2.0   # weight of the token keep-ratio loss

# -- Out-of-distribution detection ----------------------------------------------
OOD_PERCENTILE                 = 95    # threshold used for the summary printed by run_ood_cv.py
ODIN_TEMPERATURE               = 1000
COVARIANCE_SHRINKAGE           = True  # Ledoit-Wolf
OOD_CLUSTER_DISTANCE_THRESHOLD = 0.15  # cosine distance, within-fold cut in consensus.py

# -- Per-architecture settings ----------------------------------------------------
# Everything not listed here (data, preprocessing, epochs, seed, ...) is identical
# for all the architectures, on purpose: the backbone is the only thing that
# should change. These are the values used for the paper.
_PRESETS = {
    "dvit": {},
    "resnet50": dict(BATCH_SIZE=32,         # ResNet50 needs more memory
                     CROP_SIZE=256,
                     LEARNING_RATE=3e-4,
                     LAMBDA_RATIO=0,        # no token pruning, so no keep-ratio loss
                     OOD_PERCENTILE=80),
    "vit": dict(TIMM_MODEL="deit_small_patch16_224",
                # CROP_SIZE must stay 224: the DeiT-small patch16 only accepts that
                # resolution (timm stops with a positional-embedding size error)
                LAMBDA_RATIO=0,
                OOD_PERCENTILE=85),
}
if ARCH not in _PRESETS:
    raise ValueError(f"Unknown ARCH={ARCH!r}. Available: {sorted(_PRESETS)}")
globals().update(_PRESETS[ARCH])
