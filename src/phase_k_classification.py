"""
Phase K: GPR 이미지 분류 (Classification) 접근법

배경
----
Phase G~J에서 bbox 기반 YOLO 검출 접근법이 pipe/rebar에 실패.
Phase J 진단: Near-domain(같은 MIS 내)에서도 Recall=0 → H2 확정
  → bbox 위치 파악 자체가 ZON마다 달라 불가능

새 접근법
---------
  검출(Detection) = 분류 + 위치 파악  ← pipe/rebar에서 위치 파악이 근본 실패
  분류(Classification) = 분류만        ← bbox 불필요

  EfficientNet-B0 (ImageNet pretrained) → 마지막 FC만 교체
  → "이 GPR 이미지에 pipe/rebar/tunnel 중 무엇이 있는가?" 판단

평가 방식
---------
  Phase G와 동일한 held-out test ZON 풀
  각 이미지 → 예측 클래스가 정답 클래스와 일치하면 TP
  Recall = TP / (TP + FN)  per class

사용법
------
  /c/Python314/python.exe src/phase_k_classification.py
"""

import sys
import json
import random
import shutil
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec,
)

# ── 경로 ──────────────────────────────────────────────────────────────
GZ_DATA    = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST   = BASE_DIR / "guangzhou_labeled/manifest.json"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_k"
MODEL_PATH = BASE_DIR / "models/phase_k_cls.pt"

# ── 상수 ──────────────────────────────────────────────────────────────
CATEGORIES      = ["pipe", "rebar", "tunnel"]
CLS2IDX         = {c: i for i, c in enumerate(CATEGORIES)}
IMG_SIZE        = 224       # EfficientNet 입력 크기
EPOCHS          = 60
LR              = 1e-4
BATCH           = 8
N_TEST_PER_CLS  = 10        # Phase G와 동일
SEED            = 42
DT_SEC          = (8.0 / 512) * 1e-9

# ── 색상 ──────────────────────────────────────────────────────────────
CLS_COLORS = {"pipe": "#3498db", "rebar": "#2ecc71", "tunnel": "#f39c12"}


# ─────────────────────────────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    """IDS .dt → 640×640 BGR uint8. 실패 시 None."""
    try:
        data, _ = read_ids_dt(str(dt_path))
        if data is None or data.shape[1] < 10:
            return None
        d = dc_removal(data)
        d = background_removal(d)
        d = bandpass_filter(d, DT_SEC, 500.0, 4000.0)
        d = gain_sec(d, tpow=1.0, alpha=0.0, dt=DT_SEC)
        mn, mx = np.percentile(d, [2, 98])
        norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
        gray = (norm * 255).astype(np.uint8)
        bgr  = cv2.cvtColor(cv2.resize(gray, (640, 640)), cv2.COLOR_GRAY2BGR)
        return bgr
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# ZON 수집
# ─────────────────────────────────────────────────────────────────────

def collect_zon_dirs(cls: str) -> list[Path]:
    cls_dir = GZ_DATA / cls
    if not cls_dir.exists():
        return []
    return sorted(
        d for d in cls_dir.rglob("*.ZON")
        if d.is_dir() and "ASCII" not in str(d)
    )


def find_dt_in_zon(zon_dir: Path) -> Path | None:
    dts = sorted(zon_dir.glob("*.dt"))
    return dts[0] if dts else None


def load_manifest_zon_dirs() -> set[Path]:
    try:
        raw  = MANIFEST.read_bytes()
        data = json.loads(raw.decode("cp949"))
    except Exception:
        return set()
    return {
        Path(e["source"]).parent
        for e in data.get("images", [])
        if e.get("source")
    }


def build_zon_pools(rng: random.Random):
    """
    Phase G와 동일한 방식으로 train_pool / test_pool 구성.
    단, 분류 모델 학습에는 manifest ZON도 사용.
    """
    trained     = load_manifest_zon_dirs()
    train_pools = {}
    test_pools  = {}

    for cls in CATEGORIES:
        all_zons  = collect_zon_dirs(cls)
        available = [z for z in all_zons if z not in trained]
        shuffled  = available.copy()
        rng.shuffle(shuffled)

        n_test = min(N_TEST_PER_CLS, max(1, len(shuffled) // 2))
        test_pools[cls]  = shuffled[:n_test]
        train_pools[cls] = shuffled[n_test:]

        print(f"  {cls:6s}: 전체={len(all_zons):3d}, "
              f"manifest제외={len(available):3d}, "
              f"test={len(test_pools[cls]):3d}, "
              f"train풀={len(train_pools[cls]):3d}")

    return trained, train_pools, test_pools


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class GPRDataset(Dataset):
    """
    items: [(bgr_np, cls_idx), ...]
    transform: torchvision transform
    """
    def __init__(self, items: list, transform=None):
        self.items     = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        bgr, cls_idx = self.items[idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.transform:
            from PIL import Image
            pil = Image.fromarray(rgb)
            tensor = self.transform(pil)
        else:
            tensor = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return tensor, cls_idx


def make_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.2),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────────────────

def build_model(n_classes: int = 3) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # 마지막 분류기만 교체
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────────────────

def train_model(train_items, val_items, device) -> nn.Module:
    train_ds = GPRDataset(train_items, make_transforms(train=True))
    val_ds   = GPRDataset(val_items,   make_transforms(train=False))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=0, pin_memory=False)

    model = build_model(n_classes=len(CATEGORIES)).to(device)

    # backbone 파라미터는 lr 낮게, 분류기는 높게
    backbone_params = [p for n, p in model.named_parameters()
                       if "classifier" not in n]
    head_params     = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": head_params,     "lr": LR},
    ], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        scheduler.step()
        avg_loss = total_loss / len(train_ds)

        # Val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total if total > 0 else 0.0

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={avg_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(MODEL_PATH))

    print(f"    최적 val_acc={best_val_acc:.3f}  → {MODEL_PATH.name}")

    # 최적 가중치 복원
    model.load_state_dict(torch.load(str(MODEL_PATH),
                                     map_location=device,
                                     weights_only=True))
    return model, history


# ─────────────────────────────────────────────────────────────────────
# 평가 (Recall per class)
# ─────────────────────────────────────────────────────────────────────

def evaluate_recall(model, test_pools: dict, device) -> dict:
    model.eval()
    tf = make_transforms(train=False)
    from PIL import Image

    recall = {}
    for cls, zon_list in test_pools.items():
        cls_idx = CLS2IDX[cls]
        tp = fn = 0
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                continue
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil  = Image.fromarray(rgb)
            inp  = tf(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = model(inp).argmax(dim=1).item()
            if pred_idx == cls_idx:
                tp += 1
            else:
                fn += 1
        total = tp + fn
        recall[cls] = round(tp / total, 4) if total > 0 else 0.0
        pred_name = CATEGORIES[pred_idx] if total > 0 else "?"
        print(f"    {cls:6s}: TP={tp} FN={fn} → Recall={recall[cls]:.3f}")
    return recall


# ─────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────

def plot_results(recall: dict, history: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")

    # (1) Recall 막대그래프
    ax = axes[0]
    ax.set_facecolor("#2a2a4a")
    classes = list(recall.keys())
    vals    = [recall[c] for c in classes]
    colors  = [CLS_COLORS[c] for c in classes]
    bars = ax.bar(classes, vals, color=colors, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", color="white",
                fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Recall", color="white", fontsize=12)
    ax.set_title("Phase K: Classification Recall\n(per class, test ZON)",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", alpha=0.25, color="white", linestyle="--")
    ax.set_xticklabels(classes, color="white", fontsize=12)

    # (2) 학습 곡선
    ax2 = axes[1]
    ax2.set_facecolor("#2a2a4a")
    epochs = list(range(1, len(history["train_loss"]) + 1))
    ax2.plot(epochs, history["train_loss"], color="#e74c3c",
             linewidth=2, label="Train Loss")
    ax2r = ax2.twinx()
    ax2r.plot(epochs, history["val_acc"], color="#2ecc71",
              linewidth=2, label="Val Acc")
    ax2r.set_ylim(0, 1.1)
    ax2r.tick_params(colors="white")
    ax2r.set_ylabel("Val Accuracy", color="#2ecc71", fontsize=11)
    ax2.set_xlabel("Epoch", color="white", fontsize=11)
    ax2.set_ylabel("Loss", color="#e74c3c", fontsize=11)
    ax2.set_title("Training Curve", color="white",
                  fontsize=11, fontweight="bold")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_color("#555")
    ax2.grid(alpha=0.2, color="white", linestyle="--")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#2a2a4a", labelcolor="white", fontsize=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase_k_results.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  결과 저장: {out_path.relative_to(BASE_DIR)}")


def plot_confusion(model, test_pools: dict, device):
    """클래스별 예측 혼동 행렬 (어디로 잘못 분류되는지)."""
    from PIL import Image
    tf  = make_transforms(train=False)
    model.eval()
    confusion = np.zeros((3, 3), dtype=int)   # [true][pred]

    for true_cls, zon_list in test_pools.items():
        true_idx = CLS2IDX[true_cls]
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            inp = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_idx = model(inp).argmax(dim=1).item()
            confusion[true_idx][pred_idx] += 1

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#1a1a2e")
    ax.set_facecolor("#2a2a4a")
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CATEGORIES, color="white")
    ax.set_yticklabels(CATEGORIES, color="white")
    ax.set_xlabel("Predicted", color="white", fontsize=11)
    ax.set_ylabel("True", color="white", fontsize=11)
    ax.set_title("Confusion Matrix", color="white",
                 fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(confusion[i][j]),
                    ha="center", va="center",
                    color="white" if confusion[i][j] < confusion.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "phase_k_confusion.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  혼동행렬: {out_path.relative_to(BASE_DIR)}")


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print("  Phase K: GPR 이미지 분류 (EfficientNet-B0)")
    print(f"{'='*65}")
    print(f"  device: {device}")
    print(f"  epochs: {EPOCHS}, lr: {LR}, batch: {BATCH}")

    rng = random.Random(SEED)

    # ── ZON 풀 구성 ────────────────────────────────────────────────
    print("\n[Step 1] ZON 풀 구성...")
    trained_set, train_pools, test_pools = build_zon_pools(rng)

    # ── 학습 데이터 수집 ───────────────────────────────────────────
    # manifest ZON + train_pool ZON 모두 사용
    print("\n[Step 2] 학습 이미지 로드...")
    train_items = []   # [(bgr, cls_idx), ...]
    val_items   = []

    all_train_zons: dict[str, list[Path]] = {}

    # manifest ZON (Phase E-2 학습 데이터)
    for cls in CATEGORIES:
        cls_zons = [z for z in collect_zon_dirs(cls) if z in trained_set]
        all_train_zons.setdefault(cls, []).extend(cls_zons)

    # train_pool에서 추가 (최대 30개)
    for cls, pool in train_pools.items():
        extra = pool[:30]
        all_train_zons.setdefault(cls, []).extend(extra)

    for cls, zon_list in all_train_zons.items():
        cls_idx = CLS2IDX[cls]
        items_for_cls = []
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                continue
            items_for_cls.append((bgr, cls_idx))

        # 80/20 분할
        rng.shuffle(items_for_cls)
        n_val = max(1, len(items_for_cls) // 5)
        val_items.extend(items_for_cls[:n_val])
        train_items.extend(items_for_cls[n_val:])
        print(f"  {cls:6s}: train={len(items_for_cls) - n_val}, val={n_val}")

    print(f"\n  전체: train={len(train_items)}, val={len(val_items)}")

    if len(train_items) < 5:
        print("  [오류] 학습 데이터 부족")
        return

    # ── 학습 ───────────────────────────────────────────────────────
    print("\n[Step 3] 모델 학습...")
    model, history = train_model(train_items, val_items, device)

    # ── 평가 ───────────────────────────────────────────────────────
    print("\n[Step 4] Test ZON Recall 평가...")
    recall = evaluate_recall(model, test_pools, device)

    # ── 혼동 행렬 ──────────────────────────────────────────────────
    print("\n[Step 5] 혼동 행렬 생성...")
    plot_confusion(model, test_pools, device)

    # ── 시각화 ─────────────────────────────────────────────────────
    print("\n[Step 6] 결과 시각화...")
    plot_results(recall, history)

    # ── 요약 ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  [Phase K 결과 요약]")
    print(f"{'='*65}")
    for cls in CATEGORIES:
        r = recall.get(cls, 0.0)
        bar = "#" * int(r * 20)
        note = " ← YOLO 검출 대비 개선!" if r > 0.1 else ""
        print(f"  {cls:6s}: {r:.3f}  [{bar:<20}]{note}")
    print(f"\n  출력: {OUTPUT_DIR.relative_to(BASE_DIR)}/")

    # Phase G YOLO 검출 결과와 비교
    print("\n  [비교] YOLO 검출 (Phase G, N=20) vs 분류 (Phase K)")
    phase_g_ref = {"pipe": 0.0, "rebar": 0.0, "tunnel": 1.0}
    for cls in CATEGORIES:
        g = phase_g_ref.get(cls, "?")
        k = recall.get(cls, 0.0)
        diff = f"+{k-g:.3f}" if isinstance(g, float) and k > g else \
               (f"{k-g:.3f}" if isinstance(g, float) else "")
        print(f"  {cls:6s}: YOLO={g}  →  분류={k:.3f}  ({diff})")


if __name__ == "__main__":
    main()
