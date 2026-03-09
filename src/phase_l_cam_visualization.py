"""
Phase L: GradCAM 시각화 — 모델이 어디를 보고 분류했는가

배경
----
Phase K에서 EfficientNet-B0으로 pipe=0.667 rebar=0.900 달성.
하지만 "모델이 GPR 이미지의 어느 부분을 보고 결정했는가?"는 불명확.

GradCAM (Gradient-weighted Class Activation Map):
  - 예측 클래스 점수를 마지막 Conv 특징맵에 대해 역전파
  - 기여도 높은 영역 → 히트맵 (빨강=중요, 파랑=덜 중요)
  - 원본 이미지에 반투명 오버레이

출력
----
  phase_l/gradcam_grid.png      : 클래스별 4개 샘플 × GradCAM 그리드
  phase_l/gradcam_{cls}_{i}.png : 개별 이미지

사용법
------
  /c/Python314/python.exe src/phase_l_cam_visualization.py
"""

import sys
import json
import random
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec,
)

# ── 경로 ──────────────────────────────────────────────────────────────
GZ_DATA    = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST   = BASE_DIR / "guangzhou_labeled/manifest.json"
MODEL_PATH = BASE_DIR / "models/phase_k_cls.pt"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_l"

# ── 상수 ──────────────────────────────────────────────────────────────
CATEGORIES = ["pipe", "rebar", "tunnel"]
CLS2IDX    = {c: i for i, c in enumerate(CATEGORIES)}
IMG_SIZE   = 224
N_SAMPLES  = 4        # 클래스당 시각화할 샘플 수
SEED       = 42
DT_SEC     = (8.0 / 512) * 1e-9

CLS_COLORS = {
    "pipe":   (52,  152, 219),   # 파랑
    "rebar":  (46,  204, 113),   # 초록
    "tunnel": (243, 156,  18),   # 주황
}


# ─────────────────────────────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
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


def build_test_pool(rng: random.Random) -> dict[str, list[Path]]:
    """Phase K와 동일한 held-out test 풀 구성."""
    trained = load_manifest_zon_dirs()
    test_pools = {}
    for cls in CATEGORIES:
        all_zons  = collect_zon_dirs(cls)
        available = [z for z in all_zons if z not in trained]
        shuffled  = available.copy()
        rng.shuffle(shuffled)
        n_test = min(10, max(1, len(shuffled) // 2))
        test_pools[cls] = shuffled[:n_test]
    return test_pools


# ─────────────────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────────────────

def load_model(device) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CATEGORIES)),
    )
    model.load_state_dict(torch.load(str(MODEL_PATH),
                                     map_location=device,
                                     weights_only=True))
    model.eval()
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────
# GradCAM 적용
# ─────────────────────────────────────────────────────────────────────

def make_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


def gradcam_on_image(model, cam_extractor, bgr: np.ndarray, device):
    """
    반환:
      pred_cls  : 예측 클래스 이름
      conf      : 예측 신뢰도 (softmax)
      cam_img   : GradCAM 히트맵이 오버레이된 PIL Image (224×224 RGB)
      orig_img  : 원본 PIL Image (224×224 RGB)
    """
    tf  = make_transform()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)))

    inp    = tf(Image.fromarray(rgb)).unsqueeze(0).to(device)
    out    = model(inp)                         # (1, 3)
    probs  = torch.softmax(out, dim=1)[0]
    pred   = out.argmax(dim=1).item()
    conf   = probs[pred].item()

    # GradCAM 추출
    activation_map = cam_extractor(pred, out)   # list[Tensor]
    cam_tensor = activation_map[0]              # (1, H, W)

    # torchcam overlay_mask: PIL Image + mask tensor → PIL Image
    result_pil = overlay_mask(
        pil,
        Image.fromarray(cam_tensor.squeeze().cpu().numpy()),
        alpha=0.5,
        colormap="jet",
    )

    return CATEGORIES[pred], conf, result_pil, pil


# ─────────────────────────────────────────────────────────────────────
# 시각화 그리드
# ─────────────────────────────────────────────────────────────────────

def plot_cam_grid(samples: dict):
    """
    samples = {
      cls: [(true_cls, pred_cls, conf, cam_pil, orig_pil), ...]
    }
    3행(클래스) × N_SAMPLES×2열(원본+CAM) 그리드
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_cols  = N_SAMPLES * 2   # 원본 + CAM 교대
    n_rows  = len(CATEGORIES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.2, n_rows * 2.5),
        facecolor="#1a1a2e",
    )
    fig.suptitle(
        "Phase L: GradCAM — 모델이 어디를 보고 분류했는가\n"
        "(짝수열=원본, 홀수열=GradCAM 히트맵  |  빨강=집중, 파랑=무시)",
        color="white", fontsize=11, fontweight="bold", y=1.01,
    )

    for row_i, cls in enumerate(CATEGORIES):
        cls_samples = samples.get(cls, [])
        color_rgb   = tuple(c / 255 for c in CLS_COLORS[cls])

        for col_pair in range(N_SAMPLES):
            ax_orig = axes[row_i][col_pair * 2]
            ax_cam  = axes[row_i][col_pair * 2 + 1]

            for ax in (ax_orig, ax_cam):
                ax.set_facecolor("#111")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#333")

            if col_pair >= len(cls_samples):
                ax_orig.text(0.5, 0.5, "N/A", ha="center", va="center",
                             color="#555", transform=ax_orig.transAxes)
                ax_cam.text(0.5, 0.5, "N/A", ha="center", va="center",
                            color="#555", transform=ax_cam.transAxes)
                continue

            true_cls, pred_cls, conf, cam_pil, orig_pil = cls_samples[col_pair]

            # 원본
            ax_orig.imshow(orig_pil)
            if col_pair == 0:
                ax_orig.set_ylabel(f"True: {cls}", color=color_rgb,
                                   fontsize=10, fontweight="bold", labelpad=6)

            # CAM
            ax_cam.imshow(cam_pil)
            match = pred_cls == true_cls
            title_color = "#2ecc71" if match else "#e74c3c"
            ax_cam.set_title(
                f"Pred: {pred_cls}\n{conf:.0%}",
                color=title_color, fontsize=8, pad=3,
            )
            # 테두리 색으로 정오 표시
            for sp in ax_cam.spines.values():
                sp.set_color(title_color)
                sp.set_linewidth(2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "gradcam_grid.png"
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  그리드 저장: {out_path.relative_to(BASE_DIR)}")
    return out_path


def plot_single_cam(true_cls, pred_cls, conf, cam_pil, orig_pil,
                    idx: int, correct: bool):
    """클래스별 개별 이미지 저장."""
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.2), facecolor="#1a1a2e")
    for ax, img, title in zip(axes,
                               [orig_pil, cam_pil],
                               ["원본 GPR", f"GradCAM (pred={pred_cls} {conf:.0%})"]):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#111")
        for sp in ax.spines.values():
            sp.set_color("#2ecc71" if correct else "#e74c3c")
            sp.set_linewidth(1.5)

    status = "O" if correct else "X"
    fig.suptitle(f"[{status}] True={true_cls}  Pred={pred_cls}  conf={conf:.1%}",
                 color="#2ecc71" if correct else "#e74c3c",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"gradcam_{true_cls}_{idx:02d}.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print("  Phase L: GradCAM 시각화")
    print(f"{'='*60}")
    print(f"  device: {device}")

    if not MODEL_PATH.exists():
        print(f"  [오류] Phase K 모델 없음: {MODEL_PATH}")
        print("  phase_k_classification.py를 먼저 실행하세요.")
        return

    # 모델 로드
    print("\n[Step 1] Phase K 모델 로드...")
    model = load_model(device)

    # GradCAM: EfficientNet-B0의 마지막 Conv 블록 타겟
    # features[-1] = MBConv 마지막 블록 → Conv2d 출력 대상
    target_layer = model.features[-1]
    cam_extractor = GradCAM(model, target_layer=target_layer)

    # test pool 구성 (Phase K와 동일)
    print("[Step 2] Test ZON 풀 구성...")
    rng = random.Random(SEED)
    test_pools = build_test_pool(rng)
    for cls, pool in test_pools.items():
        print(f"  {cls}: {len(pool)}개 ZON")

    # 각 클래스별 샘플 수집 및 CAM 생성
    print("\n[Step 3] GradCAM 생성...")
    samples = {cls: [] for cls in CATEGORIES}
    summary = {cls: {"tp": 0, "fn": 0} for cls in CATEGORIES}

    for cls in CATEGORIES:
        print(f"\n  [{cls.upper()}]")
        collected = 0
        for zon_dir in test_pools[cls]:
            if collected >= N_SAMPLES:
                break
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                continue

            try:
                pred_cls, conf, cam_pil, orig_pil = gradcam_on_image(
                    model, cam_extractor, bgr, device
                )
            except Exception as e:
                print(f"    [오류] {zon_dir.name}: {e}")
                continue

            correct = (pred_cls == cls)
            if correct:
                summary[cls]["tp"] += 1
            else:
                summary[cls]["fn"] += 1

            samples[cls].append((cls, pred_cls, conf, cam_pil, orig_pil))

            # 개별 이미지 저장
            plot_single_cam(cls, pred_cls, conf, cam_pil, orig_pil,
                            collected, correct)

            status = "O" if correct else "X"
            print(f"    [{status}] pred={pred_cls} conf={conf:.1%}  "
                  f"({zon_dir.parent.name}/{zon_dir.name})")
            collected += 1

    # 그리드 저장
    print("\n[Step 4] 그리드 이미지 생성...")
    plot_cam_grid(samples)

    # 요약
    print(f"\n{'='*60}")
    print("  [Phase L 요약]")
    print(f"{'='*60}")
    for cls in CATEGORIES:
        tp = summary[cls]["tp"]
        fn = summary[cls]["fn"]
        total = tp + fn
        recall = tp / total if total > 0 else 0.0
        print(f"  {cls:6s}: {tp}/{total}  Recall={recall:.3f}")
    print(f"\n  출력: {OUTPUT_DIR.relative_to(BASE_DIR)}/")
    print("  → gradcam_grid.png : 전체 그리드")
    print("  → gradcam_*.png    : 클래스별 개별 이미지")


if __name__ == "__main__":
    main()
