"""
Phase E-1: 자동 라벨링 스크립트

guangzhou_labeled/images/ 의 PNG 파일에서
신호 에너지(행별 표준편차) 기반으로 bbox를 자동 생성.

- pipe_*  → class 1 (pipe)
- rebar_* → class 2 (rebar)

결과:
  - guangzhou_labeled/labels/*.txt  (YOLO 형식)
  - guangzhou_labeled/auto_label_review.png  (검토용 시각화)
"""

import sys
import json
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

LABELED_DIR = Path(__file__).parent.parent / "guangzhou_labeled"
IMAGES_DIR  = LABELED_DIR / "images"
LABELS_DIR  = LABELED_DIR / "labels"

CLASS_ID = {"pipe": 1, "rebar": 2}


# ──────────────────────────────────────────────
# 핵심: 신호 에너지 기반 bbox 감지
# ──────────────────────────────────────────────

def detect_signal_bbox(img_gray: np.ndarray, cls_name: str):
    """
    GPR B-scan 이미지에서 신호 활성 영역을 감지하여 bbox 반환.

    전략:
      1. 행별 표준편차 계산 → 신호가 강한 행 탐지
      2. 최상단 직접파(direct wave) 행 스킵
      3. 열별 표준편차로 좌우 범위 결정
      4. 여백(padding) 추가

    Returns:
      (x_center, y_center, width, height) in [0,1] normalized,
      or None if no signal found
    """
    H, W = img_gray.shape
    img_f = img_gray.astype(np.float32)

    # ─ 1. 행별 에너지(std) ─
    row_std = np.std(img_f, axis=1)   # shape (H,)

    # ─ 2. 직접파 스킵: 상단 3% ─
    skip_top = max(1, int(H * 0.03))

    # ─ 3. 활성 행 탐지 ─
    # 하단 50%는 주로 무신호 구역 → 상단 절반만 고려
    upper_half = row_std[skip_top: H // 2]
    if len(upper_half) == 0:
        upper_half = row_std[skip_top:]

    threshold_row = np.percentile(row_std[skip_top:], 55)
    active_mask = row_std[skip_top:] > threshold_row
    active_indices = np.where(active_mask)[0] + skip_top

    if len(active_indices) < 5:
        # fallback: 상단 40% 전체
        y1 = skip_top
        y2 = int(H * 0.40)
    else:
        y1 = int(active_indices[0])
        y2 = int(active_indices[-1])

        # rebar: 아치가 반복되므로 y 범위가 넓게 잡힘 → 적절
        # pipe:  밝은 선 주변이 잡힘 → 여백 추가로 보정

    # ─ 4. 열별 에너지로 좌우 범위 ─
    region = img_f[y1:y2 + 1]
    if region.shape[0] < 2:
        x1, x2 = 0, W - 1
    else:
        col_std = np.std(region, axis=0)
        threshold_col = np.percentile(col_std, 15)
        active_cols = np.where(col_std > threshold_col)[0]
        if len(active_cols) < 5:
            x1, x2 = 0, W - 1
        else:
            x1 = int(active_cols[0])
            x2 = int(active_cols[-1])

    # ─ 5. 여백 추가 ─
    pad_y = int(H * 0.03)
    pad_x = int(W * 0.02)
    y1 = max(0,     y1 - pad_y)
    y2 = min(H - 1, y2 + pad_y)
    x1 = max(0,     x1 - pad_x)
    x2 = min(W - 1, x2 + pad_x)

    # ─ 6. YOLO 정규화 ─
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H

    return cx, cy, bw, bh, (x1, y1, x2, y2)


# ──────────────────────────────────────────────
# 검토용 시각화
# ──────────────────────────────────────────────

def make_review_image(results: list):
    n   = len(results)
    cols = 5
    rows = (n + cols - 1) // cols

    CLASS_COLORS = {1: '#3498db', 2: '#2ecc71'}

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 3.2),
                             facecolor='#1a1a2e')
    fig.suptitle('Phase E-1 자동 라벨링 검토 (auto_label_review.png)',
                 color='white', fontsize=13, fontweight='bold')

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]

        raw = res["img_path"].read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if res["bbox_px"] is not None:
            x1, y1, x2, y2 = res["bbox_px"]
            cls_id = res["class_id"]
            color  = CLASS_COLORS.get(cls_id, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 5),
                    res["cls_name"],
                    color='white', fontsize=7,
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))

        ax.set_title(res["img_name"], color='white', fontsize=7)
        ax.axis('off')

    # 빈 칸
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout()
    out = LABELED_DIR / "auto_label_review.png"
    plt.savefig(str(out), dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return out


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("Phase E-1: 자동 라벨링")
    print(f"  이미지: {IMAGES_DIR}")
    print(f"  라벨:   {LABELS_DIR}")

    manifest_path = LABELED_DIR / "manifest.json"
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)["images"]

    results = []
    ok_count = 0

    for entry in manifest:
        img_path = IMAGES_DIR / entry["image"]
        lbl_path = LABELS_DIR / entry["label"]
        cls_name = entry["class"]
        cls_id   = CLASS_ID[cls_name]

        raw = img_path.read_bytes()
        img_gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

        cx, cy, bw, bh, bbox_px = detect_signal_bbox(img_gray, cls_name)

        # YOLO 형식으로 저장
        lbl_path.write_text(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        results.append({
            "img_name": entry["image"],
            "img_path": img_path,
            "cls_name": cls_name,
            "class_id": cls_id,
            "bbox_px":  bbox_px,
            "yolo":     (cx, cy, bw, bh),
        })
        print(f"  ✓ {entry['image']:20s}  [{cls_name:5s}]  "
              f"cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}")
        ok_count += 1

    print(f"\n  라벨 저장: {ok_count}개")

    # 검토 이미지
    print("\n검토 이미지 생성...")
    review_path = make_review_image(results)
    print(f"  → {review_path}")

    print(f"\n{'='*55}")
    print("자동 라벨링 완료")
    print(f"{'='*55}")
    print(f"  라벨 파일: {LABELS_DIR}")
    print(f"  검토 이미지: {review_path}")
    print(f"\n  ※ auto_label_review.png 를 열어 bbox가 올바른지 확인하세요.")
    print(f"     크게 벗어난 이미지만 수동으로 수정하면 됩니다.")
    print(f"\n  다음 단계:")
    print(f"    python phase_e1_finetune.py")


if __name__ == "__main__":
    main()
