"""
Phase H: pipe/rebar bbox 탐지 v3 - 클래스별 깊이 구간 특화

Phase G 문제: detect_signal_bbox(v1) 사용 -> 모든 클래스 bw=0.998 (전폭)
               -> pipe/rebar Recall=0 (클래스 특징 학습 불가)

Phase H 개선: detect_bbox_v3 - 클래스별 깊이 구간 분리
  pipe  : 5-65% 깊이, 에너지 피크 행 중심, 높이 30% 제한
  rebar : 5-45% 얕은 구간, 활성 행 포착, 높이 40% 제한
  tunnel: CC 기반 (v2 계승, 이미 Recall=1.0)

추가: 라벨 미리보기 이미지 저장 (실험 전 시각적 검증)
     Phase G vs Phase H 비교 곡선 PNG

사용법
------
  /c/Python314/python.exe src/phase_h_bbox_v3.py
"""

import sys
import json
import shutil
import random
import time
import csv
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec,
)

# -- 경로 ------------------------------------------------------------------
GZ_DATA    = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST   = BASE_DIR / "guangzhou_labeled/manifest.json"
E2_WEIGHTS = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"
FDTD_DIR   = BASE_DIR / "data/gpr/yolo_fdtd"
SYNTH_DIR  = BASE_DIR / "data/gpr/yolo_multiclass"
WORK_DIR   = BASE_DIR / "data/gpr/yolo_gz_phase_h"
MODEL_OUT  = BASE_DIR / "models/yolo_runs/phase_h"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_h"
PHASE_G_CSV = BASE_DIR / "src/output/week4_multiclass/phase_g/recall_results.csv"

# -- 실험 상수 -------------------------------------------------------------
CATEGORIES   = ["pipe", "rebar", "tunnel"]
CLASS_IDS    = {"pipe": 1, "rebar": 2, "tunnel": 3}
CLASS_NAMES  = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS = {"pipe": "#3498db", "rebar": "#2ecc71", "tunnel": "#f39c12"}

N_VALUES       = [1, 3, 5, 10, 20]
N_TEST_PER_CLS = 10
N_SYNTH_TRAIN  = 120
SEED           = 42

DT_SEC    = (8.0 / 512) * 1e-9
CONF_EVAL = 0.05
IMGSZ     = 640


# =========================================================================
# 전처리
# =========================================================================

def preprocess_dt(dt_path: Path):
    """IDS .dt -> 640x640 BGR. 실패 시 None."""
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


# =========================================================================
# bbox 탐지 v3 - 클래스별 깊이 구간 특화
# =========================================================================

def _to_yolo(x1, y1, x2, y2, H, W, pad_y_ratio=0.02, pad_x_ratio=0.01):
    """픽셀 bbox -> YOLO 정규화 (cx, cy, bw, bh, (x1,y1,x2,y2))."""
    pad_y = int(H * pad_y_ratio)
    pad_x = int(W * pad_x_ratio)
    x1 = max(0,     x1 - pad_x)
    y1 = max(0,     y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh, (x1, y1, x2, y2)


def _detect_pipe_v3(img_f, H, W, skip_top):
    """
    pipe: 중간 깊이(5-65%) 구간에서 수평 강반사 띠 탐지.
    - 행별 에너지(RMS) 계산
    - 에너지 > 70th percentile 활성 행 추출
    - 높이 30% 초과 시 피크 중심으로 잘라냄
    - x 범위: 전폭 (pipe는 스캔 전체에 걸쳐 나타남)
    """
    y_end  = int(H * 0.65)
    region = img_f[skip_top:y_end, :]

    if region.shape[0] < 3:
        y1, y2 = skip_top, int(H * 0.6)
    else:
        row_energy = np.mean(region ** 2, axis=1)
        peak_rel   = int(np.argmax(row_energy))
        peak_abs   = peak_rel + skip_top

        threshold = np.percentile(row_energy, 70)
        active    = np.where(row_energy > threshold)[0]

        if len(active) >= 3:
            y1 = int(active[0])  + skip_top
            y2 = int(active[-1]) + skip_top
        else:
            half = int(H * 0.08)
            y1   = max(skip_top, peak_abs - half)
            y2   = min(y_end - 1, peak_abs + half)

        # 높이 30% 제한
        max_h = int(H * 0.30)
        if y2 - y1 > max_h:
            center = (y1 + y2) // 2
            y1 = max(skip_top, center - max_h // 2)
            y2 = min(H - 1,    y1 + max_h)

    x1 = int(W * 0.02)
    x2 = int(W * 0.98)
    return _to_yolo(x1, y1, x2, y2, H, W)


def _detect_rebar_v3(img_f, H, W, skip_top):
    """
    rebar: 얕은 구간(5-45%) 집중.
    - rebar는 항상 얕은 깊이에 촘촘한 아치 패턴
    - 행 에너지 > 60th percentile 활성 행 전부 포착
    - 높이 40% 제한
    - x 범위: 전폭
    """
    y_end  = int(H * 0.45)
    region = img_f[skip_top:y_end, :]

    if region.shape[0] < 3:
        y1, y2 = skip_top, int(H * 0.40)
    else:
        row_energy = np.mean(region ** 2, axis=1)
        threshold  = np.percentile(row_energy, 60)
        active     = np.where(row_energy > threshold)[0]

        if len(active) >= 3:
            y1 = int(active[0])  + skip_top
            y2 = int(active[-1]) + skip_top
        else:
            y1 = skip_top
            y2 = y_end - 1

        # 높이 40% 제한
        max_h = int(H * 0.40)
        if y2 - y1 > max_h:
            y2 = min(H - 1, y1 + max_h)

    x1 = int(W * 0.02)
    x2 = int(W * 0.98)
    return _to_yolo(x1, y1, x2, y2, H, W)


def _detect_tunnel_v3(img_gray, img_f, H, W, skip_top):
    """
    tunnel: CC 기반 (Phase E-2 v2 계승).
    이미 Recall=1.0 달성 - 변경 없음.
    """
    max_y      = int((H - skip_top) * 0.35)
    work_region = img_f[skip_top:skip_top + max(5, max_y), :]

    thresh_val = np.percentile(work_region, 72)
    binary     = (work_region > thresh_val).astype(np.uint8) * 255

    kx = max(3, W // 30)
    ky = max(3, H // 60)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)

    min_w = int(W * 0.06)
    valid = []
    if num_labels > 1:
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w >= min_w and h >= 3:
                valid.append({
                    'x': x, 'y': y + skip_top,
                    'x2': x + w, 'y2': y + h + skip_top,
                    'area': area,
                })

    if not valid:
        y1, y2 = skip_top, int(H * 0.50)
        x1, x2 = int(W * 0.05), int(W * 0.95)
        return _to_yolo(x1, y1, x2, y2, H, W)

    top_n = sorted(valid, key=lambda c: c['area'], reverse=True)[:4]
    x1 = min(c['x']  for c in top_n)
    y1 = min(c['y']  for c in top_n)
    x2 = max(c['x2'] for c in top_n)
    y2 = max(c['y2'] for c in top_n)
    return _to_yolo(x1, y1, x2, y2, H, W)


def detect_bbox_v3(img_gray: np.ndarray, cls_name: str):
    """클래스별 특화 bbox 탐지 v3."""
    H, W     = img_gray.shape
    img_f    = img_gray.astype(np.float32)
    skip_top = max(1, int(H * 0.05))

    if cls_name == "pipe":
        return _detect_pipe_v3(img_f, H, W, skip_top)
    elif cls_name == "rebar":
        return _detect_rebar_v3(img_f, H, W, skip_top)
    else:
        return _detect_tunnel_v3(img_gray, img_f, H, W, skip_top)


# =========================================================================
# 라벨 미리보기 (실험 전 시각적 검증)
# =========================================================================

def save_label_preview(train_pools: dict, n_sample: int = 3):
    """
    클래스별 샘플 이미지에 v3 bbox 그려서 저장.
    실험 전에 라벨 품질 확인용.
    """
    preview_dir = OUTPUT_DIR / "label_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for cls in CATEGORIES:
        pool = train_pools[cls]
        sample = pool[:min(n_sample, len(pool))]

        for i, zon_dir in enumerate(sample):
            dts = sorted(zon_dir.glob("*.dt"))
            if not dts:
                continue
            bgr = preprocess_dt(dts[0])
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cx, cy, bw, bh, (x1, y1, x2, y2) = detect_bbox_v3(gray, cls)

            # bbox 그리기
            vis = bgr.copy()
            color_map = {"pipe": (255, 100, 50), "rebar": (50, 200, 50), "tunnel": (50, 150, 255)}
            color = color_map[cls]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls} cy={cy:.2f} bh={bh:.2f}"
            cv2.putText(vis, label_text, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            out_path = preview_dir / f"{cls}_{i:02d}.png"
            ok, buf = cv2.imencode(".png", vis)
            if ok:
                out_path.write_bytes(buf.tobytes())

    print(f"  라벨 미리보기 저장: {preview_dir.relative_to(BASE_DIR)} ({n_sample}장/클래스)")


# =========================================================================
# ZON 수집 & 풀 구성
# =========================================================================

def collect_zon_dirs(cls: str) -> list[Path]:
    cls_dir = GZ_DATA / cls
    if not cls_dir.exists():
        return []
    return sorted(
        d for d in cls_dir.rglob("*.ZON")
        if d.is_dir() and "ASCII" not in str(d)
    )


def find_dt_in_zon(zon_dir: Path):
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
    trained = load_manifest_zon_dirs()
    train_pools: dict[str, list[Path]] = {}
    test_pools:  dict[str, list[Path]] = {}

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

    return train_pools, test_pools


# =========================================================================
# 데이터셋 구성
# =========================================================================

def build_dataset(sel_zones: dict[str, list[Path]]) -> Path:
    """선택된 ZON(v3 라벨) + 합성 -> YOLO 데이터셋."""
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = WORK_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_s = random.Random(SEED)
    idx   = 0

    # 실제 ZON 이미지 (v3 bbox 라벨)
    for cls, zon_list in sel_zones.items():
        cls_id = CLASS_IDS[cls]
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                continue

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            cx, cy, bw, bh, _ = detect_bbox_v3(gray, cls)  # v3 사용
            label_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

            stem = f"gz_{cls}_{idx:04d}"
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                continue
            raw_png = buf.tobytes()

            for split in ("train", "val"):
                (WORK_DIR / "images" / split / f"{stem}.png").write_bytes(raw_png)
                (WORK_DIR / "labels" / split / f"{stem}.txt").write_text(label_line)
            idx += 1

    # 합성 데이터
    def copy_synth(src_dir: Path, split: str, n: int, prefix: str) -> int:
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            return 0
        imgs = sorted(img_dir.glob("*.png"))
        rng_s.shuffle(imgs)
        cnt = 0
        for img_p in imgs[:n]:
            lbl_p  = lbl_dir / img_p.with_suffix(".txt").name
            stem2  = f"{prefix}_{img_p.stem}"
            (WORK_DIR / "images" / split / f"{stem2}.png").write_bytes(img_p.read_bytes())
            dst = WORK_DIR / "labels" / split / f"{stem2}.txt"
            dst.write_bytes(lbl_p.read_bytes() if lbl_p.exists() else b"")
            cnt += 1
        return cnt

    for synth_dir, prefix in [(FDTD_DIR, "fdtd"), (SYNTH_DIR, "synth")]:
        if synth_dir.exists():
            copy_synth(synth_dir, "train", N_SYNTH_TRAIN // 2, prefix)
            copy_synth(synth_dir, "val",   N_SYNTH_TRAIN // 6, prefix)

    yaml_path = WORK_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {WORK_DIR.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 4\n"
        "names: ['sinkhole', 'pipe', 'rebar', 'tunnel']\n",
        encoding="utf-8",
    )

    n_tr = len(list((WORK_DIR / "images/train").glob("*.png")))
    n_vl = len(list((WORK_DIR / "images/val").glob("*.png")))
    print(f"    데이터셋: train={n_tr}, val={n_vl}  (ZON 이미지: {idx}개)")
    return yaml_path


# =========================================================================
# Fine-tuning
# =========================================================================

def finetune(yaml_path: Path, n: int):
    from ultralytics import YOLO

    run_name = f"run_n{n:02d}"
    best_pt  = MODEL_OUT / run_name / "weights/best.pt"

    if best_pt.exists():
        print(f"    [스킵] 이미 학습된 가중치: {best_pt.relative_to(BASE_DIR)}")
        return best_pt

    if not E2_WEIGHTS.exists():
        print(f"    [오류] E-2 가중치 없음: {E2_WEIGHTS}")
        return None

    model = YOLO(str(E2_WEIGHTS))
    model.train(
        data=str(yaml_path),
        epochs=30,
        batch=2,
        imgsz=416,
        lr0=5e-5,
        lrf=0.01,
        optimizer="AdamW",
        cos_lr=True,
        freeze=5,
        dropout=0.1,
        patience=15,
        warmup_epochs=2,
        mosaic=0.0,
        plots=False,
        project=str(MODEL_OUT),
        name=run_name,
        exist_ok=True,
        verbose=False,
        workers=0,
        cache=False,
        amp=False,
    )
    return best_pt if best_pt.exists() else None


# =========================================================================
# Recall 평가
# =========================================================================

def evaluate_recall(model, test_pools: dict):
    recall  = {}
    details = {}

    for cls, zon_list in test_pools.items():
        tp = fn = skip = 0
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                skip += 1
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                skip += 1
                continue

            preds = model.predict(bgr, conf=CONF_EVAL, imgsz=IMGSZ, verbose=False)
            detected = []
            if preds and preds[0].boxes is not None and len(preds[0].boxes):
                detected = [
                    CLASS_NAMES[int(c)]
                    for c in preds[0].boxes.cls.cpu().numpy()
                    if int(c) < len(CLASS_NAMES)
                ]

            if cls in detected:
                tp += 1
            else:
                fn += 1

        total = tp + fn
        recall[cls]  = round(tp / total, 4) if total > 0 else 0.0
        details[cls] = {"tp": tp, "fn": fn, "skip": skip, "total": total}

    return recall, details


# =========================================================================
# 시각화 - Phase G vs Phase H 비교 곡선
# =========================================================================

def load_phase_g_results() -> list[dict]:
    """Phase G recall_results.csv 로드."""
    if not PHASE_G_CSV.exists():
        return []
    results = []
    try:
        with open(PHASE_G_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append({
                    "n": int(row["n_zon"]),
                    "recall": {cls: float(row[f"{cls}_recall"]) for cls in CATEGORIES},
                })
    except Exception:
        pass
    return sorted(results, key=lambda r: r["n"])


def plot_comparison(results_h: list[dict]) -> Path:
    """Phase G(점선) vs Phase H(실선) 비교 곡선."""
    results_g = load_phase_g_results()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1a1a2e")
    fig.suptitle(
        "Phase G(v1) vs Phase H(v3): bbox 개선 효과\n"
        f"(E-2 fine-tuning 30ep | conf>={CONF_EVAL} | test={N_TEST_PER_CLS} ZON/cls)",
        color="white", fontsize=12, fontweight="bold",
    )

    xs_h = [r["n"] for r in results_h]

    for ax, cls in zip(axes, CATEGORIES):
        ax.set_facecolor("#2a2a4a")
        color = CLASS_COLORS[cls]

        # Phase H (실선)
        ys_h = [r["recall"][cls] for r in results_h]
        ax.plot(xs_h, ys_h, "o-", color=color, linewidth=2.5,
                markersize=8, label="H (v3)", zorder=3)
        for xi, yi in zip(xs_h, ys_h):
            ax.text(xi, yi + 0.05, f"{yi:.2f}", ha="center",
                    color=color, fontsize=9, fontweight="bold")

        # Phase G (점선, 있을 때만)
        if results_g:
            xs_g = [r["n"] for r in results_g]
            ys_g = [r["recall"][cls] for r in results_g]
            ax.plot(xs_g, ys_g, "s--", color=color, linewidth=1.5,
                    markersize=6, alpha=0.5, label="G (v1)")

        ax.set_title(cls, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("학습 ZON 수", color="white", fontsize=10)
        ax.set_ylabel("Recall", color="white", fontsize=10)
        ax.set_xticks(xs_h)
        ax.set_ylim(-0.05, 1.15)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#555")
        ax.grid(alpha=0.25, color="white", linestyle="--")
        ax.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=9)

    plt.tight_layout()

    fig.canvas.draw()
    w, h  = fig.canvas.get_width_height()
    buf   = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr   = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    plt.close()

    out_path = OUTPUT_DIR.parent / "phase_h_comparison.png"
    if ok:
        out_path.write_bytes(encoded.tobytes())
        print(f"  비교 곡선 저장: {out_path.name}")
    return out_path


# =========================================================================
# 메인
# =========================================================================

def main():
    print("\n" + "=" * 65)
    print("  Phase H: pipe/rebar bbox v3 - 클래스별 깊이 구간 특화")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics")
        return

    if not E2_WEIGHTS.exists():
        print(f"[오류] E-2 가중치 없음: {E2_WEIGHTS}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # 1. ZON 풀 구성
    print("\n[1/4] ZON 풀 구성...")
    rng = random.Random(SEED)
    train_pools, test_pools = build_zon_pools(rng)

    max_n   = min(len(train_pools[cls]) for cls in CATEGORIES)
    valid_n = [n for n in N_VALUES if n <= max_n]
    if len(valid_n) < len(N_VALUES):
        skipped = [n for n in N_VALUES if n > max_n]
        print(f"  [주의] 학습 풀 부족으로 N={skipped} 제외 (최대={max_n})")
    print(f"\n  테스트 풀: 클래스당 {N_TEST_PER_CLS}개 고정")
    print(f"  실험 N 값: {valid_n}")

    # 2. 라벨 미리보기 (실험 전 검증)
    print("\n[2/4] 라벨 미리보기 저장 (v3 bbox 시각 검증)...")
    save_label_preview(train_pools, n_sample=3)

    # 3. 기존 결과 로드
    csv_path   = OUTPUT_DIR / "recall_results.csv"
    csv_header = (
        ["n_zon"]
        + [f"{cls}_recall" for cls in CATEGORIES]
        + [f"{cls}_tp"     for cls in CATEGORIES]
        + [f"{cls}_total"  for cls in CATEGORIES]
    )

    existing_n: set[int] = set()
    results:    list[dict] = []

    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n_val = int(row["n_zon"])
                existing_n.add(n_val)
                results.append({
                    "n": n_val,
                    "recall": {cls: float(row[f"{cls}_recall"]) for cls in CATEGORIES},
                })
        print(f"\n  기존 결과 로드: N = {sorted(existing_n)}")

    # 4. N별 실험 루프
    print(f"\n[3/4] 실험 루프 (총 {len(valid_n)}회)...")
    f_csv  = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if not existing_n:
        writer.writerow(csv_header)

    for n in valid_n:
        if n in existing_n:
            print(f"\n  -- N = {n:2d} : 기존 결과 재사용 --")
            continue

        print(f"\n{'─' * 55}")
        print(f"  실험: N = {n} ZON/클래스")
        print(f"{'─' * 55}")
        t0 = time.time()

        rng_sel   = random.Random(SEED + n)
        sel_zones = {}
        for cls in CATEGORIES:
            pool = train_pools[cls]
            k    = min(n, len(pool))
            sel_zones[cls] = rng_sel.sample(pool, k)
            print(f"  {cls:6s}: {k}개 ZON 선택 (pool={len(pool)})")

        print("\n  데이터셋 구성 (v3 bbox)...")
        yaml_path = build_dataset(sel_zones)

        print(f"\n  Fine-tuning (30 에폭, E-2 가중치)...")
        best_pt = finetune(yaml_path, n)
        if best_pt is None:
            print("  [스킵] Fine-tuning 실패")
            continue

        print("\n  Recall 평가...")
        eval_model = YOLO(str(best_pt))
        recall, det = evaluate_recall(eval_model, test_pools)

        elapsed = time.time() - t0
        print(f"\n  -- N={n} 결과  ({elapsed:.0f}s) --")
        for cls in CATEGORIES:
            d = det[cls]
            print(f"  {cls:6s}: Recall={recall[cls]:.3f}  "
                  f"TP={d['tp']}, FN={d['fn']}, skip={d['skip']}")

        row_data = (
            [n]
            + [recall[cls]       for cls in CATEGORIES]
            + [det[cls]["tp"]    for cls in CATEGORIES]
            + [det[cls]["total"] for cls in CATEGORIES]
        )
        writer.writerow(row_data)
        f_csv.flush()

        results.append({"n": n, "recall": recall})
        existing_n.add(n)

    f_csv.close()

    # 5. 비교 곡선
    print(f"\n[4/4] Phase G vs H 비교 곡선 생성...")
    results.sort(key=lambda r: r["n"])
    if len(results) >= 2:
        plot_comparison(results)
    else:
        print("  결과 2개 미만 -> 곡선 생략")

    # 최종 요약
    print(f"\n{'=' * 65}")
    print("Phase H 완료 - bbox v3 실험 결과")
    print(f"{'=' * 65}")
    print(f"\n  {'N':>4}  {'pipe':>8}  {'rebar':>8}  {'tunnel':>8}")
    print("  " + "-" * 38)
    for r in results:
        print(f"  {r['n']:>4}  "
              f"{r['recall']['pipe']:>8.3f}  "
              f"{r['recall']['rebar']:>8.3f}  "
              f"{r['recall']['tunnel']:>8.3f}")

    # Phase G 비교
    results_g = load_phase_g_results()
    if results_g:
        print(f"\n  [Phase G vs H 비교]")
        print(f"  {'N':>4}  {'pipe_G':>8} {'pipe_H':>8}  {'rebar_G':>8} {'rebar_H':>8}  {'tunnel_G':>9} {'tunnel_H':>9}")
        print("  " + "-" * 65)
        g_map = {r["n"]: r["recall"] for r in results_g}
        h_map = {r["n"]: r["recall"] for r in results}
        for n in sorted(set(g_map) & set(h_map)):
            g = g_map[n]
            h = h_map[n]
            print(f"  {n:>4}  "
                  f"{g['pipe']:>8.3f} {h['pipe']:>8.3f}  "
                  f"{g['rebar']:>8.3f} {h['rebar']:>8.3f}  "
                  f"{g['tunnel']:>9.3f} {h['tunnel']:>9.3f}")

    print(f"\n  CSV      : {csv_path.relative_to(BASE_DIR)}")
    print(f"  미리보기 : src/output/week4_multiclass/phase_h/label_preview/")
    print(f"  비교 곡선: src/output/week4_multiclass/phase_h_comparison.png")
    print(f"  모델     : {MODEL_OUT.relative_to(BASE_DIR)}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
