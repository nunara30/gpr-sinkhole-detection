"""
Phase I: bbox v2 ZON 일반화 실험

Phase G (v1, 전폭 bbox) vs Phase I (v2, CC 기반 클래스별 bbox)
  pipe  : CC 기반 → 가장 넓은 connected component (수평 반사 띠)
  rebar : CC 기반 → 상위 5개 면적 CC 합치기 (반복 아치)
  tunnel: v1 유지 (Phase G에서 Recall=1.0 달성, 변경 불필요)

Phase H에서 tunnel을 CC로 바꿨다가 회귀했으므로 tunnel은 v1 유지.

사용법
------
  /c/Python314/python.exe src/phase_i_bbox_v2_experiment.py
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
WORK_DIR   = BASE_DIR / "data/gpr/yolo_gz_phase_i"
MODEL_OUT  = BASE_DIR / "models/yolo_runs/phase_i"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_i"
PHASE_G_CSV = BASE_DIR / "src/output/week4_multiclass/phase_g/recall_results.csv"

# -- 상수 ------------------------------------------------------------------
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
        return cv2.cvtColor(cv2.resize(gray, (640, 640)), cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


# =========================================================================
# bbox 탐지
# =========================================================================

def _to_yolo(x1, y1, x2, y2, H, W):
    pad_y = int(H * 0.02)
    pad_x = int(W * 0.02)
    x1 = max(0,     x1 - pad_x)
    y1 = max(0,     y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh


def detect_bbox_v1_tunnel(img_gray: np.ndarray):
    """tunnel: Phase G v1 그대로 (이미 Recall=1.0)."""
    H, W    = img_gray.shape
    img_f   = img_gray.astype(np.float32)
    row_std = np.std(img_f, axis=1)

    skip_top = max(1, int(H * 0.03))
    thr_row  = np.percentile(row_std[skip_top:], 55)
    act_idx  = np.where(row_std[skip_top:] > thr_row)[0] + skip_top

    if len(act_idx) < 5:
        y1, y2 = skip_top, int(H * 0.40)
    else:
        y1, y2 = int(act_idx[0]), int(act_idx[-1])

    region = img_f[y1:y2 + 1]
    if region.shape[0] < 2:
        x1, x2 = 0, W - 1
    else:
        col_std  = np.std(region, axis=0)
        act_cols = np.where(col_std > np.percentile(col_std, 15))[0]
        x1, x2  = (int(act_cols[0]), int(act_cols[-1])) \
                  if len(act_cols) >= 5 else (0, W - 1)

    return _to_yolo(x1, y1, x2, y2, H, W)


def _fallback(H, W, skip_top):
    return _to_yolo(0, skip_top, W - 1, int(H * 0.50), H, W)


def detect_bbox_v2_pipe(img_gray: np.ndarray):
    """pipe: CC 기반 - 가장 넓은 수평 반사 띠 (Phase E-2 detect_bbox_v2 계승)."""
    H, W     = img_gray.shape
    skip_top = max(1, int(H * 0.05))
    work     = img_gray[skip_top:, :].astype(np.float32)

    max_y       = int(work.shape[0] * 0.60)
    work_region = work[:max(5, max_y), :]

    thresh_val = np.percentile(work_region, 72)
    binary     = (work_region > thresh_val).astype(np.uint8) * 255

    kx = max(3, W // 30)
    ky = max(3, H // 60)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
    if num_labels <= 1:
        return _fallback(H, W, skip_top)

    min_w = int(W * 0.06)
    valid = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= 3:
            valid.append({'x': x, 'y': y + skip_top,
                          'x2': x + w, 'y2': y + h + skip_top, 'w': w})

    if not valid:
        return _fallback(H, W, skip_top)

    # pipe: 가장 넓은 CC
    best = max(valid, key=lambda c: c['w'])
    return _to_yolo(best['x'], best['y'], best['x2'], best['y2'], H, W)


def detect_bbox_v2_rebar(img_gray: np.ndarray):
    """rebar: CC 기반 - 상위 5개 면적 CC 합치기 (Phase E-2 detect_bbox_v2 계승)."""
    H, W     = img_gray.shape
    skip_top = max(1, int(H * 0.05))
    work     = img_gray[skip_top:, :].astype(np.float32)

    max_y       = int(work.shape[0] * 0.50)
    work_region = work[:max(5, max_y), :]

    thresh_val = np.percentile(work_region, 72)
    binary     = (work_region > thresh_val).astype(np.uint8) * 255

    kx = max(3, W // 30)
    ky = max(3, H // 60)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)
    if num_labels <= 1:
        return _fallback(H, W, skip_top)

    min_w = int(W * 0.06)
    valid = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= 3:
            valid.append({'x': x, 'y': y + skip_top,
                          'x2': x + w, 'y2': y + h + skip_top, 'area': area})

    if not valid:
        return _fallback(H, W, skip_top)

    # rebar: 상위 5개 면적 CC 합치기
    top5 = sorted(valid, key=lambda c: c['area'], reverse=True)[:5]
    x1 = min(c['x']  for c in top5)
    y1 = min(c['y']  for c in top5)
    x2 = max(c['x2'] for c in top5)
    y2 = max(c['y2'] for c in top5)
    return _to_yolo(x1, y1, x2, y2, H, W)


def detect_bbox(img_gray: np.ndarray, cls_name: str):
    """클래스별 최적 bbox 탐지."""
    if cls_name == "pipe":
        return detect_bbox_v2_pipe(img_gray)
    elif cls_name == "rebar":
        return detect_bbox_v2_rebar(img_gray)
    else:   # tunnel
        return detect_bbox_v1_tunnel(img_gray)


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

    return train_pools, test_pools


# =========================================================================
# 데이터셋 구성
# =========================================================================

def build_dataset(sel_zones: dict) -> Path:
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = WORK_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_s = random.Random(SEED)
    idx   = 0

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
            cx, cy, bw, bh = detect_bbox(gray, cls)
            label_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

            stem = f"gz_{cls}_{idx:04d}"
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                continue

            for split in ("train", "val"):
                (WORK_DIR / "images" / split / f"{stem}.png").write_bytes(buf.tobytes())
                (WORK_DIR / "labels" / split / f"{stem}.txt").write_text(label_line)
            idx += 1

    def copy_synth(src_dir: Path, split: str, n: int, prefix: str):
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            return
        imgs = sorted(img_dir.glob("*.png"))
        rng_s.shuffle(imgs)
        for img_p in imgs[:n]:
            lbl_p = lbl_dir / img_p.with_suffix(".txt").name
            stem2 = f"{prefix}_{img_p.stem}"
            (WORK_DIR / "images" / split / f"{stem2}.png").write_bytes(img_p.read_bytes())
            dst = WORK_DIR / "labels" / split / f"{stem2}.txt"
            dst.write_bytes(lbl_p.read_bytes() if lbl_p.exists() else b"")

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
    print(f"    데이터셋: train={n_tr}, val={n_vl}  (ZON={idx}개)")
    return yaml_path


# =========================================================================
# Fine-tuning
# =========================================================================

def finetune(yaml_path: Path, n: int):
    from ultralytics import YOLO

    run_name = f"run_n{n:02d}"
    best_pt  = MODEL_OUT / run_name / "weights/best.pt"

    if best_pt.exists():
        print(f"    [스킵] 이미 학습: {best_pt.relative_to(BASE_DIR)}")
        return best_pt

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
                skip += 1; continue
            bgr = preprocess_dt(dt)
            if bgr is None:
                skip += 1; continue

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
# 비교 시각화
# =========================================================================

def load_phase_g_results():
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


def plot_comparison(results_i: list):
    results_g = load_phase_g_results()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1a1a2e")
    fig.suptitle(
        "Phase G (v1: 에너지 bbox) vs Phase I (v2: CC bbox)\n"
        "pipe/rebar: CC 기반 | tunnel: v1 유지 | E-2 30ep fine-tuning",
        color="white", fontsize=11, fontweight="bold",
    )

    xs_i = [r["n"] for r in results_i]

    for ax, cls in zip(axes, CATEGORIES):
        ax.set_facecolor("#2a2a4a")
        color = CLASS_COLORS[cls]

        ys_i = [r["recall"][cls] for r in results_i]
        ax.plot(xs_i, ys_i, "o-", color=color, linewidth=2.5,
                markersize=8, label="I (v2 CC)", zorder=3)
        for xi, yi in zip(xs_i, ys_i):
            ax.text(xi, yi + 0.05, f"{yi:.2f}", ha="center",
                    color=color, fontsize=9, fontweight="bold")

        if results_g:
            xs_g = [r["n"] for r in results_g]
            ys_g = [r["recall"][cls] for r in results_g]
            ax.plot(xs_g, ys_g, "s--", color=color, linewidth=1.5,
                    markersize=6, alpha=0.5, label="G (v1 energy)")

        ax.set_title(cls, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("학습 ZON 수", color="white", fontsize=10)
        ax.set_ylabel("Recall", color="white", fontsize=10)
        ax.set_xticks(xs_i)
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

    out_path = OUTPUT_DIR.parent / "phase_i_comparison.png"
    if ok:
        out_path.write_bytes(encoded.tobytes())
        print(f"  비교 곡선: {out_path.name}")


# =========================================================================
# 메인
# =========================================================================

def main():
    print("\n" + "=" * 65)
    print("  Phase I: bbox v2(CC) ZON 일반화 실험")
    print("  pipe/rebar: CC 기반 | tunnel: v1 유지")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics"); return

    if not E2_WEIGHTS.exists():
        print(f"[오류] E-2 가중치 없음: {E2_WEIGHTS}"); return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] ZON 풀 구성...")
    rng = random.Random(SEED)
    train_pools, test_pools = build_zon_pools(rng)

    max_n   = min(len(train_pools[cls]) for cls in CATEGORIES)
    valid_n = [n for n in N_VALUES if n <= max_n]
    print(f"  실험 N 값: {valid_n}")

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
        print(f"  기존 결과: N={sorted(existing_n)}")

    print(f"\n[2/3] 실험 루프 ({len(valid_n)}회)...")
    f_csv  = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if not existing_n:
        writer.writerow(csv_header)

    for n in valid_n:
        if n in existing_n:
            print(f"\n  -- N={n}: 기존 결과 재사용 --"); continue

        print(f"\n{'─'*55}")
        print(f"  N = {n} ZON/클래스")
        print(f"{'─'*55}")
        t0 = time.time()

        rng_sel   = random.Random(SEED + n)
        sel_zones = {
            cls: rng_sel.sample(train_pools[cls], min(n, len(train_pools[cls])))
            for cls in CATEGORIES
        }
        for cls in CATEGORIES:
            print(f"  {cls}: {len(sel_zones[cls])}개 ZON")

        print("  데이터셋 구성...")
        yaml_path = build_dataset(sel_zones)

        print("  Fine-tuning...")
        best_pt = finetune(yaml_path, n)
        if best_pt is None:
            print("  [스킵] Fine-tuning 실패"); continue

        print("  Recall 평가...")
        eval_model = YOLO(str(best_pt))
        recall, det = evaluate_recall(eval_model, test_pools)

        elapsed = time.time() - t0
        print(f"\n  N={n} 결과 ({elapsed:.0f}s)")
        for cls in CATEGORIES:
            d = det[cls]
            print(f"  {cls:6s}: Recall={recall[cls]:.3f}  TP={d['tp']} FN={d['fn']}")

        writer.writerow(
            [n]
            + [recall[cls]       for cls in CATEGORIES]
            + [det[cls]["tp"]    for cls in CATEGORIES]
            + [det[cls]["total"] for cls in CATEGORIES]
        )
        f_csv.flush()
        results.append({"n": n, "recall": recall})
        existing_n.add(n)

    f_csv.close()

    print(f"\n[3/3] 비교 곡선...")
    results.sort(key=lambda r: r["n"])
    if len(results) >= 2:
        plot_comparison(results)

    results_g = load_phase_g_results()
    g_map = {r["n"]: r["recall"] for r in results_g}
    i_map = {r["n"]: r["recall"] for r in results}

    print(f"\n{'='*65}")
    print("Phase I 완료 - G vs I 비교")
    print(f"{'='*65}")
    print(f"\n  {'N':>4}  {'pipe_G':>8} {'pipe_I':>8}  {'rebar_G':>8} {'rebar_I':>8}  {'tunnel_G':>9} {'tunnel_I':>9}")
    print("  " + "-" * 65)
    for n in sorted(set(g_map) & set(i_map)):
        g = g_map[n]
        i = i_map[n]
        print(f"  {n:>4}  "
              f"{g['pipe']:>8.3f} {i['pipe']:>8.3f}  "
              f"{g['rebar']:>8.3f} {i['rebar']:>8.3f}  "
              f"{g['tunnel']:>9.3f} {i['tunnel']:>9.3f}")

    print(f"\n  CSV: {csv_path.relative_to(BASE_DIR)}")
    print(f"  비교곡선: src/output/week4_multiclass/phase_i_comparison.png")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
