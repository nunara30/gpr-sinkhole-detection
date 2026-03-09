"""
Phase I: 수동 라벨 + E-2 라벨 혼합 학습 & ZON 일반화 평가

phase_i_label_tool.py 로 라벨링 완료 후 실행.

파이프라인
----------
  1. 수동 라벨 (data/gpr/phase_i_manual/) 확인
  2. E-2 기존 라벨 + 수동 라벨 + 합성 데이터 혼합
  3. E-2 가중치에서 50 에폭 fine-tuning
  4. Phase G 고정 테스트 ZON에서 Recall 평가
  5. Phase G/H vs Phase I 비교 곡선 저장

사용법
------
  /c/Python314/python.exe src/phase_i_train.py
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
GZ_DATA       = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST      = BASE_DIR / "guangzhou_labeled/manifest.json"
E2_WEIGHTS    = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"
FDTD_DIR      = BASE_DIR / "data/gpr/yolo_fdtd"
SYNTH_DIR     = BASE_DIR / "data/gpr/yolo_multiclass"
MANUAL_DIR    = BASE_DIR / "data/gpr/phase_i_manual"
GZ_LABELED    = BASE_DIR / "guangzhou_labeled"      # E-2 기존 수동 라벨
WORK_DIR      = BASE_DIR / "data/gpr/yolo_gz_phase_i"
MODEL_OUT     = BASE_DIR / "models/yolo_runs/phase_i"
OUTPUT_DIR    = BASE_DIR / "src/output/week4_multiclass/phase_i"
PHASE_G_CSV   = BASE_DIR / "src/output/week4_multiclass/phase_g/recall_results.csv"

# -- 상수 ------------------------------------------------------------------
CATEGORIES   = ["pipe", "rebar", "tunnel"]
CLASS_IDS    = {"pipe": 1, "rebar": 2, "tunnel": 3}
CLASS_NAMES  = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS = {"pipe": "#3498db", "rebar": "#2ecc71", "tunnel": "#f39c12"}

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
# ZON 풀 구성 (Phase G와 동일 seed → 동일 test pool)
# =========================================================================

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


def collect_zon_dirs(cls: str) -> list[Path]:
    cls_dir = GZ_DATA / cls
    if not cls_dir.exists():
        return []
    return sorted(
        d for d in cls_dir.rglob("*.ZON")
        if d.is_dir() and "ASCII" not in str(d)
    )


def build_test_pool() -> dict[str, list[Path]]:
    """Phase G와 동일한 seed=42로 고정 테스트 풀 재구성."""
    rng     = random.Random(SEED)
    trained = load_manifest_zon_dirs()
    test_pools: dict[str, list[Path]] = {}

    for cls in CATEGORIES:
        all_zons  = collect_zon_dirs(cls)
        available = [z for z in all_zons if z not in trained]
        shuffled  = available.copy()
        rng.shuffle(shuffled)
        n_test = min(N_TEST_PER_CLS, max(1, len(shuffled) // 2))
        test_pools[cls] = shuffled[:n_test]

    return test_pools


def find_dt_in_zon(zon_dir: Path):
    dts = sorted(zon_dir.glob("*.dt"))
    return dts[0] if dts else None


# =========================================================================
# 데이터셋 구성
# =========================================================================

def build_dataset() -> Path:
    """
    수동 라벨 + E-2 기존 라벨 + 합성 → YOLO 데이터셋.

    소스별 이미지 수:
      - 수동 라벨 (Phase I): pipe+rebar 최대 20장
      - E-2 기존 라벨 (guangzhou_labeled/): pipe 15 + rebar 10 + tunnel 10
      - 합성 (FDTD+analytic): 120장
    """
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = WORK_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_s = random.Random(SEED)
    idx   = 0
    stats = {"manual": 0, "e2": 0, "synth": 0}

    # 1. 수동 라벨 (Phase I)
    manual_imgs = list((MANUAL_DIR / "images").glob("*.png")) \
        if (MANUAL_DIR / "images").exists() else []
    manual_imgs = [
        p for p in manual_imgs
        if (MANUAL_DIR / "labels" / p.with_suffix(".txt").name).exists()
        and (MANUAL_DIR / "labels" / p.with_suffix(".txt").name).stat().st_size > 0
    ]

    if not manual_imgs:
        print("  [경고] 수동 라벨 없음. phase_i_label_tool.py 먼저 실행하세요.")

    for img_p in manual_imgs:
        lbl_p = MANUAL_DIR / "labels" / img_p.with_suffix(".txt").name
        stem  = f"manual_{idx:04d}"
        for split in ("train", "val"):
            (WORK_DIR / "images" / split / f"{stem}.png").write_bytes(img_p.read_bytes())
            (WORK_DIR / "labels" / split / f"{stem}.txt").write_bytes(lbl_p.read_bytes())
        idx += 1
        stats["manual"] += 1

    # 2. E-2 기존 라벨 (guangzhou_labeled/)
    gz_img_dir = GZ_LABELED / "images"
    gz_lbl_dir = GZ_LABELED / "labels"
    if gz_img_dir.exists():
        gz_imgs = sorted(gz_img_dir.glob("*.png"))
        for img_p in gz_imgs:
            lbl_p = gz_lbl_dir / img_p.with_suffix(".txt").name
            if not lbl_p.exists():
                continue
            stem = f"e2_{idx:04d}"
            for split in ("train", "val"):
                (WORK_DIR / "images" / split / f"{stem}.png").write_bytes(img_p.read_bytes())
                (WORK_DIR / "labels" / split / f"{stem}.txt").write_bytes(lbl_p.read_bytes())
            idx += 1
            stats["e2"] += 1

    # 3. 합성 데이터
    def copy_synth(src_dir: Path, split: str, n: int, prefix: str):
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
            n = copy_synth(synth_dir, "train", N_SYNTH_TRAIN // 2, prefix)
            copy_synth(synth_dir, "val", N_SYNTH_TRAIN // 6, prefix)
            stats["synth"] += n

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
    print(f"  데이터셋: train={n_tr}, val={n_vl}")
    print(f"    수동라벨={stats['manual']}  E-2라벨={stats['e2']}  합성={stats['synth']}")
    return yaml_path


# =========================================================================
# Fine-tuning
# =========================================================================

def finetune(yaml_path: Path):
    from ultralytics import YOLO

    best_pt = MODEL_OUT / "run" / "weights/best.pt"
    if best_pt.exists():
        print(f"  [스킵] 이미 학습된 가중치: {best_pt.relative_to(BASE_DIR)}")
        return best_pt

    if not E2_WEIGHTS.exists():
        print(f"  [오류] E-2 가중치 없음: {E2_WEIGHTS}")
        return None

    model = YOLO(str(E2_WEIGHTS))
    model.train(
        data=str(yaml_path),
        epochs=50,          # Phase G/H보다 많이 (수동 라벨 충분히 학습)
        batch=4,
        imgsz=416,
        lr0=3e-5,           # 더 낮은 LR (정밀 수동 라벨 overfit 방지)
        lrf=0.01,
        optimizer="AdamW",
        cos_lr=True,
        freeze=3,           # freeze 5 → 3 (더 많은 레이어 fine-tune)
        dropout=0.1,
        patience=20,
        warmup_epochs=3,
        mosaic=0.0,
        plots=False,
        project=str(MODEL_OUT),
        name="run",
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
# 비교 시각화
# =========================================================================

def load_csv_results(csv_path: Path, phase_label: str) -> list[dict]:
    if not csv_path.exists():
        return []
    results = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                results.append({
                    "n": int(row["n_zon"]),
                    "recall": {cls: float(row[f"{cls}_recall"]) for cls in CATEGORIES},
                    "phase": phase_label,
                })
    except Exception:
        pass
    return sorted(results, key=lambda r: r["n"])


def plot_comparison(recall_i: dict):
    """Phase G(점선) vs Phase I(수평 실선) 비교."""
    results_g = load_csv_results(PHASE_G_CSV, "G")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#1a1a2e")
    fig.suptitle(
        "Phase G(자동라벨 v1) vs Phase I(수동라벨 추가)\n"
        f"Phase I: E-2 라벨 + 수동 pipe/rebar + 합성 | conf>={CONF_EVAL}",
        color="white", fontsize=11, fontweight="bold",
    )

    for ax, cls in zip(axes, CATEGORIES):
        ax.set_facecolor("#2a2a4a")
        color = CLASS_COLORS[cls]

        # Phase I (수평 점선 - 단일 값)
        ri = recall_i.get(cls, 0.0)
        if results_g:
            xs = [r["n"] for r in results_g]
            ax.axhline(ri, color=color, linewidth=2.5, linestyle="-",
                       label=f"I (수동라벨) = {ri:.3f}", zorder=3)

        # Phase G (점선 곡선)
        if results_g:
            xs = [r["n"] for r in results_g]
            ys = [r["recall"][cls] for r in results_g]
            ax.plot(xs, ys, "s--", color=color, linewidth=1.5,
                    markersize=6, alpha=0.6, label="G (자동라벨 v1)")
            for xi, yi in zip(xs, ys):
                ax.text(xi, yi + 0.04, f"{yi:.2f}", ha="center",
                        color=color, fontsize=8, alpha=0.7)

        ax.set_title(cls, color="white", fontsize=13, fontweight="bold")
        ax.set_xlabel("Phase G 학습 ZON 수", color="white", fontsize=10)
        ax.set_ylabel("Recall", color="white", fontsize=10)
        if results_g:
            ax.set_xticks([r["n"] for r in results_g])
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
        print(f"  비교 곡선 저장: {out_path.name}")
    return out_path


# =========================================================================
# 메인
# =========================================================================

def main():
    print("\n" + "=" * 65)
    print("  Phase I: 수동 라벨 + E-2 라벨 혼합 학습")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics")
        return

    # 수동 라벨 확인
    manual_lbl_dir = MANUAL_DIR / "labels"
    manual_count = 0
    if manual_lbl_dir.exists():
        manual_count = sum(
            1 for f in manual_lbl_dir.glob("*.txt")
            if f.stat().st_size > 0
        )

    print(f"\n  수동 라벨: {manual_count}장")
    if manual_count == 0:
        print("\n  [경고] 수동 라벨이 없습니다.")
        print("  먼저 실행: /c/Python314/python.exe src/phase_i_label_tool.py")
        return
    if manual_count < 5:
        print(f"  [경고] 수동 라벨이 {manual_count}장으로 너무 적습니다.")
        print("  최소 5장 이상 권장. 계속 진행합니까? (y/n)")
        if input().strip().lower() != "y":
            return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # 1. 테스트 풀 (Phase G와 동일)
    print("\n[1/4] 테스트 풀 구성 (Phase G 동일 seed=42)...")
    test_pools = build_test_pool()
    for cls in CATEGORIES:
        print(f"  {cls}: {len(test_pools[cls])}개 ZON")

    # 2. 데이터셋
    print("\n[2/4] 데이터셋 구성...")
    yaml_path = build_dataset()

    # 3. Fine-tuning
    print("\n[3/4] Fine-tuning (50 에폭)...")
    t0      = time.time()
    best_pt = finetune(yaml_path)
    if best_pt is None:
        print("  [오류] Fine-tuning 실패")
        return
    print(f"  완료: {time.time() - t0:.0f}s")

    # 4. Recall 평가
    print("\n[4/4] Recall 평가...")
    model  = YOLO(str(best_pt))
    recall, details = evaluate_recall(model, test_pools)

    print(f"\n{'=' * 65}")
    print("Phase I 결과")
    print(f"{'=' * 65}")
    print(f"\n  {'클래스':8s}  {'Recall':>8}  {'TP':>4}  {'FN':>4}  {'skip':>4}")
    print("  " + "-" * 35)
    for cls in CATEGORIES:
        d = details[cls]
        print(f"  {cls:8s}  {recall[cls]:>8.3f}  "
              f"{d['tp']:>4}  {d['fn']:>4}  {d['skip']:>4}")

    # Phase G 비교
    results_g = load_csv_results(PHASE_G_CSV, "G")
    if results_g:
        print(f"\n  [Phase G N=20 vs Phase I 비교]")
        g20 = next((r for r in results_g if r["n"] == 20), None)
        if g20:
            for cls in CATEGORIES:
                delta = recall[cls] - g20["recall"][cls]
                sign  = "+" if delta >= 0 else ""
                print(f"  {cls:8s}: G={g20['recall'][cls]:.3f}  "
                      f"I={recall[cls]:.3f}  ({sign}{delta:.3f})")

    # CSV 저장
    csv_path = OUTPUT_DIR / "recall_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase"] + [f"{cls}_recall" for cls in CATEGORIES]
                   + [f"{cls}_tp" for cls in CATEGORIES]
                   + [f"{cls}_total" for cls in CATEGORIES])
        w.writerow(
            ["I"]
            + [recall[cls]         for cls in CATEGORIES]
            + [details[cls]["tp"]  for cls in CATEGORIES]
            + [details[cls]["total"] for cls in CATEGORIES]
        )

    # 비교 곡선
    plot_comparison(recall)

    print(f"\n  CSV     : {csv_path.relative_to(BASE_DIR)}")
    print(f"  비교곡선: src/output/week4_multiclass/phase_i_comparison.png")
    print(f"  모델    : {MODEL_OUT.relative_to(BASE_DIR)}/run/weights/best.pt")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
