"""
Phase G: ZON 일반화 실험 - 라벨링 곡선

실험 설계
----------
  독립변수: 학습에 포함한 ZON 수 N  (1, 3, 5, 10, 20)
  종속변수: 미학습 ZON에서의 Recall  (pipe / rebar / tunnel)

파이프라인
----------
  1. ZON 디렉토리 수집, manifest 사용 ZON 제외
  2. 클래스당 N_TEST_PER_CLS ZON → 고정 테스트 풀 (seed=42)
  3. N = 1, 3, 5, 10, 20 각각:
       a. 학습 풀에서 N ZON 선택 → 신호 에너지 자동 라벨링
       b. FDTD 합성 데이터와 혼합 → YOLO 데이터셋 구성
       c. E-2 가중치에서 30에폭 fine-tuning  (이미 존재하면 스킵)
       d. 테스트 ZON에서 클래스별 Recall 계산
  4. recall_results.csv + 학습 곡선 PNG 저장

사전 요구사항
-------------
  - models/yolo_runs/finetune_gz_e2/run/weights/best.pt  (Phase E-2 결과)
  - pip install ultralytics

사용법
------
  python src/phase_g_labeling_curve.py
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

# ── 경로 ──────────────────────────────────────────────────────────────
GZ_DATA    = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST   = BASE_DIR / "guangzhou_labeled/manifest.json"
E2_WEIGHTS = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"
FDTD_DIR   = BASE_DIR / "data/gpr/yolo_fdtd"
SYNTH_DIR  = BASE_DIR / "data/gpr/yolo_multiclass"
WORK_DIR   = BASE_DIR / "data/gpr/yolo_gz_phase_g"   # 임시 데이터셋
MODEL_OUT  = BASE_DIR / "models/yolo_runs/phase_g"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_g"

# ── 실험 상수 ──────────────────────────────────────────────────────────
CATEGORIES     = ["pipe", "rebar", "tunnel"]
CLASS_IDS      = {"pipe": 1, "rebar": 2, "tunnel": 3}
CLASS_NAMES    = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS   = {"pipe": "#3498db", "rebar": "#2ecc71", "tunnel": "#f39c12"}

N_VALUES       = [1, 3, 5, 10, 20]
N_TEST_PER_CLS = 10
N_SYNTH_TRAIN  = 120    # FDTD + 해석적 합성 데이터 수 (train)
SEED           = 42

DT_SEC    = (8.0 / 512) * 1e-9
CONF_EVAL = 0.05          # 평가 conf 임계값
IMGSZ     = 640           # 추론 이미지 크기


# ─────────────────────────────────────────────────────────────────────
# 전처리
# ─────────────────────────────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    """IDS .dt → 640×640 BGR. 실패 시 None."""
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


def detect_signal_bbox(img_gray: np.ndarray) -> tuple[float, float, float, float]:
    """신호 에너지 기반 bbox → (cx, cy, bw, bh) 정규화 [0,1].
    phase_e1_auto_label.py 로직 재사용."""
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

    py, px = int(H * 0.03), int(W * 0.02)
    y1 = max(0, y1 - py);  y2 = min(H - 1, y2 + py)
    x1 = max(0, x1 - px);  x2 = min(W - 1, x2 + px)

    return (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, \
           (x2 - x1) / W,     (y2 - y1) / H


# ─────────────────────────────────────────────────────────────────────
# ZON 수집 & 풀 구성
# ─────────────────────────────────────────────────────────────────────

def collect_zon_dirs(cls: str) -> list[Path]:
    """클래스 하위 모든 *.ZON 폴더 수집 (ASCII 디렉토리 제외)."""
    cls_dir = GZ_DATA / cls
    if not cls_dir.exists():
        return []
    return sorted(
        d for d in cls_dir.rglob("*.ZON")
        if d.is_dir() and "ASCII" not in str(d)
    )


def find_dt_in_zon(zon_dir: Path) -> Path | None:
    """ZON 폴더 안 첫 번째 .dt 파일."""
    dts = sorted(zon_dir.glob("*.dt"))
    return dts[0] if dts else None


def load_manifest_zon_dirs() -> set[Path]:
    """manifest.json에 등록된 ZON 디렉토리 집합 (Phase E-2 학습에 사용된 ZON)."""
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


def build_zon_pools(rng: random.Random) \
        -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    train_pool / test_pool 구성.
      - manifest ZON 제외
      - test_pool : N_TEST_PER_CLS개 고정 (rng로 재현 가능)
      - train_pool: 나머지 전부
    """
    trained      = load_manifest_zon_dirs()
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


# ─────────────────────────────────────────────────────────────────────
# 데이터셋 구성
# ─────────────────────────────────────────────────────────────────────

def build_dataset(sel_zones: dict[str, list[Path]]) -> Path:
    """
    선택된 ZON 이미지 + FDTD 합성 → WORK_DIR 아래 YOLO 데이터셋.

    sel_zones: {cls: [zon_dir, ...]}  (클래스별 N ZON)
    반환: dataset.yaml 경로
    """
    # 디렉토리 초기화
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = WORK_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_s = random.Random(SEED)
    idx   = 0

    # ── 실제 ZON 이미지 (train & val 동일 복사) ──
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
            cx, cy, bw, bh = detect_signal_bbox(gray)
            label_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"

            stem   = f"gz_{cls}_{idx:04d}"
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                continue
            raw_png = buf.tobytes()

            for split in ("train", "val"):
                (WORK_DIR / "images" / split / f"{stem}.png").write_bytes(raw_png)
                (WORK_DIR / "labels" / split / f"{stem}.txt").write_text(label_line)
            idx += 1

    # ── 합성 데이터 (train) ──
    def copy_synth(src_dir: Path, split: str, n: int, prefix: str) -> int:
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            return 0
        imgs = sorted(img_dir.glob("*.png"))
        rng_s.shuffle(imgs)
        cnt = 0
        for img_p in imgs[:n]:
            lbl_p = lbl_dir / img_p.with_suffix(".txt").name
            stem2 = f"{prefix}_{img_p.stem}"
            (WORK_DIR / "images" / split / f"{stem2}.png").write_bytes(
                img_p.read_bytes())
            dst = WORK_DIR / "labels" / split / f"{stem2}.txt"
            dst.write_bytes(lbl_p.read_bytes() if lbl_p.exists() else b"")
            cnt += 1
        return cnt

    for synth_dir, prefix in [(FDTD_DIR, "fdtd"), (SYNTH_DIR, "synth")]:
        if synth_dir.exists():
            copy_synth(synth_dir, "train", N_SYNTH_TRAIN // 2, prefix)
            copy_synth(synth_dir, "val",   N_SYNTH_TRAIN // 6, prefix)

    # ── dataset.yaml ──
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


# ─────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────

def finetune(yaml_path: Path, n: int) -> Path | None:
    """E-2 가중치에서 30에폭 quick fine-tuning.
    이미 best.pt가 존재하면 재학습 없이 경로 반환."""
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


# ─────────────────────────────────────────────────────────────────────
# Recall 평가
# ─────────────────────────────────────────────────────────────────────

def evaluate_recall(
    model,
    test_pools: dict[str, list[Path]],
) -> tuple[dict[str, float], dict[str, dict]]:
    """
    각 클래스의 테스트 ZON에 대해 Recall 계산.

    TP: 해당 클래스가 conf >= CONF_EVAL 로 탐지된 경우
    FN: 탐지 실패
    """
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


# ─────────────────────────────────────────────────────────────────────
# 시각화
# ─────────────────────────────────────────────────────────────────────

def plot_curve(results: list[dict]) -> Path:
    """ZON 수 vs Recall 학습 곡선 PNG 저장."""
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#2a2a4a")

    for cls in CATEGORIES:
        xs = [r["n"] for r in results]
        ys = [r["recall"][cls] for r in results]
        ax.plot(xs, ys, "o-",
                color=CLASS_COLORS[cls], linewidth=2, markersize=8,
                label=cls)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.025, f"{yi:.2f}", ha="center",
                    color=CLASS_COLORS[cls], fontsize=9, fontweight="bold")

    ax.set_xlabel("학습 ZON 수 (클래스당)", color="white", fontsize=12)
    ax.set_ylabel(f"Recall (미학습 ZON {N_TEST_PER_CLS}개 기준)",
                  color="white", fontsize=12)
    ax.set_title(
        f"Phase G: ZON 일반화 실험 - 라벨링 곡선\n"
        f"(E-2 가중치 → 30에폭 fine-tuning  |  conf ≥ {CONF_EVAL})",
        color="white", fontsize=11, fontweight="bold",
    )
    ax.set_xticks([r["n"] for r in results])
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(alpha=0.25, color="white", linestyle="--")
    ax.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=11)

    plt.tight_layout()

    # buffer_rgba 방식 저장 (한글 경로 대비)
    fig.canvas.draw()
    w, h  = fig.canvas.get_width_height()
    buf   = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr   = cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    plt.close()

    out_path = OUTPUT_DIR.parent / "phase_g_labeling_curve.png"
    if ok:
        out_path.write_bytes(encoded.tobytes())
        print(f"  학습 곡선 저장: {out_path.name}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  Phase G: ZON 일반화 실험 - 라벨링 곡선")
    print("=" * 65)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics")
        return

    if not E2_WEIGHTS.exists():
        print(f"[오류] E-2 가중치 없음: {E2_WEIGHTS}")
        print("      Phase E-2 를 먼저 실행하세요.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)

    # ─ 1. ZON 풀 구성 ────────────────────────────────────────────────
    print("\n[1/3] ZON 풀 구성...")
    rng = random.Random(SEED)
    train_pools, test_pools = build_zon_pools(rng)

    max_n = min(len(train_pools[cls]) for cls in CATEGORIES)
    valid_n = [n for n in N_VALUES if n <= max_n]
    if len(valid_n) < len(N_VALUES):
        skipped = [n for n in N_VALUES if n > max_n]
        print(f"  [주의] 학습 풀 부족으로 N={skipped} 제외 (최대={max_n})")
    print(f"\n  테스트 풀: 클래스당 {N_TEST_PER_CLS}개 고정")
    print(f"  실험 N 값: {valid_n}")

    # ─ 2. 기존 결과 로드 (재개 지원) ────────────────────────────────
    csv_path = OUTPUT_DIR / "recall_results.csv"
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
                    "recall": {
                        cls: float(row[f"{cls}_recall"]) for cls in CATEGORIES
                    },
                })
        print(f"\n  기존 결과 로드: N = {sorted(existing_n)}")

    # ─ 3. N별 실험 루프 ──────────────────────────────────────────────
    print(f"\n[2/3] 실험 루프 (총 {len(valid_n)}회)...")
    f_csv = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(f_csv)
    if not existing_n:           # 새 파일이면 헤더 추가
        writer.writerow(csv_header)

    for n in valid_n:
        if n in existing_n:
            print(f"\n  ── N = {n:2d} : 기존 결과 재사용 ──")
            continue

        print(f"\n{'─' * 55}")
        print(f"  실험: N = {n} ZON/클래스")
        print(f"{'─' * 55}")
        t0 = time.time()

        # ZON 선택
        rng_sel   = random.Random(SEED + n)
        sel_zones = {}
        for cls in CATEGORIES:
            pool = train_pools[cls]
            k    = min(n, len(pool))
            sel_zones[cls] = rng_sel.sample(pool, k)
            print(f"  {cls:6s}: {k}개 ZON 선택 (pool={len(pool)})")

        # 데이터셋 구성
        print("\n  데이터셋 구성...")
        yaml_path = build_dataset(sel_zones)

        # Fine-tuning
        print(f"\n  Fine-tuning (30 에폭, E-2 가중치 베이스)...")
        best_pt = finetune(yaml_path, n)
        if best_pt is None:
            print("  [스킵] Fine-tuning 실패")
            continue

        # Recall 평가
        print("\n  Recall 평가 (고정 테스트 ZON)...")
        eval_model = YOLO(str(best_pt))
        recall, det = evaluate_recall(eval_model, test_pools)

        elapsed = time.time() - t0
        print(f"\n  ── N={n} 결과  ({elapsed:.0f}s) ──")
        for cls in CATEGORIES:
            d = det[cls]
            print(f"  {cls:6s}: Recall={recall[cls]:.3f}  "
                  f"TP={d['tp']}, FN={d['fn']}, skip={d['skip']}")

        # CSV 저장
        row_data = (
            [n]
            + [recall[cls]    for cls in CATEGORIES]
            + [det[cls]["tp"]    for cls in CATEGORIES]
            + [det[cls]["total"] for cls in CATEGORIES]
        )
        writer.writerow(row_data)
        f_csv.flush()

        results.append({"n": n, "recall": recall})
        existing_n.add(n)

    f_csv.close()

    # ─ 4. 학습 곡선 ──────────────────────────────────────────────────
    print(f"\n[3/3] 학습 곡선 생성...")
    results.sort(key=lambda r: r["n"])
    if len(results) >= 2:
        plot_curve(results)
    else:
        print("  결과 2개 미만 → 곡선 생략")

    # ─ 최종 요약 ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("Phase G 완료")
    print(f"{'=' * 65}")
    print(f"\n  {'N':>4}  {'pipe':>8}  {'rebar':>8}  {'tunnel':>8}")
    print("  " + "-" * 38)
    for r in results:
        print(f"  {r['n']:>4}  "
              f"{r['recall']['pipe']:>8.3f}  "
              f"{r['recall']['rebar']:>8.3f}  "
              f"{r['recall']['tunnel']:>8.3f}")
    print(f"\n  CSV  : {csv_path}")
    print(f"  곡선 : src/output/week4_multiclass/phase_g_labeling_curve.png")
    print(f"  모델 : {MODEL_OUT}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
