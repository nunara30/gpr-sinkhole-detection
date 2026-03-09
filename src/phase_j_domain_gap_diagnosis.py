"""
Phase J: 도메인 갭 원인 진단 실험

가설
----
  H1 (다양성 부족): 같은 MIS 내 ZON끼리는 신호 패턴이 유사 → 테스트가 같은 MIS면 Recall 높음
  H2 (근본적 차이): ZON마다 신호가 본질적으로 달라 → 같은 MIS 내에서도 Recall 낮음

실험 설계
---------
  조건 A (Near-domain):  학습 = MIS_X 앞 N ZON, 테스트 = MIS_X 뒤 N ZON  (같은 MIS)
  조건 B (Far-domain):   학습 = MIS_X 앞 N ZON, 테스트 = MIS_Y ZON         (다른 MIS)

  → 조건 A Recall >> 조건 B Recall : H1 지지 (데이터 다양성 문제)
  → 조건 A Recall ≈  조건 B Recall : H2 지지 (신호 자체 문제)

클래스별 MIS 매핑
-----------------
  pipe  : 기준 MIS = 001.MIS (6 ZON), 테스트용 원거리 = 1.15.MIS (10 ZON)
  rebar : 기준 MIS = foshanJL.MIS (48 ZON), 테스트용 원거리 = yangben.MIS (5 ZON)

부가 시각화
-----------
  각 클래스별 MIS별 GPR 이미지 그리드 → 육안 패턴 비교

사용법
------
  /c/Python314/python.exe src/phase_j_domain_gap_diagnosis.py
"""

import sys
import json
import shutil
import random
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
WORK_DIR   = BASE_DIR / "data/gpr/yolo_gz_phase_j"
MODEL_OUT  = BASE_DIR / "models/yolo_runs/phase_j"
OUTPUT_DIR = BASE_DIR / "src/output/week4_multiclass/phase_j"

# ── 실험 상수 ──────────────────────────────────────────────────────────
CLASS_IDS   = {"pipe": 1, "rebar": 2, "tunnel": 3}
CLASS_NAMES = ["sinkhole", "pipe", "rebar", "tunnel"]
CONF_EVAL   = 0.05
IMGSZ       = 640
DT_SEC      = (8.0 / 512) * 1e-9
N_SYNTH     = 120
SEED        = 42

# ── MIS 설정 ───────────────────────────────────────────────────────────
MIS_CONFIG = {
    "pipe": {
        "base_mis":  "1030.MIS",         # 42 ZON available (학습+Near 테스트용)
        "far_mis":   "1.15.MIS",         # 7 ZON available (Far 테스트용)
        "n_train":   5,
        "n_test":    5,
    },
    "rebar": {
        "base_mis":  "foshanJL.MIS",     # 45 ZON available
        "far_mis":   "yangben2.MIS",     # 4 ZON available
        "n_train":   5,
        "n_test":    5,
    },
}


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
    """Phase G와 동일한 에너지 기반 bbox."""
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
# ZON 수집
# ─────────────────────────────────────────────────────────────────────

def collect_zon_dirs_in_mis(cls: str, mis_name: str) -> list[Path]:
    """특정 MIS 폴더 안의 ZON 디렉토리만 반환."""
    mis_dir = GZ_DATA / cls / mis_name
    if not mis_dir.exists():
        return []
    return sorted(
        d for d in mis_dir.iterdir()
        if d.is_dir() and d.suffix.upper() == ".ZON"
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


# ─────────────────────────────────────────────────────────────────────
# 데이터셋 구성
# ─────────────────────────────────────────────────────────────────────

def build_dataset(train_zones: dict[str, list[Path]], tag: str) -> Path:
    """train_zones: {cls: [zon_dir, ...]} → YOLO 데이터셋 WORK_DIR/tag/"""
    ds_dir = WORK_DIR / tag
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        d = ds_dir / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_s = random.Random(SEED)
    idx   = 0

    for cls, zon_list in train_zones.items():
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
            stem = f"gz_{cls}_{idx:04d}"
            ok, buf = cv2.imencode(".png", bgr)
            if not ok:
                continue
            raw_png = buf.tobytes()
            for split in ("train", "val"):
                (ds_dir / "images" / split / f"{stem}.png").write_bytes(raw_png)
                (ds_dir / "labels" / split / f"{stem}.txt").write_text(label_line)
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
            (ds_dir / "images" / split / f"{stem2}.png").write_bytes(img_p.read_bytes())
            dst = ds_dir / "labels" / split / f"{stem2}.txt"
            dst.write_bytes(lbl_p.read_bytes() if lbl_p.exists() else b"")

    for synth_src, pfx in [(FDTD_DIR, "fdtd"), (SYNTH_DIR, "synth")]:
        if synth_src.exists():
            copy_synth(synth_src, "train", N_SYNTH // 2, pfx)
            copy_synth(synth_src, "val",   N_SYNTH // 6, pfx)

    yaml_path = ds_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {ds_dir.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 4\n"
        "names: ['sinkhole', 'pipe', 'rebar', 'tunnel']\n",
        encoding="utf-8",
    )

    n_tr = len(list((ds_dir / "images/train").glob("*.png")))
    print(f"    데이터셋 [{tag}]: train={n_tr}  (ZON={idx}개)")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────

def finetune(yaml_path: Path, tag: str) -> Path | None:
    from ultralytics import YOLO

    best_pt = MODEL_OUT / tag / "weights/best.pt"
    if best_pt.exists():
        print(f"    [스킵] {best_pt.relative_to(BASE_DIR)}")
        return best_pt
    if not E2_WEIGHTS.exists():
        print(f"    [오류] E-2 가중치 없음")
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
        name=tag,
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

def evaluate_recall(model, test_zones: dict[str, list[Path]]) -> dict[str, float]:
    recall = {}
    for cls, zon_list in test_zones.items():
        tp = fn = 0
        for zon_dir in zon_list:
            dt = find_dt_in_zon(zon_dir)
            if dt is None:
                continue
            bgr = preprocess_dt(dt)
            if bgr is None:
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
        recall[cls] = round(tp / total, 4) if total > 0 else 0.0
        print(f"      {cls}: TP={tp} FN={fn} → Recall={recall[cls]:.3f}")
    return recall


# ─────────────────────────────────────────────────────────────────────
# 시각화 A: MIS별 GPR 이미지 그리드 (패턴 시각 비교)
# ─────────────────────────────────────────────────────────────────────

def plot_mis_visual_grid():
    """각 클래스×MIS 조합에서 최대 3개 ZON 이미지를 그리드로 저장."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls, cfg in MIS_CONFIG.items():
        base_mis = cfg["base_mis"]
        far_mis  = cfg["far_mis"]

        rows = []  # [(mis_label, bgr_list), ...]
        for mis_name, label in [(base_mis, f"기준 MIS\n({base_mis})"),
                                (far_mis,  f"원거리 MIS\n({far_mis})")]:
            zones = collect_zon_dirs_in_mis(cls, mis_name)[:4]
            imgs  = []
            for z in zones:
                dt = find_dt_in_zon(z)
                if dt is None:
                    continue
                bgr = preprocess_dt(dt)
                if bgr is not None:
                    imgs.append(cv2.cvtColor(cv2.resize(bgr, (200, 200)),
                                             cv2.COLOR_BGR2RGB))
                if len(imgs) >= 3:
                    break
            rows.append((label, imgs))

        n_cols = 3
        fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 6),
                                 facecolor="#1a1a2e")
        fig.suptitle(f"[{cls}] MIS별 GPR 이미지 패턴 비교",
                     color="white", fontsize=13, fontweight="bold")

        for row_i, (mis_label, imgs) in enumerate(rows):
            for col_i in range(n_cols):
                ax = axes[row_i][col_i]
                ax.set_facecolor("#111")
                if col_i < len(imgs):
                    ax.imshow(imgs[col_i], cmap="gray")
                    if col_i == 0:
                        ax.set_ylabel(mis_label, color="white",
                                      fontsize=9, labelpad=8)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                            color="#666", transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#333")

        plt.tight_layout()
        out_path = OUTPUT_DIR / f"visual_grid_{cls}.png"
        ok, buf = cv2.imencode(".png",
                               cv2.cvtColor(
                                   np.array(fig.canvas.buffer_rgba())[:, :, :3],
                                   cv2.COLOR_RGB2BGR))
        fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                    facecolor="#1a1a2e")
        plt.close(fig)
        print(f"  시각화 저장: {out_path.relative_to(BASE_DIR)}")


# ─────────────────────────────────────────────────────────────────────
# 시각화 B: Near vs Far Recall 막대그래프
# ─────────────────────────────────────────────────────────────────────

def plot_near_far_bar(results: dict):
    """
    results = {
      "pipe":  {"near": 0.33, "far": 0.0},
      "rebar": {"near": 0.40, "far": 0.0},
    }
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    classes  = list(results.keys())
    near_vals = [results[c]["near"] for c in classes]
    far_vals  = [results[c]["far"]  for c in classes]

    x    = np.arange(len(classes))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#1a1a2e")
    ax.set_facecolor("#2a2a4a")

    bars_near = ax.bar(x - w/2, near_vals, w, label="Near-domain (같은 MIS)",
                       color="#3498db", alpha=0.9)
    bars_far  = ax.bar(x + w/2, far_vals,  w, label="Far-domain (다른 MIS)",
                       color="#e74c3c", alpha=0.9)

    for bar in bars_near:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", color="white",
                fontsize=11, fontweight="bold")
    for bar in bars_far:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", color="white",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="white", fontsize=12)
    ax.set_ylabel("Recall", color="white", fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.set_title(
        "Phase J: Near vs Far Domain Recall\n"
        "(같은 MIS 내 일반화  vs  다른 MIS 일반화)",
        color="white", fontsize=11, fontweight="bold",
    )
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.grid(axis="y", alpha=0.25, color="white", linestyle="--")
    ax.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=10)

    # 해석 텍스트
    for cls_i, cls in enumerate(classes):
        nr = results[cls]["near"]
        fr = results[cls]["far"]
        if nr > fr + 0.1:
            verdict = "H1 지지\n(다양성 부족)"
            col = "#f39c12"
        elif nr < 0.15 and fr < 0.15:
            verdict = "H2 지지\n(근본 차이)"
            col = "#e74c3c"
        else:
            verdict = "불명확"
            col = "#aaaaaa"
        ax.text(x[cls_i], 1.05, verdict, ha="center", color=col,
                fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "near_vs_far_recall.png"
    fig.savefig(str(out_path), dpi=100, bbox_inches="tight",
                facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  결과 그래프: {out_path.relative_to(BASE_DIR)}")
    return out_path


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trained = load_manifest_zon_dirs()

    print("\n" + "=" * 65)
    print("  Phase J: 도메인 갭 원인 진단 실험")
    print("=" * 65)
    print("\n  가설:")
    print("    H1 - 데이터 다양성 부족 → Near Recall >> Far Recall")
    print("    H2 - 신호 자체 차이     → Near ≈ Far (둘 다 낮음)")

    # ── 1. 시각 비교 그리드 (학습 없이 즉시 생성) ───────────────
    print("\n[Step 1] MIS별 GPR 이미지 시각 비교 그리드 생성...")
    plot_mis_visual_grid()

    # ── 2. Near vs Far 실험 ──────────────────────────────────────
    results = {}

    for cls, cfg in MIS_CONFIG.items():
        print(f"\n{'─'*55}")
        print(f"  [{cls.upper()}]")
        print(f"    기준 MIS : {cfg['base_mis']}")
        print(f"    원거리MIS: {cfg['far_mis']}")

        base_zones = [z for z in collect_zon_dirs_in_mis(cls, cfg["base_mis"])
                      if z not in trained]
        far_zones  = [z for z in collect_zon_dirs_in_mis(cls, cfg["far_mis"])
                      if z not in trained]

        n_train = cfg["n_train"]
        n_test  = cfg["n_test"]

        if len(base_zones) < n_train + n_test:
            print(f"    [경고] {cls}/{cfg['base_mis']} ZON 부족 "
                  f"({len(base_zones)} < {n_train + n_test}). 건너뜀.")
            continue
        if len(far_zones) < 2:
            print(f"    [경고] {cls}/{cfg['far_mis']} ZON 너무 부족 "
                  f"({len(far_zones)}개). 건너뜀.")
            continue
        # far_test는 실제 가용 ZON 수에 맞게 조정
        n_test = min(n_test, len(far_zones))

        train_zon = base_zones[:n_train]
        near_test = {cls: base_zones[n_train: n_train + n_test]}
        far_test  = {cls: far_zones[:n_test]}

        print(f"    학습 ZON: {n_train}개  |  Near테스트: {n_test}개  |  Far테스트: {n_test}개")

        # Near 실험
        print(f"\n  --- Near-domain 학습/평가 ---")
        tag_near = f"{cls}_near"
        yaml_near = build_dataset({cls: train_zon}, tag_near)
        pt_near   = finetune(yaml_near, tag_near)
        near_recall = 0.0
        if pt_near:
            model = YOLO(str(pt_near))
            model.fuse()
            r = evaluate_recall(model, near_test)
            near_recall = r.get(cls, 0.0)

        # Far 실험
        print(f"\n  --- Far-domain 학습/평가 ---")
        tag_far = f"{cls}_far"
        yaml_far = build_dataset({cls: train_zon}, tag_far)
        pt_far   = finetune(yaml_far, tag_far)
        far_recall = 0.0
        if pt_far:
            model = YOLO(str(pt_far))
            model.fuse()
            r = evaluate_recall(model, far_test)
            far_recall = r.get(cls, 0.0)

        results[cls] = {"near": near_recall, "far": far_recall}
        print(f"\n  [{cls}] Near={near_recall:.3f}  Far={far_recall:.3f}")

    # ── 3. 결과 시각화 & 판정 ───────────────────────────────────
    if results:
        print("\n[Step 3] 결과 그래프 생성...")
        plot_near_far_bar(results)

        print("\n" + "=" * 65)
        print("  [Phase J 진단 결론]")
        print("=" * 65)
        for cls, r in results.items():
            nr, fr = r["near"], r["far"]
            diff = nr - fr
            if nr > fr + 0.15:
                verdict = "H1 지지 → 데이터 다양성 문제 (여러 MIS 학습 필요)"
            elif nr < 0.15 and fr < 0.15:
                verdict = "H2 지지 → ZON간 신호 자체가 본질적으로 다름"
            else:
                verdict = f"불명확 (차이={diff:+.3f}) → 추가 실험 필요"
            print(f"  {cls:6s}: Near={nr:.3f}  Far={fr:.3f}  → {verdict}")

        print(f"\n  출력: {OUTPUT_DIR.relative_to(BASE_DIR)}/")
    else:
        print("\n  [오류] 실험 결과 없음 - ZON 수 부족 또는 가중치 문제")


if __name__ == "__main__":
    main()
