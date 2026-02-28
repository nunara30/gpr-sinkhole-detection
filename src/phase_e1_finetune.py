"""
Phase E-1: Guangzhou 라벨링 데이터 Fine-tuning

phase_e1_prepare_labeling.py + 수동 라벨링 후 실행.

학습 전략:
  - Guangzhou 라벨링 이미지 25개 (pipe 15, rebar 10)
  - 합성 데이터 (Phase D-2 FDTD + 해석적) 혼합
  - Phase D-2 best weights에서 fine-tuning
  - 학습률 매우 낮게 (실측 분포에 빠르게 적응)

평가:
  - Guangzhou test set: 라벨링에 사용되지 않은 .dt 파일 직접 추론
  - 클래스별 탐지율 비교 (D-2 원본 vs E-1 fine-tuned)

사전 요구사항:
  - phase_e1_prepare_labeling.py 실행 완료
  - guangzhou_labeled/labels/*.txt 수동 라벨링 완료 (비어 있지 않아야 함)
"""

import os
import sys
import json
import shutil
import random
import time
import warnings
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec

# ── 경로 설정 ──
PROJECT_DIR   = Path(__file__).parent.parent
GZ_LABELED    = PROJECT_DIR / "guangzhou_labeled"    # prepare_labeling 출력
GZ_DATA       = PROJECT_DIR / "data" / "gpr" / "guangzhou" / "Data Set"
GZ_PIPE_DIR   = GZ_DATA / "pipe"
GZ_REBAR_DIR  = GZ_DATA / "rebar"

# 프로젝트 기반 경로
RAG_DIR       = PROJECT_DIR                       # E:\...\gpr-sinkhole-detection
DATA_DIR      = RAG_DIR / "data" / "gpr"
SYNTH_DIR     = DATA_DIR / "yolo_multiclass"     # 해석적 합성 데이터셋
FDTD_DIR      = DATA_DIR / "yolo_fdtd"           # FDTD 합성 데이터셋
D2_WEIGHTS    = RAG_DIR / "models" / "yolo_runs" / "finetune_fdtd" / "run" / "weights" / "best.pt"
FT_DIR        = RAG_DIR / "models" / "yolo_runs" / "finetune_gz_e1"
OUTPUT_DIR    = RAG_DIR / "src" / "output" / "week4_multiclass"

MIXED_DIR     = DATA_DIR / "yolo_gz_e1_mixed"    # 혼합 데이터셋 (임시)

# Guangzhou 전처리 파라미터
DT_NS    = 8.0 / 512
DT_SEC   = DT_NS * 1e-9
F_LOW_MHZ  = 500.0
F_HIGH_MHZ = 4000.0

CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']


# ──────────────────────────────────────────────
# 0. 라벨링 상태 확인
# ──────────────────────────────────────────────

def check_labels() -> list[dict]:
    """
    guangzhou_labeled/labels/*.txt 라벨 상태 확인.
    Returns: 라벨이 있는 항목만 포함된 manifest 리스트
    """
    manifest_path = GZ_LABELED / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json 없음: {manifest_path}\n"
            f"먼저 phase_e1_prepare_labeling.py 를 실행하세요."
        )

    with open(manifest_path, encoding='utf-8') as f:
        manifest_data = json.load(f)

    entries = manifest_data["images"]
    labeled, unlabeled = [], []

    for entry in entries:
        lbl_path = GZ_LABELED / "labels" / entry["label"]
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            entry["has_label"] = True
            labeled.append(entry)
        else:
            entry["has_label"] = False
            unlabeled.append(entry)

    print(f"  라벨링 완료: {len(labeled)}/{len(entries)}개")
    if unlabeled:
        names = [e["image"] for e in unlabeled]
        print(f"  라벨 없음:   {len(unlabeled)}개 → 학습에서 제외")
        if len(unlabeled) <= 5:
            for n in names:
                print(f"    - {n}")

    if len(labeled) == 0:
        raise ValueError(
            "라벨링된 이미지가 없습니다.\n"
            "guangzhou_labeled/labels/*.txt 파일을 YOLO 형식으로 작성하세요.\n"
            "비어 있는 파일은 학습에서 제외됩니다."
        )

    return labeled


# ──────────────────────────────────────────────
# 1. 혼합 데이터셋 구성
# ──────────────────────────────────────────────

def build_mixed_dataset(labeled: list[dict],
                        n_synth_train: int = 150,
                        n_synth_val: int   = 30,
                        val_ratio: float   = 0.2,
                        seed: int          = 42) -> tuple[int, int]:
    """
    라벨링된 Guangzhou + 합성 데이터 혼합.
    Returns: (n_train, n_val)
    """
    rng = random.Random(seed)

    # Guangzhou 라벨링 데이터 train/val 분할
    rng.shuffle(labeled)
    n_val_gz  = max(1, int(len(labeled) * val_ratio))
    gz_val    = labeled[:n_val_gz]
    gz_train  = labeled[n_val_gz:]

    print(f"  Guangzhou: train={len(gz_train)}, val={len(gz_val)}")

    # 출력 디렉토리 초기화
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        d = MIXED_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = {"train": 0, "val": 0}

    def copy_gz(items, split):
        for entry in items:
            img_src = GZ_LABELED / "images" / entry["image"]
            lbl_src = GZ_LABELED / "labels" / entry["label"]
            stem = f"gz_{entry['image'].replace('.png', '')}"
            dst_img = MIXED_DIR / "images" / split / f"{stem}.png"
            dst_lbl = MIXED_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_src, dst_img)
            shutil.copy2(lbl_src, dst_lbl)
            count[split] += 1

    def copy_synth(src_dir: Path, split: str, n: int, prefix: str):
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            print(f"    [경고] 합성 데이터 없음: {img_dir}")
            return
        imgs = sorted(img_dir.glob("*.png"))
        rng.shuffle(imgs)
        imgs = imgs[:n]
        for img_path in imgs:
            lbl_path = lbl_dir / img_path.with_suffix('.txt').name
            stem = f"{prefix}_{img_path.stem}"
            dst_img = MIXED_DIR / "images" / split / f"{stem}.png"
            dst_lbl = MIXED_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("")
            count[split] += 1

    # Guangzhou 라벨링 데이터 복사
    copy_gz(gz_train, "train")
    copy_gz(gz_val,   "val")

    # 합성 데이터 추가 (FDTD 우선, 없으면 해석적)
    for synth_dir, prefix in [(FDTD_DIR, "fdtd"), (SYNTH_DIR, "synth")]:
        if synth_dir.exists():
            copy_synth(synth_dir, "train", n_synth_train // 2, prefix)
            copy_synth(synth_dir, "val",   n_synth_val   // 2, prefix)

    print(f"  혼합 데이터셋: train={count['train']}, val={count['val']}")
    return count["train"], count["val"]


def create_yaml() -> Path:
    yaml_path = MIXED_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {MIXED_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 3\n"
        f"names: ['sinkhole', 'pipe', 'rebar']\n"
    )
    return yaml_path


# ──────────────────────────────────────────────
# 2. Fine-tuning
# ──────────────────────────────────────────────

def finetune(yaml_path: Path, base_weights: Path,
             epochs: int = 40, batch: int = 4) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(base_weights))
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=416,
        lr0=3e-5,       # 매우 낮은 lr (Guangzhou 분포 적응)
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        freeze=5,
        dropout=0.1,
        patience=15,
        warmup_epochs=2,
        project=str(FT_DIR),
        name="run",
        exist_ok=True,
        verbose=False,
        workers=0,
        cache=False,
        amp=False,
    )
    return FT_DIR / "run" / "weights" / "best.pt"


# ──────────────────────────────────────────────
# 3. Guangzhou 테스트 추론
# ──────────────────────────────────────────────

def load_gz_test(labeled: list[dict], n_test: int = 5) -> list[tuple[str, str, np.ndarray]]:
    """
    라벨링에 사용되지 않은 Guangzhou .dt 파일을 테스트용으로 로드.
    Returns: [(stem, class_name, rgb_array), ...]
    """
    labeled_sources = set(e["source"] for e in labeled)

    test_samples = []
    for cls_name, root_dir in [("pipe", GZ_PIPE_DIR), ("rebar", GZ_REBAR_DIR)]:
        candidates = [
            f for f in sorted(root_dir.rglob("*.dt"))
            if "ascii" not in f.parent.name.lower()
               and str(f) not in labeled_sources
        ]
        random.Random(99).shuffle(candidates)
        count = 0
        for dt_path in candidates:
            if count >= n_test:
                break
            try:
                data, _ = read_ids_dt(str(dt_path))
                if data is None or data.shape[1] < 10:
                    continue
                d = dc_removal(data)
                d = background_removal(d)
                d = bandpass_filter(d, DT_SEC, F_LOW_MHZ, F_HIGH_MHZ)
                d = gain_sec(d, tpow=1.0, alpha=0.0, dt=DT_SEC)
                mn, mx = np.percentile(d, [2, 98])
                norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
                gray = (norm * 255).astype(np.uint8)
                rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                rgb  = cv2.resize(rgb, (640, 640))
                test_samples.append((dt_path.stem, cls_name, rgb))
                count += 1
            except Exception:
                continue

    return test_samples


def evaluate_gz(original_weights: Path, ft_weights: Path,
                test_samples: list, conf: float = 0.10) -> dict:
    """원본 vs fine-tuned 모델 Guangzhou 테스트 추론 비교."""
    from ultralytics import YOLO
    import tempfile

    orig_model = YOLO(str(original_weights))
    ft_model   = YOLO(str(ft_weights))
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for stem, cls_name, rgb in test_samples:
            tmp_path = os.path.join(tmpdir, f"{stem}.png")
            cv2.imwrite(tmp_path, rgb)

            orig_dets, ft_dets = [], []
            for model, det_list in [(orig_model, orig_dets), (ft_model, ft_dets)]:
                preds = model.predict(tmp_path, conf=conf, verbose=False)
                boxes = preds[0].boxes
                if boxes is not None and len(boxes):
                    for cls_id, conf_val in zip(
                            boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                        det_list.append({
                            "cls": int(cls_id),
                            "cls_name": CLASS_NAMES[int(cls_id)],
                            "conf": float(conf_val),
                        })

            results[f"{cls_name}/{stem}"] = {
                "true_class": cls_name,
                "original":   orig_dets,
                "finetuned":  ft_dets,
            }

    return results


# ──────────────────────────────────────────────
# 4. 시각화
# ──────────────────────────────────────────────

def visualize(eval_results: dict, ft_weights: Path,
              n_train: int, n_val: int, labeled: list[dict],
              test_samples: list):
    CLASS_COLORS = {'sinkhole': '#e74c3c', 'pipe': '#3498db', 'rebar': '#2ecc71'}

    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    fig.suptitle('Phase E-1: Guangzhou 라벨 Fine-tuning 결과',
                 fontsize=16, color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor('#2a2a4a')
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_color('#555')

    # ─ 패널 1: 데이터셋 구성 ─
    ax1 = fig.add_subplot(gs[0, :2])
    from collections import Counter
    cls_count = Counter(e["class"] for e in labeled)
    labels_bar = list(cls_count.keys())
    vals_bar   = [cls_count[k] for k in labels_bar]
    colors_bar = [CLASS_COLORS.get(k, 'gray') for k in labels_bar]
    ax1.bar(labels_bar, vals_bar, color=colors_bar, alpha=0.8)
    ax1.set_ylabel('이미지 수', color='white')
    for i, v in enumerate(vals_bar):
        ax1.text(i, v + 0.1, str(v), ha='center', color='white', fontsize=10)
    style(ax1, f'라벨링 데이터 구성 (train={n_train}, val={n_val})')

    # ─ 패널 2: 탐지 수 비교 ─
    ax2 = fig.add_subplot(gs[0, 2:])
    src_keys   = list(eval_results.keys())[:8]  # 최대 8개 표시
    orig_cnts  = [len(eval_results[k]['original'])  for k in src_keys]
    ft_cnts    = [len(eval_results[k]['finetuned']) for k in src_keys]
    x2 = range(len(src_keys))
    w  = 0.35
    ax2.bar([i - w/2 for i in x2], orig_cnts, w, label='D-2 원본', color='#3498db', alpha=0.8)
    ax2.bar([i + w/2 for i in x2], ft_cnts,   w, label='E-1 Fine-tuned', color='#e74c3c', alpha=0.8)
    short_keys = [k.split('/')[0] + '\n' + k.split('/')[1][:8] for k in src_keys]
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(short_keys, fontsize=7, color='white')
    ax2.set_ylabel('탐지 수 (conf≥0.10)', color='white')
    ax2.legend(facecolor='#2a2a4a', labelcolor='white', fontsize=8)
    style(ax2, 'D-2 원본 vs E-1 Fine-tuned 탐지 비교')

    # ─ 패널 3~6: 테스트 샘플 추론 시각화 ─
    from ultralytics import YOLO
    import tempfile
    ft_model = YOLO(str(ft_weights))

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (stem, cls_name, rgb) in enumerate(test_samples[:4]):
            tmp_path = os.path.join(tmpdir, f"{stem}.png")
            cv2.imwrite(tmp_path, rgb)
            preds = ft_model.predict(tmp_path, conf=0.10, verbose=False)
            boxes = preds[0].boxes

            row, col = divmod(idx, 2)
            ax = fig.add_subplot(gs[1, col * 2 : col * 2 + 2] if idx < 2
                                 else gs[2, col * 2 : col * 2 + 2])
            ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), aspect='auto')
            ax.set_title(f'[{cls_name}] {stem[:20]}\nE-1 Fine-tuned (conf≥0.10)',
                         color=CLASS_COLORS.get(cls_name, 'white'), fontsize=8)
            ax.axis('off')

            if boxes is not None and len(boxes):
                for box in boxes:
                    x1, y1, x2b, y2b = box.xyxy[0].cpu().numpy()
                    cid  = int(box.cls[0])
                    cval = float(box.conf[0])
                    cmap_color = [int(c * 255) for c in plt.cm.tab10(cid)[:3]]
                    rect = plt.Rectangle((x1, y1), x2b - x1, y2b - y1,
                                         fill=False,
                                         edgecolor=[c / 255 for c in cmap_color],
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 4,
                            f'{CLASS_NAMES[cid]} {cval:.2f}',
                            color='white', fontsize=7,
                            bbox=dict(facecolor=[c / 255 for c in cmap_color],
                                      alpha=0.7, pad=1))

    save_path = OUTPUT_DIR / "phase_e1_gz_finetune.png"
    plt.savefig(str(save_path), dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  시각화: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("Phase E-1: Guangzhou 라벨 Fine-tuning")
    print(f"  라벨 폴더: {GZ_LABELED}")
    print(f"  D-2 가중치: {D2_WEIGHTS}")

    # 가중치 존재 확인
    if not D2_WEIGHTS.exists():
        print(f"\n[경고] Phase D-2 가중치 없음: {D2_WEIGHTS}")
        # fallback: multiclass 원본 가중치
        fallback = RAG_DIR / "models" / "yolo_runs" / "multiclass_detect" / "weights" / "best.pt"
        if fallback.exists():
            print(f"  Fallback: {fallback}")
            base_weights = fallback
        else:
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {D2_WEIGHTS}")
    else:
        base_weights = D2_WEIGHTS

    # ─ 0. 라벨 확인 ─
    print("\n[0/5] 라벨링 상태 확인...")
    labeled = check_labels()

    # ─ 1. 혼합 데이터셋 ─
    print("\n[1/5] 혼합 데이터셋 구성...")
    n_train, n_val = build_mixed_dataset(labeled)
    yaml_path = create_yaml()
    print(f"  dataset.yaml: {yaml_path}")

    # ─ 2. Fine-tuning ─
    print("\n[2/5] Fine-tuning (epochs=40, batch=4)...")
    t0 = time.time()
    ft_weights = finetune(yaml_path, base_weights, epochs=40, batch=4)
    elapsed = time.time() - t0
    print(f"  Fine-tuning 완료: {elapsed:.0f}초")
    print(f"  Best weights: {ft_weights}")

    # 메트릭 출력
    results_csv = FT_DIR / "run" / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            print(f"  최종 mAP50:    {float(last.get('metrics/mAP50(B)', 0)):.4f}")
            print(f"  최종 mAP50-95: {float(last.get('metrics/mAP50-95(B)', 0)):.4f}")

    # ─ 3. 테스트셋 로드 ─
    print("\n[3/5] Guangzhou 테스트 샘플 로드 (라벨링 외 파일)...")
    test_samples = load_gz_test(labeled, n_test=5)
    print(f"  테스트 샘플: {len(test_samples)}개")

    # ─ 4. 평가 ─
    print("\n[4/5] 추론 비교 (D-2 원본 vs E-1 fine-tuned, conf=0.10)...")
    eval_results = evaluate_gz(base_weights, ft_weights, test_samples, conf=0.10)

    # ─ 결과 요약 ─
    print(f"\n{'='*60}")
    print("Phase E-1 결과 요약")
    print(f"{'='*60}")
    print(f"\n[라벨링 데이터] {len(labeled)}개 (train≈{n_train}, val≈{n_val})")

    print("\n[탐지 비교 (conf=0.10)]")
    print(f"  {'파일':<25} {'참조':>6} {'D-2':>6} {'E-1':>6} {'변화':>6}")
    print("  " + "-" * 55)
    for key, res in eval_results.items():
        cls_name  = res["true_class"]
        orig_n    = len(res["original"])
        ft_n      = len(res["finetuned"])
        delta     = ft_n - orig_n
        sign      = '+' if delta >= 0 else ''
        short_key = key[:24]
        print(f"  {short_key:<25} {cls_name:>6} {orig_n:>6} {ft_n:>6}  {sign}{delta:>4}")

    print("\n해석:")
    print("  - pipe/rebar 탐지 증가 → 실측 적응 성공")
    print("  - 탐지 없음 지속 → 도메인 갭 여전히 큼")

    # ─ 5. 시각화 ─
    print("\n[5/5] 시각화...")
    if test_samples:
        visualize(eval_results, ft_weights, n_train, n_val, labeled, test_samples)
    else:
        print("  테스트 샘플 없음, 시각화 스킵")

    print(f"\n{'='*60}")
    print("Phase E-1 완료")
    print(f"  Fine-tuned weights: {ft_weights}")
    print(f"  시각화: {OUTPUT_DIR / 'phase_e1_gz_finetune.png'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
