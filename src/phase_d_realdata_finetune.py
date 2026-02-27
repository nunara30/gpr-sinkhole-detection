"""
Phase D-1: 실측 GPR 데이터셋 Fine-tuning

Mendeley GPR 데이터셋 (2239개 실측 B-scan JPEG, YOLO 라벨 포함):
  - augmented_cavities (553): void/cavity → class 0 (sinkhole)
  - augmented_utilities (786): pipe/utilities → class 1 (pipe)  [class 0→1 리매핑]
  - augmented_intact (900): background → 빈 라벨 (negative)

기존 합성 데이터와 혼합하여 fine-tuning → 실측 탐지 성능 향상 목표.
"""

import os
import sys
import shutil
import random
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

from week1_gpr_basics import read_dt1, read_ids_dt

# ── 경로 ──
BASE_DIR   = Path("G:/RAG_system")
DATA_DIR   = BASE_DIR / "data" / "gpr"
MENDELEY   = DATA_DIR / "mendeley_gpr" / "GPR_data"
MC_DIR     = DATA_DIR / "yolo_multiclass"          # 기존 합성 데이터셋
MIXED_DIR  = DATA_DIR / "yolo_mixed_real"          # 혼합 데이터셋
MC_WEIGHTS = BASE_DIR / "models/yolo_runs/multiclass_detect/weights/best.pt"
FT_DIR     = BASE_DIR / "models/yolo_runs/finetune_real"
OUTPUT_DIR = BASE_DIR / "src" / "output" / "week4_multiclass"

GZ_PIPE_DIR   = DATA_DIR / "guangzhou" / "Data Set" / "pipe"
GZ_REBAR_DIR  = DATA_DIR / "guangzhou" / "Data Set" / "rebar"
GZ_TUNNEL_DIR = DATA_DIR / "guangzhou" / "Data Set" / "tunnel"

for d in [MIXED_DIR / "images/train", MIXED_DIR / "images/val",
          MIXED_DIR / "labels/train", MIXED_DIR / "labels/val",
          OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. Mendeley 데이터셋 로드 + 클래스 리매핑
# ──────────────────────────────────────────────

def load_mendeley(split_ratio: float = 0.8, seed: int = 42):
    """
    Mendeley GPR 데이터셋을 로드하고 클래스 ID 리매핑:
      cavity (0→0 = sinkhole), utility (0→1 = pipe), intact (빈 라벨)
    Returns: {'train': [(img_path, label_str)], 'val': [...]}
    """
    rng = random.Random(seed)
    all_items = []  # (img_path, label_str, split_key)

    # 라벨 파일 경로 탐색 헬퍼
    def find_label(img_path: Path, folder_name: str) -> Path | None:
        # 1순위: 이미지 옆에 .txt
        candidate = img_path.with_suffix(".txt")
        if candidate.exists():
            return candidate
        # 2순위: annotations/ 하위 폴더 (Yolo_format or YOLO_format)
        ann_dir = MENDELEY / folder_name / "annotations"
        if ann_dir.exists():
            for sub in ann_dir.iterdir():
                if sub.is_dir() and "yolo" in sub.name.lower():
                    candidate2 = sub / img_path.name.replace(".jpg", ".txt")
                    if candidate2.exists():
                        return candidate2
        return None

    # 1) cavity → class 0 (sinkhole)
    cav_imgs = sorted((MENDELEY / "augmented_cavities").glob("*.jpg"))
    for img in cav_imgs:
        txt = find_label(img, "augmented_cavities")
        if txt:
            lines = txt.read_text(errors='replace').strip().splitlines()
            label = "\n".join(l.strip() for l in lines if l.strip())
            all_items.append(("cavity", img, label))

    # 2) utility → class 1 (pipe): "0 ..." → "1 ..."
    util_imgs = sorted((MENDELEY / "augmented_utilities").glob("*.jpg"))
    for img in util_imgs:
        txt = find_label(img, "augmented_utilities")
        if txt:
            lines = txt.read_text(errors='replace').strip().splitlines()
            relabeled = []
            for l in lines:
                l = l.strip()
                if l:
                    parts = l.split()
                    parts[0] = "1"  # 0 → 1 (pipe)
                    relabeled.append(" ".join(parts))
            label = "\n".join(relabeled)
            all_items.append(("utility", img, label))

    # 3) intact → 빈 라벨 (background)
    intact_imgs = sorted((MENDELEY / "augmented_intact").rglob("*.jpg"))
    for img in intact_imgs:
        all_items.append(("intact", img, ""))

    # 클래스별 stratified split
    from collections import defaultdict
    by_class = defaultdict(list)
    for item in all_items:
        by_class[item[0]].append(item)

    train_items, val_items = [], []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_train = int(len(items) * split_ratio)
        train_items.extend(items[:n_train])
        val_items.extend(items[n_train:])

    print(f"  Mendeley 로드: train={len(train_items)}, val={len(val_items)}")
    for cls in ['cavity', 'utility', 'intact']:
        n = sum(1 for x in all_items if x[0] == cls)
        print(f"    {cls}: {n}개")

    return train_items, val_items


# ──────────────────────────────────────────────
# 2. 혼합 데이터셋 구성
# ──────────────────────────────────────────────

def build_mixed_dataset(
    mendeley_train, mendeley_val,
    n_synth_train: int = 300,
    n_synth_val:   int = 75,
    seed: int = 42
):
    """
    Mendeley 실측 + 기존 합성 데이터 혼합.
    Returns: (n_train, n_val)
    """
    rng = random.Random(seed)

    # 기존 합성 train/val 이미지 수집
    synth_train_imgs = sorted((MC_DIR / "images" / "train").glob("*.png"))
    synth_val_imgs   = sorted((MC_DIR / "images" / "val").glob("*.png"))
    rng.shuffle(synth_train_imgs)
    rng.shuffle(synth_val_imgs)
    synth_train_imgs = synth_train_imgs[:n_synth_train]
    synth_val_imgs   = synth_val_imgs[:n_synth_val]

    count = {"train": 0, "val": 0}

    def copy_synth(img_paths, split):
        for img_path in img_paths:
            lbl_path = MC_DIR / "labels" / split / img_path.stem
            lbl_path = lbl_path.with_suffix(".txt")
            stem = f"synth_{img_path.stem}"
            dst_img = MIXED_DIR / "images" / split / f"{stem}.png"
            dst_lbl = MIXED_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("")
            count[split] += 1

    def copy_mendeley(items, split):
        for i, (cls, img_path, label_str) in enumerate(items):
            stem = f"mendeley_{cls}_{img_path.stem}_{i}"
            dst_img = MIXED_DIR / "images" / split / f"{stem}.jpg"
            dst_lbl = MIXED_DIR / "labels" / split / f"{stem}.txt"
            shutil.copy2(img_path, dst_img)
            dst_lbl.write_text(label_str)
            count[split] += 1

    copy_synth(synth_train_imgs, "train")
    copy_synth(synth_val_imgs,   "val")
    copy_mendeley(mendeley_train, "train")
    copy_mendeley(mendeley_val,   "val")

    print(f"  혼합 데이터셋: train={count['train']}, val={count['val']}")
    return count["train"], count["val"]


def create_mixed_yaml() -> Path:
    yaml_path = MIXED_DIR / "dataset.yaml"
    yaml_path.write_text(f"""path: {MIXED_DIR.as_posix()}
train: images/train
val:   images/val
nc: 3
names: ['sinkhole', 'pipe', 'rebar']
""")
    return yaml_path


# ──────────────────────────────────────────────
# 3. Fine-tuning
# ──────────────────────────────────────────────

def finetune_real(yaml_path: Path, base_weights: Path,
                  epochs: int = 50, batch: int = 8):
    from ultralytics import YOLO
    model = YOLO(str(base_weights))
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=416,          # 416으로 축소 (mosaic 1280→832 메모리 절약)
        lr0=5e-5,           # 매우 낮은 lr (합성 지식 보존)
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        freeze=5,           # 하위 5레이어 동결 (backbone 앞부분)
        dropout=0.1,
        patience=20,
        warmup_epochs=3,
        project=str(FT_DIR),
        name="run",
        exist_ok=True,
        verbose=False,
        workers=0,
        cache=False,        # VRAM 캐시 비활성화
        amp=False,          # AMP 비활성화 (안정성)
    )
    best = FT_DIR / "run" / "weights" / "best.pt"
    return best


# ──────────────────────────────────────────────
# 4. 평가: 실측 Guangzhou + 합성 val
# ──────────────────────────────────────────────

def _load_gz_sample(folder: Path, n: int = 5):
    """Guangzhou IDS .dt 파일 로드 → PNG 변환."""
    from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec
    files = sorted(folder.glob("*.dt"))[:n]
    samples = []
    for fpath in files:
        try:
            data, dt_ns = read_ids_dt(str(fpath))
            if data is None or data.shape[1] < 10:
                continue
            dt_sec = dt_ns * 1e-9
            d = dc_removal(data)
            d = background_removal(d)
            d = bandpass_filter(d, dt_sec, 500.0, 4000.0)
            d = gain_sec(d, tpow=1.0, alpha=0.0, dt=dt_sec)
            # PNG 변환 (640×640)
            mn, mx = np.percentile(d, [2, 98])
            norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
            gray = (norm * 255).astype(np.uint8)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            rgb = cv2.resize(rgb, (640, 640))
            samples.append((fpath.stem, rgb))
        except Exception:
            continue
    return samples


def evaluate_real(original_weights: Path, ft_weights: Path,
                  conf: float = 0.10):
    """원본 vs fine-tuned 모델로 Guangzhou 실측 데이터 추론 비교."""
    from ultralytics import YOLO
    import tempfile

    original = YOLO(str(original_weights))
    finetuned = YOLO(str(ft_weights))

    CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']
    results = {}

    sources = [
        ("GZ_pipe",   GZ_PIPE_DIR,   1),  # expected: pipe
        ("GZ_rebar",  GZ_REBAR_DIR,  2),  # expected: rebar
        ("GZ_tunnel", GZ_TUNNEL_DIR, -1), # expected: none
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, folder, exp_cls in sources:
            samples = _load_gz_sample(folder, n=5)
            orig_dets, ft_dets = [], []

            for stem, rgb in samples:
                tmp_path = os.path.join(tmpdir, f"{stem}.png")
                cv2.imwrite(tmp_path, rgb)

                for model, det_list in [(original, orig_dets),
                                         (finetuned, ft_dets)]:
                    preds = model.predict(tmp_path, conf=conf, verbose=False)
                    boxes = preds[0].boxes
                    if boxes is not None and len(boxes):
                        for cls_id, conf_val in zip(
                                boxes.cls.cpu().numpy(),
                                boxes.conf.cpu().numpy()):
                            det_list.append({
                                "file": stem,
                                "cls": int(cls_id),
                                "cls_name": CLASS_NAMES[int(cls_id)],
                                "conf": float(conf_val),
                            })

            results[name] = {
                "expected_cls": exp_cls,
                "n_samples": len(samples),
                "original": orig_dets,
                "finetuned": ft_dets,
            }
            print(f"  [{name}] orig={len(orig_dets)} dets  "
                  f"ft={len(ft_dets)} dets  (expected cls={exp_cls})")

    return results


# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────

def visualize_results(eval_results: dict, ft_weights: Path,
                      mendeley_train, mendeley_val):
    """Phase D 결과 시각화."""
    from ultralytics import YOLO
    import tempfile

    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    fig.suptitle('Phase D: 실측 데이터 Fine-tuning 결과',
                 fontsize=16, color='white', fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']
    COLORS = ['#e74c3c', '#3498db', '#2ecc71']

    # ─ 패널 1: 데이터셋 구성 ─
    ax1 = fig.add_subplot(gs[0, :2])
    labels = ['Mendeley\ncavity→sinkhole', 'Mendeley\nutility→pipe',
              'Mendeley\nintact(BG)', 'Synthetic\ntrain']
    n_tr = [sum(1 for x in mendeley_train if x[0] == 'cavity'),
            sum(1 for x in mendeley_train if x[0] == 'utility'),
            sum(1 for x in mendeley_train if x[0] == 'intact'), 300]
    n_val = [sum(1 for x in mendeley_val if x[0] == 'cavity'),
             sum(1 for x in mendeley_val if x[0] == 'utility'),
             sum(1 for x in mendeley_val if x[0] == 'intact'), 75]
    x = np.arange(len(labels))
    w = 0.35
    ax1.bar(x - w/2, n_tr,  w, label='train', color='#3498db', alpha=0.8)
    ax1.bar(x + w/2, n_val, w, label='val',   color='#e74c3c', alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8, color='white')
    ax1.set_ylabel('이미지 수', color='white'); ax1.set_title('혼합 데이터셋 구성', color='white')
    ax1.legend(facecolor='#2a2a4a', labelcolor='white')
    ax1.set_facecolor('#2a2a4a'); ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_color('#555')

    # ─ 패널 2: 탐지 수 비교 ─
    ax2 = fig.add_subplot(gs[0, 2:])
    src_names = list(eval_results.keys())
    orig_counts = [len(eval_results[s]['original']) for s in src_names]
    ft_counts   = [len(eval_results[s]['finetuned']) for s in src_names]
    x2 = np.arange(len(src_names))
    ax2.bar(x2 - w/2, orig_counts, w, label='Original',   color='#3498db', alpha=0.8)
    ax2.bar(x2 + w/2, ft_counts,   w, label='Fine-tuned', color='#e74c3c', alpha=0.8)
    ax2.set_xticks(x2); ax2.set_xticklabels(src_names, color='white')
    ax2.set_ylabel('탐지 수 (conf≥0.10)', color='white')
    ax2.set_title('원본 vs Fine-tuned 탐지 비교', color='white')
    ax2.legend(facecolor='#2a2a4a', labelcolor='white')
    ax2.set_facecolor('#2a2a4a'); ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_color('#555')

    # ─ 패널 3~6: Fine-tuned 모델 실측 추론 결과 ─
    model_ft = YOLO(str(ft_weights))
    sample_sources = [
        ("GZ_pipe",   GZ_PIPE_DIR,   "GZ Pipe (기대: pipe)"),
        ("GZ_rebar",  GZ_REBAR_DIR,  "GZ Rebar (기대: rebar)"),
        ("GZ_tunnel", GZ_TUNNEL_DIR, "GZ Tunnel (기대: 없음)"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (name, folder, title) in enumerate(sample_sources):
            samples = _load_gz_sample(folder, n=1)
            if not samples:
                continue
            stem, rgb = samples[0]
            tmp_path = os.path.join(tmpdir, f"{stem}.png")
            cv2.imwrite(tmp_path, rgb)

            preds = model_ft.predict(tmp_path, conf=0.10, verbose=False)
            boxes = preds[0].boxes

            ax = fig.add_subplot(gs[1, idx])
            ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), aspect='auto')
            ax.set_title(f'{title}\n(Fine-tuned, conf≥0.10)', color='white', fontsize=8)
            ax.axis('off')

            if boxes is not None and len(boxes):
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    color_rgb = [int(c * 255) for c in
                                 plt.cm.get_cmap('tab10')(cls_id)[:3]]
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                        fill=False,
                                        edgecolor=[c/255 for c in color_rgb],
                                        linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x1, y1-4,
                            f'{CLASS_NAMES[cls_id]} {conf_val:.2f}',
                            color='white', fontsize=7,
                            bbox=dict(facecolor=[c/255 for c in color_rgb],
                                      alpha=0.7, pad=1))

    # ─ 패널 7: Mendeley 샘플 이미지 (cavity / utility) ─
    mendeley_imgs = []
    for cls, folder_name in [('cavity','augmented_cavities'),
                               ('utility','augmented_utilities'),
                               ('intact', 'augmented_intact')]:
        imgs = sorted((MENDELEY / folder_name).rglob('*.jpg'))
        if imgs:
            mendeley_imgs.append((cls, imgs[0]))

    for idx, (cls, img_path) in enumerate(mendeley_imgs[:3]):
        ax = fig.add_subplot(gs[2, idx])
        img = cv2.imread(str(img_path))
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 라벨 표시
        txt_path = img_path.with_suffix('.txt')
        cls_map = {'cavity': 0, 'utility': 1, 'intact': -1}
        mapped_cls = cls_map[cls]
        if txt_path.exists() and mapped_cls >= 0:
            lines = txt_path.read_text().strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    H, W = img.shape[:2]
                    x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
                    bw = w * W; bh = h * H
                    color = COLORS[mapped_cls]
                    rect = plt.Rectangle((x1, y1), bw, bh,
                                        fill=False, edgecolor=color, linewidth=2)
                    ax.add_patch(rect)
        cls_name = CLASS_NAMES[mapped_cls] if mapped_cls >= 0 else 'background'
        ax.set_title(f'Mendeley: {cls_name}', color='white', fontsize=9)
        ax.axis('off')

    # ─ 패널 8: 탐지 클래스 분포 ─
    ax8 = fig.add_subplot(gs[2, 3])
    all_ft_cls = []
    for src_data in eval_results.values():
        all_ft_cls.extend([d['cls_name'] for d in src_data['finetuned']])
    if all_ft_cls:
        from collections import Counter
        cls_counts = Counter(all_ft_cls)
        ax8.bar(cls_counts.keys(), cls_counts.values(),
                color=['#e74c3c','#3498db','#2ecc71'][:len(cls_counts)])
        ax8.set_title('Fine-tuned 탐지 클래스 분포', color='white', fontsize=9)
        ax8.set_ylabel('탐지 수', color='white')
        ax8.tick_params(colors='white')
    else:
        ax8.text(0.5, 0.5, '탐지 없음\n(도메인 갭 지속)',
                 ha='center', va='center', color='#e74c3c', fontsize=12)
        ax8.set_title('Fine-tuned 탐지 결과', color='white', fontsize=9)
    ax8.set_facecolor('#2a2a4a')
    for spine in ax8.spines.values(): spine.set_color('#555')

    save_path = OUTPUT_DIR / "phase_d_realdata_finetune.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  시각화: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    from ultralytics import YOLO
    import time

    print("Phase D-1: 실측 GPR Fine-tuning")
    print(f"  Mendeley: {MENDELEY}")
    print(f"  Base model: {MC_WEIGHTS}")

    # ─ 1. 데이터 로드 ─
    print("\n[1/5] Mendeley 데이터셋 로드...")
    mendeley_train, mendeley_val = load_mendeley(split_ratio=0.8)

    # ─ 2. 혼합 데이터셋 구성 ─
    print("\n[2/5] 혼합 데이터셋 구성 (Mendeley + 합성)...")
    # 기존 MIXED_DIR 초기화
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        d = MIXED_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 메모리 절약을 위해 클래스당 최대 200개로 서브샘플
    import random as _rnd
    rng2 = _rnd.Random(42)
    def subsample(items, max_per_cls=200):
        from collections import defaultdict
        by_cls = defaultdict(list)
        for item in items:
            by_cls[item[0]].append(item)
        result = []
        for cls, lst in by_cls.items():
            rng2.shuffle(lst)
            result.extend(lst[:max_per_cls])
        return result

    mendeley_train = subsample(mendeley_train, max_per_cls=200)
    mendeley_val   = subsample(mendeley_val,   max_per_cls=50)
    print(f"  서브샘플 후: train={len(mendeley_train)}, val={len(mendeley_val)}")

    n_train, n_val = build_mixed_dataset(
        mendeley_train, mendeley_val,
        n_synth_train=200, n_synth_val=50
    )
    yaml_path = create_mixed_yaml()
    print(f"  dataset.yaml: {yaml_path}")

    # ─ 3. Fine-tuning ─
    print("\n[3/5] Fine-tuning (epochs=50)...")
    t0 = time.time()
    ft_weights = finetune_real(yaml_path, MC_WEIGHTS, epochs=50, batch=4)
    elapsed = time.time() - t0
    print(f"  Fine-tuning 완료: {elapsed:.0f}초")
    print(f"  Best weights: {ft_weights}")

    # 최종 메트릭 확인
    results_csv = FT_DIR / "run" / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            print(f"  최종 epoch: {last.get('epoch', '?')}")
            print(f"  mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}")
            print(f"  mAP50-95: {float(last.get('metrics/mAP50-95(B)', 0)):.4f}")

    # ─ 4. 실측 평가 ─
    print("\n[4/5] 실측 Guangzhou 데이터 평가 (conf=0.10)...")
    eval_results = evaluate_real(MC_WEIGHTS, ft_weights, conf=0.10)

    # ─ 결과 요약 ─
    print("\n" + "=" * 65)
    print("Phase D-1 결과 요약")
    print("=" * 65)
    print(f"\n[데이터셋] train={n_train}, val={n_val}")
    print(f"  Mendeley 실측: {len(mendeley_train)+len(mendeley_val)}개")
    print(f"  합성 데이터: 375개")

    print("\n[탐지 비교 (conf=0.10, 5샘플 기준)]")
    print(f"  {'소스':<15} {'원본':>8} {'Fine-tuned':>12} {'변화':>8}")
    print("  " + "-" * 46)
    for src_name, res in eval_results.items():
        orig = len(res['original'])
        ft   = len(res['finetuned'])
        delta = ft - orig
        sign = '+' if delta >= 0 else ''
        exp = res['expected_cls']
        note = "(FP 위험)" if exp < 0 and ft > 0 else "(TP 증가)" if exp >= 0 and ft > orig else ""
        print(f"  {src_name:<15} {orig:>8} {ft:>12}  {sign}{delta:>4} {note}")

    print("\n해석:")
    print("  - pipe 탐지 증가 → 실측 학습 효과 (기대)")
    print("  - tunnel FP 억제 → 안정성 유지")
    print("  - rebar는 실측 학습 데이터 없으므로 변화 적을 수 있음")

    # ─ 5. 시각화 ─
    print("\n[5/5] 시각화...")
    visualize_results(eval_results, ft_weights, mendeley_train, mendeley_val)

    print(f"\n{'='*65}")
    print("Phase D-1 완료")
    print(f"  Fine-tuned weights: {ft_weights}")
    print(f"  시각화: {OUTPUT_DIR/'phase_d_realdata_finetune.png'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
