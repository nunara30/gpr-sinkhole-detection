"""
Week 4 - YOLOv11 싱크홀 자동 탐지
합성 B-scan → YOLO 데이터셋 변환 → 학습 → 평가 → 실제 데이터 추론

파이프라인:
  1. 48개 .npy → 640×640 grayscale PNG + YOLO bbox 라벨
  2. 위치 변경 증강 (×5) + 전처리 버전 (×2) → ~480개
  3. YOLOv11n 학습 (150 epochs, pretrained backbone)
  4. 검증 셋 평가 (mAP50 > 0.7 목표)
  5. 실제 GPR 데이터 추론 (Frenke, Guangzhou)
"""

import sys
import time
import json
import shutil
import itertools
import warnings
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Week 1-3 imports
sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_dt1, read_ids_dt
from week2_preprocessing import (
    dc_removal, dewow, background_removal, bandpass_filter,
    gain_sec, run_pipeline, estimate_dt,
)
from week2_database import GPRDatabase
from week3_simulation import (
    synthesize_bscan, SCENARIOS, SYNTHETIC_DIR,
    soil_velocity, C0, apply_preprocessing_to_synthetic,
)


# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("G:/RAG_system")
DATA_DIR = BASE_DIR / "data" / "gpr"
YOLO_DIR = DATA_DIR / "yolo_dataset"
YOLO_IMAGES_TRAIN = YOLO_DIR / "images" / "train"
YOLO_IMAGES_VAL = YOLO_DIR / "images" / "val"
YOLO_LABELS_TRAIN = YOLO_DIR / "labels" / "train"
YOLO_LABELS_VAL = YOLO_DIR / "labels" / "val"
YOLO_RUNS_DIR = BASE_DIR / "models" / "yolo_runs"
OUTPUT_DIR = BASE_DIR / "src" / "output" / "week4"

for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL,
          YOLO_LABELS_TRAIN, YOLO_LABELS_VAL,
          YOLO_RUNS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Step 1-1: B-scan → 640×640 Grayscale PNG
# ─────────────────────────────────────────────

def npy_to_png(bscan, output_path, clip_pct=98):
    """
    B-scan float32 배열 → 640×640 grayscale PNG

    - 98th percentile 대칭 클리핑
    - [0, 255] 선형 정규화
    - cv2.resize(INTER_LINEAR) → 640×640
    """
    # 대칭 클리핑
    vmax = np.percentile(np.abs(bscan), clip_pct)
    if vmax == 0:
        vmax = 1.0
    clipped = np.clip(bscan, -vmax, vmax)

    # [0, 255] 정규화
    normalized = ((clipped + vmax) / (2 * vmax) * 255).astype(np.uint8)

    # 640×640 리사이즈
    resized = cv2.resize(normalized, (640, 640), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(str(output_path), resized)
    return resized


# ─────────────────────────────────────────────
# Step 1-2: YOLO 바운딩 박스 자동 생성
# ─────────────────────────────────────────────

def compute_yolo_bbox(meta):
    """
    메타데이터에서 YOLO normalized bbox 계산

    쌍곡선 기하학:
      수직: sample_top = (depth - radius) → time → sample index
            sample_bottom = (depth + radius) → time → sample index
            + wavelet margin (±10% 여유)
      수평: x_half = sqrt(d_bottom² - d_top²), 최소 2×radius

    Returns: (class_id, cx, cy, w, h) normalized 0-1
             or None if bbox is invalid
    """
    depth_m = meta['depth_m']
    radius_m = meta['radius_m']
    soil_epsr = meta['soil_epsr']
    sinkhole_x = meta['sinkhole_x']
    domain_x = meta['domain_x']
    n_samples = meta['n_samples']
    n_traces = meta['n_traces']
    dt_ns = meta['dt_ns']

    v = soil_velocity(soil_epsr)   # m/ns
    v_m_s = v * 1e9               # m/s

    # 수직 범위 (시간 → 샘플)
    d_top = max(0.05, depth_m - radius_m)
    d_bottom = depth_m + radius_m

    t_top_s = 2 * d_top / v_m_s
    t_bottom_s = 2 * d_bottom / v_m_s
    dt_s = dt_ns * 1e-9

    sample_top = t_top_s / dt_s
    sample_bottom = t_bottom_s / dt_s

    # wavelet margin: ±15% of bbox height (Ricker wavelet 폭 보상)
    height_samples = sample_bottom - sample_top
    margin = height_samples * 0.15
    sample_top = max(0, sample_top - margin)
    sample_bottom = min(n_samples - 1, sample_bottom + margin)

    # 수평 범위 (쌍곡선 확산)
    # 하이퍼볼라 폭: x에서 top 반사와 bottom 반사 사이 시간차가 wavelet 폭 이내인 범위
    # 간단 근사: x_half = sqrt(d_bottom^2 - d_top^2), 최소 2*radius
    if d_bottom > d_top:
        x_half = np.sqrt(d_bottom**2 - d_top**2)
    else:
        x_half = 2 * radius_m
    x_half = max(x_half, 2 * radius_m)

    # 트레이스 인덱스로 변환
    dx = meta['dx']
    trace_center = sinkhole_x / dx
    trace_half = x_half / dx

    trace_left = max(0, trace_center - trace_half)
    trace_right = min(n_traces - 1, trace_center + trace_half)

    # YOLO normalized 좌표 (0-1)
    cx = (trace_left + trace_right) / 2 / n_traces
    cy = (sample_top + sample_bottom) / 2 / n_samples
    w = (trace_right - trace_left) / n_traces
    h = (sample_bottom - sample_top) / n_samples

    # 유효성 검사
    if w < 0.01 or h < 0.01 or w > 0.95 or h > 0.95:
        return None
    # 범위 클리핑
    cx = np.clip(cx, 0.0, 1.0)
    cy = np.clip(cy, 0.0, 1.0)
    w = np.clip(w, 0.01, 1.0)
    h = np.clip(h, 0.01, 1.0)

    return (0, cx, cy, w, h)


def write_yolo_label(label_path, bbox_tuple):
    """YOLO 라벨 파일 쓰기: '0 cx cy w h'"""
    if bbox_tuple is None:
        # 빈 라벨 (객체 없음)
        label_path.write_text("", encoding='utf-8')
        return
    class_id, cx, cy, w, h = bbox_tuple
    label_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n",
                          encoding='utf-8')


# ─────────────────────────────────────────────
# Step 1-3: 위치 변경 증강 (synthesize_bscan 재호출)
# ─────────────────────────────────────────────

def generate_shifted_bscan(meta, new_sinkhole_x):
    """
    sinkhole_x 위치를 변경하여 새 B-scan 합성

    Returns: (bscan, new_meta)
    """
    bscan, _, _, new_meta = synthesize_bscan(
        freq_hz=meta['freq_hz'],
        depth_m=meta['depth_m'],
        radius_m=meta['radius_m'],
        soil_epsr=meta['soil_epsr'],
        domain_x=meta['domain_x'],
        domain_z=meta.get('domain_z', 3.0),
        dx=meta['dx'],
        n_samples=meta['n_samples'],
        sinkhole_x=new_sinkhole_x,
        add_noise=True,
    )
    return bscan, new_meta


# ─────────────────────────────────────────────
# Step 1-4: 데이터셋 준비 (전체 파이프라인)
# ─────────────────────────────────────────────

def prepare_dataset(val_ratio=0.2, n_shift_variants=4, seed=42):
    """
    YOLO 데이터셋 전체 준비

    1. 48개 원본 .npy → PNG + 라벨
    2. 48개 전처리 적용 버전
    3. 48 × n_shift_variants 위치 변경 증강
    4. Train/Val 분할 (stratified by base scenario)

    Returns: summary dict
    """
    rng = np.random.RandomState(seed)

    # 기존 데이터 정리
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL,
              YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]:
        for f in d.glob("*"):
            f.unlink()

    # ── 모든 이미지/라벨 생성 (임시 리스트) ──
    all_items = []  # list of (scenario_key, img_name, bscan, meta, variant_type)

    # 48개 원본 메타데이터 로드
    meta_files = sorted(SYNTHETIC_DIR.glob("*_meta.json"))
    print(f"  메타데이터 파일: {len(meta_files)}개")

    base_scenarios = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        label = meta_path.stem.replace('_meta', '')
        npy_path = SYNTHETIC_DIR / f"{label}.npy"
        if not npy_path.exists():
            continue

        bscan = np.load(str(npy_path))
        scenario_key = (meta['freq_mhz'], meta['depth_m'],
                        meta['radius_m'], meta['soil_epsr'])
        base_scenarios.append(scenario_key)

        # (a) 원본
        all_items.append((scenario_key, f"{label}_raw", bscan, meta, 'raw'))

        # (b) 전처리 적용 버전
        try:
            processed, _, _ = apply_preprocessing_to_synthetic(bscan, meta)
            all_items.append((scenario_key, f"{label}_proc", processed, meta, 'proc'))
        except Exception as e:
            print(f"    전처리 실패 ({label}): {e}")

        # (c) 위치 변경 증강
        shift_positions = rng.uniform(0.8, 4.2, size=n_shift_variants)
        for si, new_x in enumerate(shift_positions):
            try:
                shifted_bscan, shifted_meta = generate_shifted_bscan(meta, new_x)
                all_items.append((scenario_key, f"{label}_shift{si}",
                                  shifted_bscan, shifted_meta, 'shift'))
            except Exception as e:
                print(f"    증강 실패 ({label} shift {si}): {e}")

    print(f"  총 이미지 수: {len(all_items)}")

    # ── Train/Val 분할 (stratified by scenario_key) ──
    # 동일 base scenario 의 모든 variant는 같은 split에 배치
    unique_scenarios = list(set(base_scenarios))
    rng.shuffle(unique_scenarios)
    n_val = max(1, int(len(unique_scenarios) * val_ratio))
    val_scenarios = set(map(tuple, unique_scenarios[:n_val]))

    train_count = 0
    val_count = 0
    skipped = 0

    for scenario_key, img_name, bscan, meta, variant_type in all_items:
        # bbox 계산
        bbox = compute_yolo_bbox(meta)
        if bbox is None:
            skipped += 1
            continue

        # Split 결정
        is_val = scenario_key in val_scenarios
        img_dir = YOLO_IMAGES_VAL if is_val else YOLO_IMAGES_TRAIN
        lbl_dir = YOLO_LABELS_VAL if is_val else YOLO_LABELS_TRAIN

        # PNG 저장
        png_path = img_dir / f"{img_name}.png"
        npy_to_png(bscan, png_path)

        # 라벨 저장
        lbl_path = lbl_dir / f"{img_name}.txt"
        write_yolo_label(lbl_path, bbox)

        if is_val:
            val_count += 1
        else:
            train_count += 1

    summary = {
        'total': train_count + val_count,
        'train': train_count,
        'val': val_count,
        'skipped': skipped,
        'n_base_scenarios': len(unique_scenarios),
        'n_val_scenarios': n_val,
    }

    print(f"  Train: {train_count}, Val: {val_count}, Skipped: {skipped}")
    return summary


# ─────────────────────────────────────────────
# Step 1-5: dataset.yaml 생성
# ─────────────────────────────────────────────

def create_dataset_yaml():
    """YOLO dataset.yaml 생성"""
    yaml_content = f"""path: {str(YOLO_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

nc: 1
names: ['sinkhole']
"""
    yaml_path = YOLO_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"  dataset.yaml: {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────
# 라벨 검증 시각화
# ─────────────────────────────────────────────

def verify_labels(n_samples=4, save_path=None):
    """
    라벨 검증: 이미지에 GT bbox 오버레이

    train 이미지에서 n_samples개 무작위 선택하여 시각화
    """
    train_pngs = sorted(YOLO_IMAGES_TRAIN.glob("*.png"))
    if not train_pngs:
        print("  검증할 이미지 없음")
        return

    indices = np.random.choice(len(train_pngs), min(n_samples, len(train_pngs)),
                               replace=False)
    selected = [train_pngs[i] for i in indices]

    fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 5))
    if len(selected) == 1:
        axes = [axes]

    for ax, png_path in zip(axes, selected):
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray', aspect='equal')

        # 라벨 파일 읽기
        lbl_path = YOLO_LABELS_TRAIN / f"{png_path.stem}.txt"
        if lbl_path.exists():
            text = lbl_path.read_text().strip()
            if text:
                parts = text.split()
                cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                # pixel 좌표로 변환
                img_h, img_w = img.shape
                px = (cx - w / 2) * img_w
                py = (cy - h / 2) * img_h
                pw = w * img_w
                ph = h * img_h
                rect = Rectangle((px, py), pw, ph,
                                 linewidth=2, edgecolor='red',
                                 facecolor='none', linestyle='--')
                ax.add_patch(rect)
                ax.set_title(f"{png_path.stem}\n"
                             f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}",
                             fontsize=7)
            else:
                ax.set_title(f"{png_path.stem}\n(no object)", fontsize=7)
        else:
            ax.set_title(f"{png_path.stem}\n(no label)", fontsize=7)

        ax.axis('off')

    fig.suptitle("Label Verification (GT bbox in red)", fontsize=12,
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Step 2: 모델 학습
# ─────────────────────────────────────────────

def train_yolo(dataset_yaml, model_name="yolo11n.pt", epochs=150,
               batch=8, patience=30):
    """
    YOLOv11n 학습

    - COCO pretrained backbone (transfer learning)
    - backbone freeze (첫 10 layers)
    - OOM 시 batch=4 로 재시도

    Returns: best.pt 경로
    """
    from ultralytics import YOLO

    # 모델 로드 (fallback: yolov8n)
    try:
        model = YOLO(model_name)
        print(f"  모델: {model_name}")
    except Exception:
        model_name = "yolov8n.pt"
        model = YOLO(model_name)
        print(f"  Fallback 모델: {model_name}")

    # 학습 설정
    train_kwargs = dict(
        data=str(dataset_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        patience=patience,
        optimizer='AdamW',
        lr0=0.001,
        cos_lr=True,
        dropout=0.1,
        freeze=10,
        # Augmentation
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        degrees=0.0,
        hsv_v=0.3,
        translate=0.15,
        scale=0.3,
        # Windows: workers=0 to avoid multiprocessing DLL issues
        workers=0,
        # Output
        project=str(YOLO_RUNS_DIR),
        name="sinkhole_detect",
        exist_ok=True,
        verbose=True,
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"  OOM 발생 → batch={batch//2}로 재시도")
            import torch
            torch.cuda.empty_cache()
            train_kwargs['batch'] = batch // 2
            results = model.train(**train_kwargs)
        else:
            raise

    best_path = YOLO_RUNS_DIR / "sinkhole_detect" / "weights" / "best.pt"
    if best_path.exists():
        print(f"  Best model: {best_path}")
    else:
        # 학습 결과 디렉토리에서 찾기
        for p in YOLO_RUNS_DIR.rglob("best.pt"):
            best_path = p
            break
        print(f"  Best model: {best_path}")

    return best_path


# ─────────────────────────────────────────────
# Step 3: 평가 + 시각화
# ─────────────────────────────────────────────

def evaluate_model(weights_path, dataset_yaml):
    """
    학습된 모델 평가

    Returns: metrics dict (mAP50, mAP50-95, precision, recall)
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.val(data=str(dataset_yaml), imgsz=640, verbose=True)

    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }

    # F1 계산
    p, r = metrics['precision'], metrics['recall']
    metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"\n  === 평가 결과 ===")
    print(f"  mAP50:     {metrics['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['mAP50_95']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")

    target = 0.7
    if metrics['mAP50'] >= target:
        print(f"  ✓ mAP50 목표 달성 ({metrics['mAP50']:.3f} >= {target})")
    else:
        print(f"  ✗ mAP50 목표 미달 ({metrics['mAP50']:.3f} < {target})")

    return metrics


def visualize_predictions(weights_path, n_samples=6, save_path=None):
    """
    검증 셋 예측 결과 시각화: GT(빨강) vs Pred(초록) 박스 오버레이
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    val_pngs = sorted(YOLO_IMAGES_VAL.glob("*.png"))
    if not val_pngs:
        print("  시각화할 검증 이미지 없음")
        return

    indices = np.random.choice(len(val_pngs), min(n_samples, len(val_pngs)),
                               replace=False)
    selected = [val_pngs[i] for i in indices]

    cols = min(3, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.array(axes).flatten()

    for idx, (ax, png_path) in enumerate(zip(axes, selected)):
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray', aspect='equal')
        img_h, img_w = img.shape[:2]

        # GT bbox (빨강)
        lbl_path = YOLO_LABELS_VAL / f"{png_path.stem}.txt"
        if lbl_path.exists():
            text = lbl_path.read_text().strip()
            if text:
                parts = text.split()
                _, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                px = (cx - w / 2) * img_w
                py = (cy - h / 2) * img_h
                rect = Rectangle((px, py), w * img_w, h * img_h,
                                 linewidth=2, edgecolor='red',
                                 facecolor='none', label='GT')
                ax.add_patch(rect)

        # Prediction (초록)
        results = model.predict(str(png_path), imgsz=640, conf=0.25,
                                verbose=False)
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime',
                                 facecolor='none', label=f'Pred {conf:.2f}')
                ax.add_patch(rect)

        ax.set_title(png_path.stem, fontsize=7)
        ax.axis('off')

    # 빈 axes 숨기기
    for i in range(len(selected), len(axes)):
        axes[i].set_visible(False)

    # 범례 (한 번만)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='GT'),
        Line2D([0], [0], color='lime', linewidth=2, label='Predicted'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    fig.suptitle("Validation Predictions: GT (red) vs Predicted (green)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


def analyze_performance_by_param(weights_path, save_path=None):
    """
    파라미터별 탐지 성능 분석 (깊이, 반경, 주파수별)
    검증 셋에서 각 파라미터별 confidence 분포를 분석
    """
    from ultralytics import YOLO

    model = YOLO(str(weights_path))

    val_pngs = sorted(YOLO_IMAGES_VAL.glob("*.png"))
    if not val_pngs:
        return

    # 파라미터 추출 및 예측
    records = []
    for png_path in val_pngs:
        name = png_path.stem
        # 파일명에서 파라미터 파싱: f{freq}_d{depth}_r{radius}_er{epsr}_{variant}
        parts = name.split('_')
        try:
            freq = float(parts[0][1:])   # f400 → 400
            depth = float(parts[1][1:])  # d1.0 → 1.0
            radius = float(parts[2][1:]) # r0.5 → 0.5
            epsr = float(parts[3][2:])   # er6 → 6
        except (IndexError, ValueError):
            continue

        results = model.predict(str(png_path), imgsz=640, conf=0.1,
                                verbose=False)
        max_conf = 0.0
        if results and len(results[0].boxes):
            confs = results[0].boxes.conf.cpu().numpy()
            max_conf = float(confs.max())

        records.append({
            'freq': freq, 'depth': depth, 'radius': radius,
            'epsr': epsr, 'conf': max_conf, 'detected': max_conf >= 0.25,
        })

    if not records:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Depth별
    ax = axes[0]
    depths = sorted(set(r['depth'] for r in records))
    for d in depths:
        confs = [r['conf'] for r in records if r['depth'] == d]
        ax.boxplot(confs, positions=[d], widths=0.3)
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('Confidence')
    ax.set_title('Detection vs Depth')
    ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.5, label='conf=0.25')
    ax.legend(fontsize=8)

    # Radius별
    ax = axes[1]
    radii = sorted(set(r['radius'] for r in records))
    for r_val in radii:
        confs = [r['conf'] for r in records if r['radius'] == r_val]
        ax.boxplot(confs, positions=[r_val], widths=0.15)
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Confidence')
    ax.set_title('Detection vs Radius')
    ax.axhline(y=0.25, color='r', linestyle='--', alpha=0.5)

    # Frequency별
    ax = axes[2]
    freqs = sorted(set(r['freq'] for r in records))
    for fi, fq in enumerate(freqs):
        confs = [r['conf'] for r in records if r['freq'] == fq]
        det_rate = sum(1 for c in confs if c >= 0.25) / max(len(confs), 1)
        ax.bar(fi, det_rate, label=f'{fq:.0f}MHz')
        ax.text(fi, det_rate + 0.02, f'{det_rate:.0%}', ha='center', fontsize=9)
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate vs Frequency')
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels([f'{f:.0f}MHz' for f in freqs])
    ax.set_ylim(0, 1.1)

    fig.suptitle('Performance Analysis by Parameter', fontsize=13,
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Step 4: 실제 데이터 추론
# ─────────────────────────────────────────────

def _prepare_real_image(data, clip_pct=98):
    """실제 GPR 데이터 → 640×640 grayscale numpy (YOLO 입력용)"""
    vmax = np.percentile(np.abs(data), clip_pct)
    if vmax == 0:
        vmax = 1.0
    clipped = np.clip(data, -vmax, vmax)
    normalized = ((clipped + vmax) / (2 * vmax) * 255).astype(np.uint8)
    resized = cv2.resize(normalized, (640, 640), interpolation=cv2.INTER_LINEAR)
    return resized


def inference_on_real_data(weights_path, output_dir=None):
    """
    실제 GPR 데이터에 대한 zero-shot 추론 (도메인 갭 실험)

    - Frenke LINE00 (100MHz DT1, 하천 퇴적층 → 싱크홀 없음 예상)
    - Guangzhou rebar (2GHz IDS .dt, 철근 하이퍼볼라 → 오탐 가능)

    각 데이터에 대해 raw / preprocessed 두 버전 테스트
    """
    from ultralytics import YOLO

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))

    results_summary = []

    # ── Frenke LINE00 ──
    frenke_path = str(DATA_DIR / "frenke/2014_04_25_frenke/rawGPR/LINE00.DT1")
    if Path(frenke_path).exists():
        print(f"\n  [Frenke LINE00] 추론...")
        data_frenke, header_frenke = read_dt1(frenke_path)

        # raw 버전
        img_raw = _prepare_real_image(data_frenke)
        tmp_raw = output_dir / "_tmp_frenke_raw.png"
        cv2.imwrite(str(tmp_raw), img_raw)
        pred_raw = model.predict(str(tmp_raw), imgsz=640, conf=0.1, verbose=False)

        # preprocessed 버전
        tw_frenke = float(header_frenke.get('TOTAL TIME WINDOW', 50.0))
        dt_frenke = estimate_dt(tw_frenke, data_frenke.shape[0])
        pipeline = [
            ('DC_Removal', dc_removal, {}),
            ('Dewow', dewow, {'window': 50}),
            ('Background_Removal', background_removal, {}),
            ('Bandpass', bandpass_filter,
             {'dt': dt_frenke, 'low_mhz': 25, 'high_mhz': 250}),
            ('Gain_SEC', gain_sec,
             {'tpow': 1.5, 'alpha': 0.2, 'dt': dt_frenke}),
        ]
        processed_frenke, _, _ = run_pipeline(
            data_frenke, dt_frenke, 0.25, pipeline)
        img_proc = _prepare_real_image(processed_frenke)
        tmp_proc = output_dir / "_tmp_frenke_proc.png"
        cv2.imwrite(str(tmp_proc), img_proc)
        pred_proc = model.predict(str(tmp_proc), imgsz=640, conf=0.1,
                                  verbose=False)

        # 시각화
        _plot_real_inference(
            img_raw, pred_raw, img_proc, pred_proc,
            title="Frenke LINE00 (100MHz, river sediment)",
            expected="No sinkhole expected",
            save_path=output_dir / "real_frenke.png",
        )
        n_raw = len(pred_raw[0].boxes) if pred_raw else 0
        n_proc = len(pred_proc[0].boxes) if pred_proc else 0
        results_summary.append({
            'dataset': 'Frenke LINE00',
            'raw_detections': n_raw,
            'proc_detections': n_proc,
        })
        # 임시 파일 정리
        tmp_raw.unlink(missing_ok=True)
        tmp_proc.unlink(missing_ok=True)
    else:
        print(f"  [Frenke] 파일 없음: {frenke_path}")

    # ── Guangzhou rebar ──
    gz_dt_files = sorted(
        f for f in (DATA_DIR / "guangzhou/Data Set/rebar").rglob("*.dt")
        if 'ASCII' not in str(f)
    ) if (DATA_DIR / "guangzhou/Data Set/rebar").exists() else []

    if gz_dt_files:
        gz_path = str(gz_dt_files[0])
        print(f"\n  [Guangzhou rebar] 추론...")
        print(f"    파일: {gz_path}")
        data_gz, header_gz = read_ids_dt(gz_path)

        if data_gz is not None:
            # raw 버전
            img_raw = _prepare_real_image(data_gz)
            tmp_raw = output_dir / "_tmp_gz_raw.png"
            cv2.imwrite(str(tmp_raw), img_raw)
            pred_raw = model.predict(str(tmp_raw), imgsz=640, conf=0.1,
                                     verbose=False)

            # preprocessed 버전
            sweep_str = header_gz.get('sweep_time',
                        header_gz.get('SweepTime', ''))
            try:
                tw_gz = float(sweep_str) * 1e9 if float(sweep_str) < 1 else float(sweep_str)
            except (ValueError, TypeError):
                tw_gz = 25.0
            dt_gz = estimate_dt(tw_gz, data_gz.shape[0])

            pipeline_gz = [
                ('DC_Removal', dc_removal, {}),
                ('Dewow', dewow, {'window': 30}),
                ('Background_Removal', background_removal, {}),
                ('Bandpass', bandpass_filter,
                 {'dt': dt_gz, 'low_mhz': 500, 'high_mhz': 5000}),
                ('Gain_SEC', gain_sec,
                 {'tpow': 1.0, 'alpha': 0.1, 'dt': dt_gz}),
            ]
            processed_gz, _, _ = run_pipeline(
                data_gz, dt_gz, 0.01, pipeline_gz)
            img_proc = _prepare_real_image(processed_gz)
            tmp_proc = output_dir / "_tmp_gz_proc.png"
            cv2.imwrite(str(tmp_proc), img_proc)
            pred_proc = model.predict(str(tmp_proc), imgsz=640, conf=0.1,
                                      verbose=False)

            _plot_real_inference(
                img_raw, pred_raw, img_proc, pred_proc,
                title="Guangzhou Rebar (2GHz, tunnel survey)",
                expected="Rebar hyperbolas — false positives possible",
                save_path=output_dir / "real_guangzhou.png",
            )
            n_raw = len(pred_raw[0].boxes) if pred_raw else 0
            n_proc = len(pred_proc[0].boxes) if pred_proc else 0
            results_summary.append({
                'dataset': 'Guangzhou rebar',
                'raw_detections': n_raw,
                'proc_detections': n_proc,
            })
            tmp_raw.unlink(missing_ok=True)
            tmp_proc.unlink(missing_ok=True)
        else:
            print(f"    Guangzhou 파싱 실패")
    else:
        print(f"  [Guangzhou] .dt 파일 없음")

    return results_summary


def _plot_real_inference(img_raw, pred_raw, img_proc, pred_proc,
                         title="", expected="", save_path=None):
    """실제 데이터 추론 결과 2×1 시각화 (raw / processed)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, img, pred, label in [
        (ax1, img_raw, pred_raw, 'Raw'),
        (ax2, img_proc, pred_proc, 'Preprocessed'),
    ]:
        ax.imshow(img, cmap='gray', aspect='equal')
        n_det = 0
        if pred and len(pred[0].boxes):
            for box in pred[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='lime',
                                 facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 5, f'{conf:.2f}', color='lime',
                        fontsize=9, fontweight='bold')
                n_det += 1
        ax.set_title(f'{label} ({n_det} detections)', fontsize=11)
        ax.axis('off')

    fig.suptitle(f'{title}\nExpected: {expected}', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────
# Step 5: DB 기록
# ─────────────────────────────────────────────

def log_to_database(db, dataset_summary, metrics, real_results):
    """Week 4 결과를 DB에 기록"""
    # YOLO 데이터셋 등록
    dummy_data = np.zeros((640, 640), dtype=np.float32)
    ds_id = db.register_dataset(
        name=f"YOLO Sinkhole Dataset (n={dataset_summary['total']})",
        file_path=str(YOLO_DIR / "dataset.yaml"),
        data=dummy_data,
        format="YOLO_dataset",
        frequency_mhz=0,   # 혼합
        time_window_ns=0,
        dx_m=0,
    )

    # 학습 결과를 processing run으로 기록
    steps = [
        {
            'step_name': 'Dataset_Preparation',
            'parameters': {
                'train': dataset_summary['train'],
                'val': dataset_summary['val'],
                'total': dataset_summary['total'],
            },
            'elapsed_ms': 0,
        },
        {
            'step_name': 'YOLO_Training',
            'parameters': {
                'model': 'yolo11n',
                'epochs': 150,
                'batch': 8,
            },
            'elapsed_ms': 0,
        },
        {
            'step_name': 'Evaluation',
            'parameters': metrics,
            'elapsed_ms': 0,
        },
    ]

    if real_results:
        steps.append({
            'step_name': 'Real_Data_Inference',
            'parameters': {r['dataset']: f"raw={r['raw_detections']}, proc={r['proc_detections']}"
                           for r in real_results},
            'elapsed_ms': 0,
        })

    db.log_processing_run(ds_id, "Week4 YOLO Sinkhole Detection", steps)
    return ds_id


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("=" * 60)
    print("  Week 4 - YOLOv11 Sinkhole Detection")
    print("=" * 60)

    t_start = time.perf_counter()
    db = GPRDatabase()

    # ── Step 1: 데이터 준비 ──
    print("\n[1] 데이터셋 준비...")
    print("  1-1. B-scan → PNG + YOLO 라벨 변환")
    print("  1-2. 위치 변경 증강 (×5)")
    print("  1-3. 전처리 버전 (×2)")

    dataset_summary = prepare_dataset(val_ratio=0.2, n_shift_variants=4)

    print(f"\n  1-4. dataset.yaml 생성")
    yaml_path = create_dataset_yaml()

    print(f"\n  1-5. 라벨 검증 시각화")
    verify_labels(n_samples=4, save_path=OUTPUT_DIR / "bbox_verification.png")

    t_data = time.perf_counter() - t_start
    print(f"\n  데이터 준비 완료: {t_data:.1f}s")

    # ── Step 2: 모델 학습 ──
    print("\n[2] YOLOv11n 학습...")
    t_train_start = time.perf_counter()

    best_weights = train_yolo(
        dataset_yaml=yaml_path,
        model_name="yolo11n.pt",
        epochs=150,
        batch=8,
        patience=30,
    )

    t_train = time.perf_counter() - t_train_start
    print(f"\n  학습 완료: {t_train/60:.1f}분")

    # ── Step 3: 평가 ──
    print("\n[3] 모델 평가...")
    metrics = evaluate_model(best_weights, yaml_path)

    print("\n  예측 시각화...")
    visualize_predictions(
        best_weights, n_samples=6,
        save_path=OUTPUT_DIR / "val_predictions.png",
    )

    print("\n  파라미터별 성능 분석...")
    analyze_performance_by_param(
        best_weights,
        save_path=OUTPUT_DIR / "param_analysis.png",
    )

    # ── Step 4: 실제 데이터 추론 ──
    print("\n[4] 실제 데이터 추론 (zero-shot)...")
    real_results = inference_on_real_data(best_weights, output_dir=OUTPUT_DIR)

    # ── Step 5: DB 기록 ──
    print("\n[5] DB 기록...")
    log_to_database(db, dataset_summary, metrics, real_results)

    # ── 최종 요약 ──
    t_total = time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print("  Week 4 최종 요약")
    print("=" * 60)
    print(f"  데이터셋: {dataset_summary['total']}개 "
          f"(train={dataset_summary['train']}, val={dataset_summary['val']})")
    print(f"  mAP50: {metrics['mAP50']:.4f} "
          f"({'✓ 달성' if metrics['mAP50'] >= 0.7 else '✗ 미달'})")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    if real_results:
        print(f"  실제 데이터 추론:")
        for r in real_results:
            print(f"    {r['dataset']}: raw={r['raw_detections']} "
                  f"/ proc={r['proc_detections']} detections")
    print(f"\n  총 소요 시간: {t_total/60:.1f}분")
    print(f"  출력:")
    for f in ['bbox_verification.png', 'val_predictions.png',
              'param_analysis.png', 'real_frenke.png', 'real_guangzhou.png']:
        p = OUTPUT_DIR / f
        print(f"    {'✓' if p.exists() else '✗'} {f}")
    print(f"  모델: {best_weights}")

    db.print_summary()
    print("완료!")
