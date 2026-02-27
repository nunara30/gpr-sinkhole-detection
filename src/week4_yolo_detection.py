"""
Week 4 - YOLOv11 GPR 객체 탐지 (다중 클래스 + negative sampling)

단일 클래스(sinkhole) 기능 유지 + 4클래스 확장:
  Class 0: sinkhole (공동 → 다중 산란 쌍곡선)
  Class 1: pipe     (금속관 → 단일 강한 쌍곡선)
  Class 2: rebar    (철근 배열 → 주기적 소형 쌍곡선)
  (negative): background (객체 없음 → 빈 라벨, negative sample)

파이프라인:
  1. 합성 B-scan → 640×640 grayscale PNG + YOLO bbox 라벨
  2. 위치 변경 증강 (×4) + 전처리 버전 (×2) → ~1100개
  3. YOLOv11n 학습 (200 epochs, pretrained backbone)
  4. 검증 셋 평가 (mAP50 > 0.8 목표)
  5. 실제 GPR 데이터 추론 (Frenke, Guangzhou pipe/rebar/tunnel)
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
    synthesize_pipe_bscan, synthesize_rebar_bscan, synthesize_background_bscan,
    run_multiclass_simulations,
    PIPE_SCENARIOS, REBAR_SCENARIOS, BACKGROUND_SCENARIOS,
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

# 다중 클래스 YOLO 경로
YOLO_MC_DIR = DATA_DIR / "yolo_multiclass"
YOLO_MC_IMAGES_TRAIN = YOLO_MC_DIR / "images" / "train"
YOLO_MC_IMAGES_VAL = YOLO_MC_DIR / "images" / "val"
YOLO_MC_LABELS_TRAIN = YOLO_MC_DIR / "labels" / "train"
YOLO_MC_LABELS_VAL = YOLO_MC_DIR / "labels" / "val"
MC_OUTPUT_DIR = BASE_DIR / "src" / "output" / "week4_multiclass"

# 클래스 매핑
CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']  # nc=3, background는 빈 라벨
CLASS_COLORS = {'sinkhole': 'red', 'pipe': 'cyan', 'rebar': 'yellow', 'background': 'gray'}

for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL,
          YOLO_LABELS_TRAIN, YOLO_LABELS_VAL,
          YOLO_MC_IMAGES_TRAIN, YOLO_MC_IMAGES_VAL,
          YOLO_MC_LABELS_TRAIN, YOLO_MC_LABELS_VAL,
          YOLO_RUNS_DIR, OUTPUT_DIR, MC_OUTPUT_DIR]:
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
# Step 1-3b: 다중 클래스 bbox + 증강
# ─────────────────────────────────────────────

def compute_yolo_bbox_multiclass(meta):
    """
    다중 클래스 메타데이터에서 YOLO bbox 계산

    Returns: list of (class_id, cx, cy, w, h) 또는 [] (background)
    """
    obj_type = meta.get('object_type', 'sinkhole')

    if obj_type == 'background':
        return []  # 빈 라벨

    if obj_type == 'sinkhole':
        # 기존 로직 (class_id=0)
        bbox = compute_yolo_bbox(meta)
        return [bbox] if bbox else []

    if obj_type == 'pipe':
        # pipe: 단일 쌍곡선, sinkhole과 유사하되 radius=pipe_diameter/2
        depth_m = meta['depth_m']
        pipe_diam = meta['pipe_diameter']
        pipe_x = meta.get('pipe_x', meta.get('domain_x', 5.0) / 2)
        soil_epsr = meta['soil_epsr']
        n_samples = meta['n_samples']
        n_traces = meta['n_traces']
        dt_ns = meta['dt_ns']
        dx = meta['dx']

        v = soil_velocity(soil_epsr)
        v_m_s = v * 1e9

        radius = pipe_diam / 2
        d_top = max(0.02, depth_m - radius)
        d_bottom = depth_m + radius

        dt_s = dt_ns * 1e-9
        sample_top = (2 * d_top / v_m_s) / dt_s
        sample_bottom = (2 * d_bottom / v_m_s) / dt_s

        height_samples = sample_bottom - sample_top
        margin = max(height_samples * 0.2, 5)
        sample_top = max(0, sample_top - margin)
        sample_bottom = min(n_samples - 1, sample_bottom + margin)

        # 수평: 쌍곡선 확산 범위
        if d_bottom > d_top:
            x_half = np.sqrt(d_bottom ** 2 - d_top ** 2)
        else:
            x_half = 2 * radius
        x_half = max(x_half, 2 * radius)

        trace_center = pipe_x / dx
        trace_half = x_half / dx
        trace_left = max(0, trace_center - trace_half)
        trace_right = min(n_traces - 1, trace_center + trace_half)

        cx = (trace_left + trace_right) / 2 / n_traces
        cy = (sample_top + sample_bottom) / 2 / n_samples
        w = (trace_right - trace_left) / n_traces
        h = (sample_bottom - sample_top) / n_samples

        if w < 0.01 or h < 0.01 or w > 0.95 or h > 0.95:
            return []
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        w = np.clip(w, 0.01, 1.0)
        h = np.clip(h, 0.01, 1.0)

        return [(1, cx, cy, w, h)]  # class_id=1

    if obj_type == 'rebar':
        # rebar: 전체 철근 배열을 감싸는 단일 bbox
        depth_m = meta['depth_m']
        rebar_positions = meta['rebar_positions']
        soil_epsr = meta['soil_epsr']
        n_samples = meta['n_samples']
        n_traces = meta['n_traces']
        dt_ns = meta['dt_ns']
        dx = meta['dx']

        v = soil_velocity(soil_epsr)
        v_m_s = v * 1e9

        rebar_radius = 0.008  # 일관성 (week3과 동일)
        d_top = max(0.02, depth_m - rebar_radius)
        d_bottom = depth_m + rebar_radius

        dt_s = dt_ns * 1e-9
        sample_top = (2 * d_top / v_m_s) / dt_s
        sample_bottom = (2 * d_bottom / v_m_s) / dt_s

        height_samples = sample_bottom - sample_top
        margin = max(height_samples * 0.3, 8)
        sample_top = max(0, sample_top - margin)
        sample_bottom = min(n_samples - 1, sample_bottom + margin)

        # 수평: 첫/마지막 철근 + 쌍곡선 확산
        x_min_rebar = min(rebar_positions)
        x_max_rebar = max(rebar_positions)
        if d_bottom > d_top:
            x_spread = np.sqrt(d_bottom ** 2 - d_top ** 2)
        else:
            x_spread = 0.1
        x_spread = max(x_spread, 0.05)

        trace_left = max(0, (x_min_rebar - x_spread) / dx)
        trace_right = min(n_traces - 1, (x_max_rebar + x_spread) / dx)

        cx = (trace_left + trace_right) / 2 / n_traces
        cy = (sample_top + sample_bottom) / 2 / n_samples
        w = (trace_right - trace_left) / n_traces
        h = (sample_bottom - sample_top) / n_samples

        if w < 0.01 or h < 0.01 or w > 0.99 or h > 0.95:
            return []
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        w = np.clip(w, 0.01, 1.0)
        h = np.clip(h, 0.01, 1.0)

        return [(2, cx, cy, w, h)]  # class_id=2

    return []


def write_yolo_label_multiclass(label_path, bbox_list):
    """다중 bbox YOLO 라벨 파일 쓰기"""
    if not bbox_list:
        label_path.write_text("", encoding='utf-8')
        return
    lines = []
    for (class_id, cx, cy, w, h) in bbox_list:
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n", encoding='utf-8')


def generate_shifted_pipe(meta, new_pipe_x):
    """파이프 x 위치 변경 재합성"""
    bscan, _, _, new_meta = synthesize_pipe_bscan(
        freq_hz=meta['freq_hz'],
        depth_m=meta['depth_m'],
        pipe_diameter=meta['pipe_diameter'],
        soil_epsr=meta['soil_epsr'],
        domain_x=meta.get('domain_x', 5.0),
        domain_z=meta.get('domain_z', 3.0),
        dx=meta['dx'],
        n_samples=meta['n_samples'],
        pipe_x=new_pipe_x,
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
# Step 1-4b: 다중 클래스 데이터셋 준비
# ─────────────────────────────────────────────

def prepare_multiclass_dataset(val_ratio=0.2, n_shift_variants=4, seed=42):
    """
    다중 클래스 YOLO 데이터셋 준비

    - sinkhole (48 base, class 0) — 기존 재사용
    - pipe (~72 base, class 1)
    - rebar (~72 base, class 2)
    - background (~27 base, negative — 빈 라벨)
    - 위치 증강 (×4) + 전처리 증강 (×2) = ×6 per base

    Returns: summary dict with per-class counts
    """
    rng = np.random.RandomState(seed)

    # 기존 데이터 정리
    for d in [YOLO_MC_IMAGES_TRAIN, YOLO_MC_IMAGES_VAL,
              YOLO_MC_LABELS_TRAIN, YOLO_MC_LABELS_VAL]:
        for f in d.glob("*"):
            f.unlink()

    all_items = []  # (scenario_key, img_name, bscan, meta, obj_type)

    meta_files = sorted(SYNTHETIC_DIR.glob("*_meta.json"))
    print(f"  총 메타데이터 파일: {len(meta_files)}개")

    class_counts = {'sinkhole': 0, 'pipe': 0, 'rebar': 0, 'background': 0}

    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        label = meta_path.stem.replace('_meta', '')
        npy_path = SYNTHETIC_DIR / f"{label}.npy"
        if not npy_path.exists():
            continue

        bscan = np.load(str(npy_path))
        obj_type = meta.get('object_type', 'sinkhole')

        # 시나리오 키 (split 기준)
        freq_mhz = meta.get('freq_mhz', meta['freq_hz'] / 1e6)
        if obj_type == 'sinkhole':
            scenario_key = ('sinkhole', freq_mhz, meta['depth_m'],
                           meta['radius_m'], meta['soil_epsr'])
        elif obj_type == 'pipe':
            scenario_key = ('pipe', freq_mhz, meta['depth_m'],
                           meta['pipe_diameter'], meta['soil_epsr'])
        elif obj_type == 'rebar':
            scenario_key = ('rebar', freq_mhz, meta['depth_m'],
                           meta.get('spacing_m', 0), meta['soil_epsr'])
        else:
            scenario_key = ('background', freq_mhz, meta['soil_epsr'],
                           meta.get('n_layers', 0), 0)

        class_counts[obj_type] += 1

        # (a) 원본
        all_items.append((scenario_key, f"{label}_raw", bscan, meta, obj_type))

        # (b) 전처리 적용 버전
        try:
            # background/pipe/rebar도 전처리 가능하도록 메타 보완
            meta_for_proc = dict(meta)
            if 'freq_mhz' not in meta_for_proc:
                meta_for_proc['freq_mhz'] = meta['freq_hz'] / 1e6
            if 'dt_ns' not in meta_for_proc:
                v = soil_velocity(meta['soil_epsr'])
                dz = meta.get('domain_z', 3.0)
                meta_for_proc['dt_ns'] = (2 * dz / v * 1.2) / meta['n_samples']
            if 'dx' not in meta_for_proc:
                meta_for_proc['dx'] = 0.01
            processed, _, _ = apply_preprocessing_to_synthetic(bscan, meta_for_proc)
            all_items.append((scenario_key, f"{label}_proc", processed, meta, obj_type))
        except Exception as e:
            pass  # 전처리 실패 시 무시

        # (c) 위치 변경 증강 (sinkhole, pipe만 — rebar/background는 skip)
        if obj_type == 'sinkhole':
            shift_positions = rng.uniform(0.8, 4.2, size=n_shift_variants)
            for si, new_x in enumerate(shift_positions):
                try:
                    shifted_bscan, shifted_meta = generate_shifted_bscan(meta, new_x)
                    all_items.append((scenario_key, f"{label}_shift{si}",
                                     shifted_bscan, shifted_meta, obj_type))
                except Exception:
                    pass
        elif obj_type == 'pipe':
            shift_positions = rng.uniform(0.8, 4.2, size=n_shift_variants)
            for si, new_x in enumerate(shift_positions):
                try:
                    shifted_bscan, shifted_meta = generate_shifted_pipe(meta, new_x)
                    all_items.append((scenario_key, f"{label}_shift{si}",
                                     shifted_bscan, shifted_meta, obj_type))
                except Exception:
                    pass
        elif obj_type == 'rebar':
            # rebar: 위치 변경 어려움 (배열 자체가 도메인 중앙) → 2개만 증강
            for si in range(min(2, n_shift_variants)):
                try:
                    shifted_bscan, _, _, shifted_meta = synthesize_rebar_bscan(
                        freq_hz=meta['freq_hz'],
                        depth_m=meta['depth_m'],
                        spacing_m=meta.get('spacing_m', 0.15),
                        n_rebars=meta.get('n_rebars', 5),
                        soil_epsr=meta['soil_epsr'],
                    )
                    all_items.append((scenario_key, f"{label}_aug{si}",
                                     shifted_bscan, shifted_meta, obj_type))
                except Exception:
                    pass
        # background: 원본 + 전처리만 (증강 불필요, 이미 다양)

    print(f"  기본 데이터 수: {class_counts}")
    print(f"  증강 후 총 이미지 수: {len(all_items)}")

    # ── Stratified Train/Val split (클래스별 80/20) ──
    unique_scenarios = list(set(item[0] for item in all_items))
    # 클래스별로 분리하여 split
    scenarios_by_class = {}
    for sk in unique_scenarios:
        cls = sk[0]
        if cls not in scenarios_by_class:
            scenarios_by_class[cls] = []
        scenarios_by_class[cls].append(sk)

    val_scenarios = set()
    for cls, sks in scenarios_by_class.items():
        rng.shuffle(sks)
        n_val = max(1, int(len(sks) * val_ratio))
        for sk in sks[:n_val]:
            val_scenarios.add(sk)

    train_count = 0
    val_count = 0
    skipped = 0
    per_class = {c: {'train': 0, 'val': 0} for c in ['sinkhole', 'pipe', 'rebar', 'background']}

    for scenario_key, img_name, bscan, meta, obj_type in all_items:
        # bbox 계산
        bbox_list = compute_yolo_bbox_multiclass(meta)
        # background는 bbox_list = [] → 빈 라벨 (정상)
        if obj_type != 'background' and not bbox_list:
            skipped += 1
            continue

        is_val = scenario_key in val_scenarios
        img_dir = YOLO_MC_IMAGES_VAL if is_val else YOLO_MC_IMAGES_TRAIN
        lbl_dir = YOLO_MC_LABELS_VAL if is_val else YOLO_MC_LABELS_TRAIN

        png_path = img_dir / f"{img_name}.png"
        npy_to_png(bscan, png_path)

        lbl_path = lbl_dir / f"{img_name}.txt"
        write_yolo_label_multiclass(lbl_path, bbox_list)

        split = 'val' if is_val else 'train'
        per_class[obj_type][split] += 1
        if is_val:
            val_count += 1
        else:
            train_count += 1

    summary = {
        'total': train_count + val_count,
        'train': train_count,
        'val': val_count,
        'skipped': skipped,
        'per_class': per_class,
    }

    print(f"\n  === 다중 클래스 데이터셋 요약 ===")
    print(f"  {'클래스':<12} {'Train':>6} {'Val':>6} {'Total':>6}")
    print(f"  {'-'*32}")
    for cls in ['sinkhole', 'pipe', 'rebar', 'background']:
        t = per_class[cls]['train']
        v = per_class[cls]['val']
        print(f"  {cls:<12} {t:>6} {v:>6} {t+v:>6}")
    print(f"  {'-'*32}")
    print(f"  {'합계':<12} {train_count:>6} {val_count:>6} {train_count+val_count:>6}")
    print(f"  Skipped: {skipped}")

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


def create_multiclass_yaml():
    """다중 클래스 YOLO dataset.yaml 생성 (nc=3, background는 빈 라벨)"""
    yaml_content = f"""path: {str(YOLO_MC_DIR).replace(chr(92), '/')}
train: images/train
val: images/val

nc: 3
names: ['sinkhole', 'pipe', 'rebar']
"""
    yaml_path = YOLO_MC_DIR / "dataset.yaml"
    yaml_path.write_text(yaml_content, encoding='utf-8')
    print(f"  multiclass dataset.yaml: {yaml_path}")
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
# Step 6: 다중 클래스 학습/평가/추론
# ─────────────────────────────────────────────

def verify_multiclass_labels(n_per_class=2, save_path=None):
    """다중 클래스 bbox 검증 시각화 (클래스별 샘플)"""
    train_pngs = sorted(YOLO_MC_IMAGES_TRAIN.glob("*.png"))
    if not train_pngs:
        print("  검증할 이미지 없음")
        return

    # 클래스별 샘플 수집
    class_samples = {'sinkhole': [], 'pipe': [], 'rebar': [], 'background': []}
    for png_path in train_pngs:
        name = png_path.stem
        if name.startswith('pipe_'):
            cls = 'pipe'
        elif name.startswith('rebar_'):
            cls = 'rebar'
        elif name.startswith('bg_'):
            cls = 'background'
        else:
            cls = 'sinkhole'
        if len(class_samples[cls]) < n_per_class:
            class_samples[cls].append(png_path)

    all_selected = []
    for cls in ['sinkhole', 'pipe', 'rebar', 'background']:
        all_selected.extend([(cls, p) for p in class_samples[cls]])

    n_total = len(all_selected)
    if n_total == 0:
        return

    cols = min(4, n_total)
    rows = (n_total + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    color_map = {0: 'red', 1: 'cyan', 2: 'yellow'}

    for idx, (cls, png_path) in enumerate(all_selected):
        ax = axes[idx]
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray', aspect='equal')

        lbl_path = YOLO_MC_LABELS_TRAIN / f"{png_path.stem}.txt"
        if lbl_path.exists():
            text = lbl_path.read_text().strip()
            if text:
                for line in text.split('\n'):
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    img_h, img_w = img.shape
                    px = (cx - w / 2) * img_w
                    py = (cy - h / 2) * img_h
                    rect = Rectangle((px, py), w * img_w, h * img_h,
                                     linewidth=2,
                                     edgecolor=color_map.get(cls_id, 'white'),
                                     facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else '?'
                    ax.text(px, py - 3, cls_name, color=color_map.get(cls_id, 'white'),
                            fontsize=8, fontweight='bold')
                ax.set_title(f"[{cls}] {png_path.stem}", fontsize=6)
            else:
                ax.set_title(f"[{cls}] {png_path.stem}\n(negative)", fontsize=6)
        ax.axis('off')

    for i in range(n_total, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Multiclass Label Verification\n(red=sinkhole, cyan=pipe, yellow=rebar)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


def train_multiclass(dataset_yaml, model_name="yolo11n.pt", epochs=200,
                     batch=8, patience=40):
    """다중 클래스 YOLO 학습"""
    from ultralytics import YOLO

    try:
        model = YOLO(model_name)
        print(f"  모델: {model_name}")
    except Exception:
        model_name = "yolov8n.pt"
        model = YOLO(model_name)
        print(f"  Fallback 모델: {model_name}")

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
        label_smoothing=0.05,
        # Augmentation
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
        degrees=0.0,
        hsv_v=0.3,
        translate=0.15,
        scale=0.3,
        workers=0,
        project=str(YOLO_RUNS_DIR),
        name="multiclass_detect",
        exist_ok=True,
        verbose=True,
    )

    try:
        results = model.train(**train_kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            print(f"  OOM → batch={batch//2}로 재시도")
            import torch
            torch.cuda.empty_cache()
            train_kwargs['batch'] = batch // 2
            results = model.train(**train_kwargs)
        else:
            raise

    best_path = YOLO_RUNS_DIR / "multiclass_detect" / "weights" / "best.pt"
    if not best_path.exists():
        for p in YOLO_RUNS_DIR.rglob("best.pt"):
            best_path = p
            break
    print(f"  Best model: {best_path}")
    return best_path


def evaluate_multiclass(weights_path, dataset_yaml):
    """다중 클래스 모델 평가 (클래스별 메트릭)"""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    results = model.val(data=str(dataset_yaml), imgsz=640, verbose=True)

    metrics = {
        'mAP50': float(results.box.map50),
        'mAP50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
    }
    p, r = metrics['precision'], metrics['recall']
    metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    # 클래스별 메트릭
    per_class_metrics = {}
    try:
        ap50_per = results.box.ap50
        for i, cls_name in enumerate(CLASS_NAMES):
            if i < len(ap50_per):
                per_class_metrics[cls_name] = {
                    'ap50': float(ap50_per[i]),
                }
    except Exception:
        pass
    metrics['per_class'] = per_class_metrics

    print(f"\n  === 다중 클래스 평가 결과 ===")
    print(f"  전체 mAP50:     {metrics['mAP50']:.4f}")
    print(f"  전체 mAP50-95:  {metrics['mAP50_95']:.4f}")
    print(f"  전체 Precision: {metrics['precision']:.4f}")
    print(f"  전체 Recall:    {metrics['recall']:.4f}")
    print(f"  전체 F1:        {metrics['f1']:.4f}")
    if per_class_metrics:
        print(f"\n  클래스별 AP50:")
        for cls_name, m in per_class_metrics.items():
            ap = m.get('ap50', 0)
            status = '✓' if ap >= 0.7 else '✗'
            print(f"    {status} {cls_name}: {ap:.4f}")

    target = 0.8
    if metrics['mAP50'] >= target:
        print(f"\n  ✓ 전체 mAP50 목표 달성 ({metrics['mAP50']:.3f} >= {target})")
    else:
        print(f"\n  ✗ 전체 mAP50 목표 미달 ({metrics['mAP50']:.3f} < {target})")

    return metrics


def visualize_multiclass_predictions(weights_path, n_samples=8, save_path=None):
    """다중 클래스 검증 셋 예측 시각화"""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    val_pngs = sorted(YOLO_MC_IMAGES_VAL.glob("*.png"))
    if not val_pngs:
        print("  시각화할 이미지 없음")
        return

    indices = np.random.choice(len(val_pngs), min(n_samples, len(val_pngs)),
                               replace=False)
    selected = [val_pngs[i] for i in indices]

    cols = min(4, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    gt_colors = {0: 'red', 1: 'cyan', 2: 'yellow'}
    pred_colors = {0: 'lime', 1: 'deepskyblue', 2: 'orange'}

    for idx, (ax, png_path) in enumerate(zip(axes, selected)):
        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap='gray', aspect='equal')
        img_h, img_w = img.shape[:2]

        # GT bbox
        lbl_path = YOLO_MC_LABELS_VAL / f"{png_path.stem}.txt"
        if lbl_path.exists():
            text = lbl_path.read_text().strip()
            if text:
                for line in text.split('\n'):
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    px = (cx - w / 2) * img_w
                    py = (cy - h / 2) * img_h
                    rect = Rectangle((px, py), w * img_w, h * img_h,
                                     linewidth=2,
                                     edgecolor=gt_colors.get(cls_id, 'red'),
                                     facecolor='none', linestyle='--')
                    ax.add_patch(rect)

        # Predictions
        results = model.predict(str(png_path), imgsz=640, conf=0.25, verbose=False)
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                color = pred_colors.get(cls_id, 'lime')
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else '?'
                rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1 - 3, f'{cls_name} {conf:.2f}', color=color,
                        fontsize=7, fontweight='bold')

        ax.set_title(png_path.stem, fontsize=6)
        ax.axis('off')

    for i in range(len(selected), len(axes)):
        axes[i].set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='GT sinkhole'),
        Line2D([0], [0], color='cyan', linewidth=2, linestyle='--', label='GT pipe'),
        Line2D([0], [0], color='yellow', linewidth=2, linestyle='--', label='GT rebar'),
        Line2D([0], [0], color='lime', linewidth=2, label='Pred sinkhole'),
        Line2D([0], [0], color='deepskyblue', linewidth=2, label='Pred pipe'),
        Line2D([0], [0], color='orange', linewidth=2, label='Pred rebar'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=8,
               ncol=2)
    fig.suptitle("Multiclass Val: GT (dashed) vs Predicted (solid)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)


def inference_multiclass_real(weights_path, output_dir=None):
    """
    다중 클래스 모델로 실측 데이터 zero-shot 추론

    - Guangzhou pipe → pipe 탐지 기대
    - Guangzhou rebar → rebar 탐지 기대
    - Guangzhou tunnel → background (탐지 없음) 기대
    - Frenke LINE00 → background (탐지 없음) 기대
    """
    from ultralytics import YOLO

    if output_dir is None:
        output_dir = MC_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    results_summary = []

    # 실측 데이터 소스
    real_datasets = [
        {
            'name': 'Guangzhou pipe',
            'dir': DATA_DIR / "guangzhou/Data Set/pipe",
            'expected_class': 'pipe',
            'max_files': 5,
        },
        {
            'name': 'Guangzhou rebar',
            'dir': DATA_DIR / "guangzhou/Data Set/rebar",
            'expected_class': 'rebar',
            'max_files': 5,
        },
        {
            'name': 'Guangzhou tunnel',
            'dir': DATA_DIR / "guangzhou/Data Set/tunnel",
            'expected_class': 'background',
            'max_files': 5,
        },
    ]

    fig_rows = []

    for ds_info in real_datasets:
        ds_dir = ds_info['dir']
        if not ds_dir.exists():
            print(f"  [{ds_info['name']}] 디렉토리 없음: {ds_dir}")
            continue

        dt_files = sorted(
            f for f in ds_dir.rglob("*.dt")
            if 'ASCII' not in str(f)
        )
        if not dt_files:
            print(f"  [{ds_info['name']}] .dt 파일 없음")
            continue

        sample_files = dt_files[:ds_info['max_files']]
        print(f"\n  [{ds_info['name']}] {len(sample_files)}개 추론...")

        ds_detections = {'sinkhole': 0, 'pipe': 0, 'rebar': 0, 'none': 0}
        sample_images = []

        for dt_file in sample_files:
            data, header = read_ids_dt(str(dt_file))
            if data is None:
                continue

            img = _prepare_real_image(data)
            tmp_path = output_dir / f"_tmp_{dt_file.stem}.png"
            cv2.imwrite(str(tmp_path), img)

            pred = model.predict(str(tmp_path), imgsz=640, conf=0.25, verbose=False)

            det_classes = []
            if pred and len(pred[0].boxes):
                for box in pred[0].boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    if cls_id < len(CLASS_NAMES):
                        ds_detections[CLASS_NAMES[cls_id]] += 1
                        det_classes.append(CLASS_NAMES[cls_id])
            if not det_classes:
                ds_detections['none'] += 1

            if len(sample_images) < 2:
                sample_images.append((img, pred, dt_file.stem))

            tmp_path.unlink(missing_ok=True)

        results_summary.append({
            'dataset': ds_info['name'],
            'expected': ds_info['expected_class'],
            'detections': ds_detections,
            'n_files': len(sample_files),
        })
        fig_rows.append((ds_info['name'], ds_info['expected_class'], sample_images))

        print(f"    탐지 결과: {ds_detections}")

    # Frenke LINE00
    frenke_path = DATA_DIR / "frenke/2014_04_25_frenke/rawGPR/LINE00.DT1"
    if frenke_path.exists():
        print(f"\n  [Frenke LINE00] 추론...")
        data_frenke, header_frenke = read_dt1(str(frenke_path))
        img = _prepare_real_image(data_frenke)
        tmp_path = output_dir / "_tmp_frenke.png"
        cv2.imwrite(str(tmp_path), img)
        pred = model.predict(str(tmp_path), imgsz=640, conf=0.25, verbose=False)

        n_det = len(pred[0].boxes) if pred and pred[0].boxes is not None else 0
        frenke_classes = {}
        if pred and len(pred[0].boxes):
            for box in pred[0].boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else '?'
                frenke_classes[cls_name] = frenke_classes.get(cls_name, 0) + 1

        results_summary.append({
            'dataset': 'Frenke LINE00',
            'expected': 'background',
            'detections': frenke_classes if frenke_classes else {'none': 1},
            'n_files': 1,
        })
        print(f"    탐지: {n_det}개 {frenke_classes}")
        tmp_path.unlink(missing_ok=True)

    # 시각화: 실측 추론 결과 그리드
    if fig_rows:
        n_rows = len(fig_rows)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        pred_colors = {0: 'lime', 1: 'deepskyblue', 2: 'orange'}

        for row_idx, (ds_name, expected, samples) in enumerate(fig_rows):
            for col_idx in range(2):
                ax = axes[row_idx, col_idx]
                if col_idx < len(samples):
                    img, pred, name = samples[col_idx]
                    ax.imshow(img, cmap='gray', aspect='equal')
                    n_det = 0
                    if pred and len(pred[0].boxes):
                        for box in pred[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls_id = int(box.cls[0].cpu().numpy())
                            color = pred_colors.get(cls_id, 'lime')
                            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else '?'
                            rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=2, edgecolor=color,
                                             facecolor='none')
                            ax.add_patch(rect)
                            ax.text(x1, y1 - 3, f'{cls_name} {conf:.2f}',
                                    color=color, fontsize=8, fontweight='bold')
                            n_det += 1
                    ax.set_title(f"{ds_name}\n{name} ({n_det} det)\nExpected: {expected}",
                                 fontsize=8)
                else:
                    ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                            ha='center', va='center')
                ax.axis('off')

        fig.suptitle("Multiclass Real Data Inference (Zero-shot)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        save_p = output_dir / "multiclass_real_inference.png"
        fig.savefig(save_p, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_p}")
        plt.close(fig)

    return results_summary


def log_multiclass_to_database(db, dataset_summary, metrics, real_results):
    """다중 클래스 결과를 DB에 기록"""
    dummy_data = np.zeros((640, 640), dtype=np.float32)
    ds_id = db.register_dataset(
        name=f"YOLO Multiclass Dataset (n={dataset_summary['total']})",
        file_path=str(YOLO_MC_DIR / "dataset.yaml"),
        data=dummy_data,
        format="YOLO_multiclass_dataset",
        frequency_mhz=0,
        time_window_ns=0,
        dx_m=0,
    )

    steps = [
        {
            'step_name': 'Multiclass_Dataset',
            'parameters': {
                'train': dataset_summary['train'],
                'val': dataset_summary['val'],
                'total': dataset_summary['total'],
                'per_class': {k: v for k, v in dataset_summary.get('per_class', {}).items()},
            },
            'elapsed_ms': 0,
        },
        {
            'step_name': 'Multiclass_Training',
            'parameters': {
                'model': 'yolo11n',
                'epochs': 200,
                'batch': 8,
                'nc': 3,
                'label_smoothing': 0.05,
            },
            'elapsed_ms': 0,
        },
        {
            'step_name': 'Multiclass_Evaluation',
            'parameters': {
                k: v for k, v in metrics.items() if k != 'per_class'
            },
            'elapsed_ms': 0,
        },
    ]

    if real_results:
        real_params = {}
        for r in real_results:
            real_params[r['dataset']] = str(r.get('detections', {}))
        steps.append({
            'step_name': 'Multiclass_Real_Inference',
            'parameters': real_params,
            'elapsed_ms': 0,
        })

    db.log_processing_run(ds_id, "Week4 Multiclass YOLO Detection", steps)
    return ds_id


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)

    # ── 모드 선택: multiclass (기본) vs singleclass ──
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'multi'], default='multi',
                        help='single: 기존 단일 클래스, multi: 다중 클래스')
    parser.add_argument('--skip-synth', action='store_true',
                        help='합성 데이터 생성 건너뛰기 (이미 존재할 때)')
    args = parser.parse_args()

    t_start = time.perf_counter()
    db = GPRDatabase()

    if args.mode == 'single':
        # ════════════════════════════════════════
        #   단일 클래스 (기존 Week 4 코드)
        # ════════════════════════════════════════
        print("=" * 60)
        print("  Week 4 - YOLOv11 Sinkhole Detection (Single Class)")
        print("=" * 60)

        print("\n[1] 데이터셋 준비...")
        dataset_summary = prepare_dataset(val_ratio=0.2, n_shift_variants=4)
        yaml_path = create_dataset_yaml()
        verify_labels(n_samples=4, save_path=OUTPUT_DIR / "bbox_verification.png")

        print("\n[2] YOLOv11n 학습...")
        best_weights = train_yolo(yaml_path, epochs=150, batch=8, patience=30)

        print("\n[3] 평가...")
        metrics = evaluate_model(best_weights, yaml_path)
        visualize_predictions(best_weights, save_path=OUTPUT_DIR / "val_predictions.png")
        analyze_performance_by_param(best_weights, save_path=OUTPUT_DIR / "param_analysis.png")

        print("\n[4] 실제 데이터 추론...")
        real_results = inference_on_real_data(best_weights, output_dir=OUTPUT_DIR)

        print("\n[5] DB 기록...")
        log_to_database(db, dataset_summary, metrics, real_results)

    else:
        # ════════════════════════════════════════
        #   다중 클래스 (4클래스 확장 + negative)
        # ════════════════════════════════════════
        print("=" * 60)
        print("  Week 4 - Multiclass YOLO Detection")
        print("  (sinkhole + pipe + rebar + background)")
        print("=" * 60)

        # ── Step 1: 합성 데이터 생성 ──
        if not args.skip_synth:
            print("\n[1] 다중 클래스 합성 데이터 생성...")
            synth_results = run_multiclass_simulations(db=db)
            print(f"  신규 합성: {len(synth_results)}개")
        else:
            print("\n[1] 합성 데이터 생성 건너뛰기 (--skip-synth)")

        # ── Step 2: 데이터셋 준비 ──
        print("\n[2] 다중 클래스 YOLO 데이터셋 준비...")
        mc_summary = prepare_multiclass_dataset(
            val_ratio=0.2, n_shift_variants=4, seed=42)
        mc_yaml = create_multiclass_yaml()

        print("\n  라벨 검증 시각화...")
        verify_multiclass_labels(
            n_per_class=2,
            save_path=MC_OUTPUT_DIR / "multiclass_bbox_verify.png")

        t_data = time.perf_counter() - t_start
        print(f"\n  데이터 준비 완료: {t_data:.1f}s")

        # ── Step 3: 학습 ──
        print("\n[3] 다중 클래스 YOLO 학습...")
        t_train_start = time.perf_counter()
        mc_weights = train_multiclass(
            mc_yaml, model_name="yolo11n.pt",
            epochs=200, batch=8, patience=40)
        t_train = time.perf_counter() - t_train_start
        print(f"\n  학습 완료: {t_train/60:.1f}분")

        # ── Step 4: 평가 ──
        print("\n[4] 다중 클래스 평가...")
        mc_metrics = evaluate_multiclass(mc_weights, mc_yaml)

        print("\n  예측 시각화...")
        visualize_multiclass_predictions(
            mc_weights, n_samples=8,
            save_path=MC_OUTPUT_DIR / "multiclass_val_predictions.png")

        # ── Step 5: 실측 추론 ──
        print("\n[5] 실측 데이터 zero-shot 추론...")
        mc_real_results = inference_multiclass_real(
            mc_weights, output_dir=MC_OUTPUT_DIR)

        # ── Step 6: DB 기록 ──
        print("\n[6] DB 기록...")
        log_multiclass_to_database(db, mc_summary, mc_metrics, mc_real_results)

        # ── 최종 요약 ──
        t_total = time.perf_counter() - t_start
        print("\n" + "=" * 60)
        print("  다중 클래스 YOLO 최종 요약")
        print("=" * 60)
        print(f"\n  데이터셋: {mc_summary['total']}개 "
              f"(train={mc_summary['train']}, val={mc_summary['val']})")
        if 'per_class' in mc_summary:
            for cls, counts in mc_summary['per_class'].items():
                print(f"    {cls}: train={counts['train']}, val={counts['val']}")

        print(f"\n  전체 mAP50: {mc_metrics['mAP50']:.4f} "
              f"({'✓' if mc_metrics['mAP50'] >= 0.8 else '✗'})")
        print(f"  Precision:  {mc_metrics['precision']:.4f}")
        print(f"  Recall:     {mc_metrics['recall']:.4f}")
        print(f"  F1:         {mc_metrics['f1']:.4f}")

        if mc_metrics.get('per_class'):
            print(f"\n  클래스별 AP50:")
            for cls_name, m in mc_metrics['per_class'].items():
                print(f"    {cls_name}: {m.get('ap50', 0):.4f}")

        if mc_real_results:
            print(f"\n  실측 추론:")
            for r in mc_real_results:
                print(f"    {r['dataset']} (expected={r['expected']}): "
                      f"{r.get('detections', {})}")

        print(f"\n  총 소요 시간: {t_total/60:.1f}분")
        print(f"  출력:")
        for f_name in ['multiclass_bbox_verify.png',
                        'multiclass_val_predictions.png',
                        'multiclass_real_inference.png']:
            p = MC_OUTPUT_DIR / f_name
            print(f"    {'✓' if p.exists() else '✗'} {f_name}")
        print(f"  모델: {mc_weights}")

    db.print_summary()
    print("완료!")
