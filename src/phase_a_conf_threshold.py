"""
Phase A: Confidence Threshold 실험

단일 클래스(sinkhole) vs 다중 클래스(sinkhole/pipe/rebar) 모델을
다양한 confidence threshold(0.1, 0.15, 0.25)에서 실측 데이터에 적용.

목표:
- threshold 낮출 때 TP가 나오는지 확인
- 두 모델 간 탐지 패턴 비교
- FP 발생 임계값 찾기
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_dt1, read_ids_dt
from week2_preprocessing import dc_removal, dewow, background_removal, bandpass_filter, gain_sec

warnings.filterwarnings('ignore')

# ── 경로 ──
BASE_DIR = Path("G:/RAG_system")
DATA_DIR = BASE_DIR / "data" / "gpr"
OUTPUT_DIR = BASE_DIR / "src" / "output" / "week4_multiclass"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SINGLE_WEIGHTS = BASE_DIR / "models/yolo_runs/sinkhole_detect/weights/best.pt"
MULTI_WEIGHTS  = BASE_DIR / "models/yolo_runs/multiclass_detect/weights/best.pt"

CONF_THRESHOLDS = [0.10, 0.15, 0.25]

CLASS_NAMES_SINGLE = ['sinkhole']
CLASS_NAMES_MULTI  = ['sinkhole', 'pipe', 'rebar']
CLASS_COLORS       = {'sinkhole': '#ff4444', 'pipe': '#00ccff', 'rebar': '#ffee00'}

# ── 실측 데이터 정보 ──
REAL_DATASETS = [
    {
        'name': 'GZ pipe',
        'dir': DATA_DIR / "guangzhou/Data Set/pipe",
        'expected': 'pipe',
        'max_files': 5,
        'reader': 'ids_dt',
        'bandpass': (500e6, 5e9),
    },
    {
        'name': 'GZ rebar',
        'dir': DATA_DIR / "guangzhou/Data Set/rebar",
        'expected': 'rebar',
        'max_files': 5,
        'reader': 'ids_dt',
        'bandpass': (500e6, 5e9),
    },
    {
        'name': 'GZ tunnel',
        'dir': DATA_DIR / "guangzhou/Data Set/tunnel",
        'expected': 'background',
        'max_files': 3,
        'reader': 'ids_dt',
        'bandpass': (500e6, 5e9),
    },
    {
        'name': 'Frenke',
        'file': DATA_DIR / "frenke/2014_04_25_frenke/rawGPR/LINE00.DT1",
        'expected': 'background',
        'reader': 'dt1',
        'bandpass': (25e6, 250e6),
    },
]


def _bscan_to_img(data: np.ndarray) -> np.ndarray:
    """B-scan numpy → 640×640 grayscale uint8 이미지"""
    data = data.astype(np.float32)
    p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
    if p98 - p2 < 1e-10:
        img = np.zeros((640, 640), dtype=np.uint8)
    else:
        data_clipped = np.clip(data, p2, p98)
        img = ((data_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    return cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)


def _apply_simple_preprocess(data: np.ndarray, bp_low_hz: float, bp_high_hz: float,
                              dt_ns: float = 0.2) -> np.ndarray:
    """간단한 전처리 체인
    bp_low_hz, bp_high_hz: 대역통과 차단 주파수 (Hz)
    """
    dt_sec = dt_ns * 1e-9
    data = dc_removal(data)
    data = dewow(data, window=20)
    data = background_removal(data)
    # bandpass_filter(data, dt, low_mhz, high_mhz)
    data = bandpass_filter(data, dt_sec, bp_low_hz / 1e6, bp_high_hz / 1e6)
    data = gain_sec(data, tpow=1.0, alpha=0.0, dt=dt_sec)
    return data


def load_real_datasets():
    """실측 데이터 로드 (raw + preprocessed 쌍)"""
    loaded = []

    for ds in REAL_DATASETS:
        if 'file' in ds:
            # 단일 파일 (Frenke)
            fpath = ds['file']
            if not fpath.exists():
                print(f"  [경고] {fpath} 없음 - 건너뜀")
                continue
            data, header = read_dt1(str(fpath))
            if data is None:
                continue
            tw = float(header.get('TOTAL TIME WINDOW', 50.0))
            dt_ns = tw / data.shape[0]
            data_proc = _apply_simple_preprocess(data, *ds['bandpass'], dt_ns)
            loaded.append({
                'name': ds['name'],
                'expected': ds['expected'],
                'samples': [(data, data_proc, fpath.stem)],
            })
        else:
            # 디렉토리 (Guangzhou)
            ds_dir = ds['dir']
            if not ds_dir.exists():
                print(f"  [경고] {ds_dir} 없음 - 건너뜀")
                continue
            dt_files = sorted(f for f in ds_dir.rglob("*.dt") if 'ASCII' not in str(f))
            samples = []
            for dt_file in dt_files[:ds['max_files']]:
                data, header = read_ids_dt(str(dt_file))
                if data is None:
                    continue
                dt_ns = 0.1  # Guangzhou 2GHz 기본값
                data_proc = _apply_simple_preprocess(data, *ds['bandpass'], dt_ns)
                samples.append((data, data_proc, dt_file.stem))
            if samples:
                loaded.append({
                    'name': ds['name'],
                    'expected': ds['expected'],
                    'samples': samples,
                })

    return loaded


def run_inference_at_conf(model, img: np.ndarray, conf: float) -> list:
    """이미지에 대해 특정 conf로 추론 → [(cls_id, conf_score, xyxy), ...]"""
    tmp_path = OUTPUT_DIR / "_tmp_infer.png"
    cv2.imwrite(str(tmp_path), img)
    results = model.predict(str(tmp_path), imgsz=640, conf=conf, verbose=False)
    tmp_path.unlink(missing_ok=True)

    detections = []
    if results and results[0].boxes is not None and len(results[0].boxes):
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf_score = float(box.conf[0].cpu().numpy())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            detections.append((cls_id, conf_score, xyxy))
    return detections


def build_result_table(datasets_loaded, single_model, multi_model):
    """
    모든 데이터셋 × conf 임계값 × 모델 조합 결과 수집
    Returns: dict[ds_name][conf][model] = {'total_det': int, 'by_class': dict}
    """
    from ultralytics import YOLO

    print("\n[Phase A] 추론 시작...")
    table = {}

    for ds in datasets_loaded:
        name = ds['name']
        print(f"\n  데이터셋: {name} ({len(ds['samples'])}개 파일)")
        table[name] = {}

        for conf in CONF_THRESHOLDS:
            table[name][conf] = {'single': {}, 'multi': {}}

            # 단일 클래스 모델
            s_counts = {'sinkhole': 0}
            m_counts = {'sinkhole': 0, 'pipe': 0, 'rebar': 0}

            for raw, proc, stem in ds['samples']:
                img_raw = _bscan_to_img(raw)
                img_proc = _bscan_to_img(proc)

                for img in [img_raw, img_proc]:
                    # single
                    dets = run_inference_at_conf(single_model, img, conf)
                    for cls_id, sc, _ in dets:
                        cname = CLASS_NAMES_SINGLE[cls_id] if cls_id < len(CLASS_NAMES_SINGLE) else '?'
                        s_counts[cname] = s_counts.get(cname, 0) + 1

                    # multi
                    dets = run_inference_at_conf(multi_model, img, conf)
                    for cls_id, sc, _ in dets:
                        cname = CLASS_NAMES_MULTI[cls_id] if cls_id < len(CLASS_NAMES_MULTI) else '?'
                        m_counts[cname] = m_counts.get(cname, 0) + 1

            total_s = sum(s_counts.values())
            total_m = sum(m_counts.values())
            table[name][conf]['single'] = {'total': total_s, 'by_class': s_counts}
            table[name][conf]['multi']  = {'total': total_m, 'by_class': m_counts}

            print(f"    conf={conf:.2f}  single={total_s}건  multi={total_m}건 {m_counts}")

    return table


def visualize_threshold_comparison(table, datasets_loaded, save_path=None):
    """
    결과 시각화:
    - 상단: 데이터셋별 / conf별 탐지 수 막대그래프 (단일 vs 다중)
    - 하단: 대표 이미지 + 다중 모델 추론 (conf=0.10)
    """
    from ultralytics import YOLO
    multi_model = YOLO(str(MULTI_WEIGHTS))

    n_ds = len(table)
    n_conf = len(CONF_THRESHOLDS)

    fig = plt.figure(figsize=(16, 6 + 4 * min(n_ds, 3)))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2], hspace=0.4)

    # ── 상단: 막대그래프 ──
    ax_bar = fig.add_subplot(gs[0])
    ds_names = list(table.keys())
    x = np.arange(len(ds_names))
    bar_width = 0.12
    colors_conf = ['#1f77b4', '#ff7f0e', '#2ca02c']  # conf 0.10, 0.15, 0.25

    for ci, conf in enumerate(CONF_THRESHOLDS):
        single_totals = [table[n][conf]['single']['total'] for n in ds_names]
        multi_totals  = [table[n][conf]['multi']['total']  for n in ds_names]

        offset_s = (ci - n_conf / 2) * bar_width - bar_width / 2
        offset_m = offset_s + bar_width

        ax_bar.bar(x + offset_s, single_totals, bar_width * 0.9,
                   color=colors_conf[ci], alpha=0.5,
                   label=f'Single conf={conf:.2f}')
        ax_bar.bar(x + offset_m, multi_totals, bar_width * 0.9,
                   color=colors_conf[ci], alpha=0.9,
                   label=f'Multi  conf={conf:.2f}')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(ds_names, fontsize=11)
    ax_bar.set_ylabel('총 탐지 수 (raw+proc 합산)', fontsize=10)
    ax_bar.set_title('Phase A: Confidence Threshold별 탐지 수 비교 (단일 vs 다중 클래스 모델)',
                     fontsize=12, fontweight='bold')
    ax_bar.legend(fontsize=8, ncol=3, loc='upper right')
    ax_bar.axhline(0, color='k', linewidth=0.5)

    # 기대 클래스 레이블
    expected_map = {ds['name']: ds['expected'] for ds in datasets_loaded}
    for i, name in enumerate(ds_names):
        exp = expected_map.get(name, '?')
        color = '#00cc44' if exp == 'background' else '#ff6600'
        ax_bar.text(x[i], -0.8, f'expect:\n{exp}', ha='center', va='top',
                    fontsize=8, color=color)

    ax_bar.set_ylim(bottom=-1.5)
    ax_bar.grid(axis='y', alpha=0.3)

    # ── 하단: 대표 이미지 + conf=0.10 추론 ──
    n_show = min(n_ds, 4)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[1], wspace=0.05)

    for di, ds in enumerate(datasets_loaded[:n_show]):
        ax_img = fig.add_subplot(gs_bot[di])
        raw, proc, stem = ds['samples'][0]
        img = _bscan_to_img(proc)  # 전처리 버전 사용
        ax_img.imshow(img, cmap='gray', aspect='auto')

        # conf=0.10으로 다중 모델 추론
        dets = run_inference_at_conf(multi_model, img, conf=0.10)
        for cls_id, sc, xyxy in dets:
            x1, y1, x2, y2 = xyxy
            cname = CLASS_NAMES_MULTI[cls_id] if cls_id < len(CLASS_NAMES_MULTI) else '?'
            color = CLASS_COLORS.get(cname, 'white')
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
            ax_img.add_patch(rect)
            ax_img.text(x1, y1 - 3, f'{cname} {sc:.2f}',
                        color=color, fontsize=7, fontweight='bold')

        exp = expected_map.get(ds['name'], '?')
        n_det = len(dets)
        det_status = '✓' if (exp == 'background' and n_det == 0) else \
                     ('✓' if (exp != 'background' and n_det > 0) else '✗')
        ax_img.set_title(f"{ds['name']}\nexpect={exp} | det={n_det} {det_status}",
                         fontsize=8, color='white',
                         bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))
        ax_img.axis('off')

    plt.suptitle('Phase A: Confidence Threshold 실험 (conf=0.10 추론 예시 포함)',
                 fontsize=13, fontweight='bold', y=0.98)

    if save_path is None:
        save_path = OUTPUT_DIR / "confidence_threshold_experiment.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"\n  [저장] {save_path}")
    return save_path


def print_summary_table(table, datasets_loaded):
    """결과 요약 테이블 출력"""
    expected_map = {ds['name']: ds['expected'] for ds in datasets_loaded}

    print("\n" + "=" * 70)
    print("Phase A 결과 요약")
    print("=" * 70)
    print(f"{'데이터셋':<14} {'기대':<12} {'모델':<8}", end="")
    for conf in CONF_THRESHOLDS:
        print(f" conf={conf:.2f}", end="")
    print()
    print("-" * 70)

    for name in table:
        exp = expected_map.get(name, '?')
        for model_key, model_label in [('single', 'Single'), ('multi', 'Multi ')]:
            print(f"{name:<14} {exp:<12} {model_label:<8}", end="")
            for conf in CONF_THRESHOLDS:
                total = table[name][conf][model_key]['total']
                print(f"   {total:>5}  ", end="")
            print()
        print()

    print("=" * 70)
    print("해석:")
    print("  - background 기대 → 탐지 수 = 0이 정답 (FP 없음)")
    print("  - pipe/rebar 기대 → 탐지 수 > 0이 정답 (TP)")
    print("  - conf 낮출수록 recall↑ / precision↓")
    print("=" * 70)


def main():
    from ultralytics import YOLO

    print("Phase A: Confidence Threshold 실험")
    print(f"  단일 클래스 모델: {SINGLE_WEIGHTS}")
    print(f"  다중 클래스 모델: {MULTI_WEIGHTS}")
    print(f"  Conf 임계값: {CONF_THRESHOLDS}")

    # 모델 로드
    print("\n[1/3] 모델 로드...")
    single_model = YOLO(str(SINGLE_WEIGHTS))
    multi_model  = YOLO(str(MULTI_WEIGHTS))

    # 실측 데이터 로드
    print("[2/3] 실측 데이터 로드...")
    datasets_loaded = load_real_datasets()
    print(f"  로드된 데이터셋: {len(datasets_loaded)}개")
    for ds in datasets_loaded:
        print(f"    {ds['name']}: {len(ds['samples'])}개 파일, 기대={ds['expected']}")

    if not datasets_loaded:
        print("  [오류] 실측 데이터 없음 - 종료")
        return

    # 추론 실행
    print("[3/3] 추론 실행 (단일/다중 × 3 임계값)...")
    table = build_result_table(datasets_loaded, single_model, multi_model)

    # 결과 출력
    print_summary_table(table, datasets_loaded)

    # 시각화
    save_path = visualize_threshold_comparison(table, datasets_loaded)
    print(f"\n[완료] 결과 이미지: {save_path}")


if __name__ == "__main__":
    main()
