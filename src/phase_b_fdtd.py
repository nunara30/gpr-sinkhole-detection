"""
Phase B: gprMax FDTD 시뮬레이션

기존 해석적 합성 대신 실제 FDTD 전자파 시뮬레이션으로
더 현실적인 GPR B-scan 생성.

파이프라인:
  1. 컴팩트 .in 파일 생성 (sinkhole/pipe/rebar)
  2. gprMax 실행 → HDF5 .out 파일
  3. HDF5 파싱 → B-scan numpy array
  4. 자동 YOLO bbox 생성 + PNG 저장
  5. 해석적 vs FDTD 시각적 비교
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path

import numpy as np
import h5py
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 경로 설정 ──
sys.path.insert(0, str(Path(__file__).parent))
os.environ['PATH'] = r'C:\Users\jbcgl\miniconda3\envs\gpr_rag\Library\bin;' + os.environ['PATH']
sys.path.insert(0, str(Path(__file__).parent.parent / "gprMax"))

BASE_DIR   = Path("G:/RAG_system")
MODELS_DIR = BASE_DIR / "models"
FDTD_IN_DIR  = MODELS_DIR / "fdtd_compact"
FDTD_OUT_DIR = BASE_DIR / "data" / "gpr" / "fdtd_bscan"
OUTPUT_DIR   = BASE_DIR / "src" / "output" / "week4_multiclass"

FDTD_IN_DIR.mkdir(parents=True, exist_ok=True)
FDTD_OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

C0 = 299792458.0  # 빛의 속도 [m/s]


# ─────────────────────────────────────────────
# 1. .in 파일 생성
# ─────────────────────────────────────────────

def _time_window(depth_m: float, soil_epsr: float, margin: float = 1.4) -> float:
    """최대 깊이에 대한 왕복 시간 + 마진"""
    v = C0 / np.sqrt(soil_epsr)
    return 2 * depth_m / v * margin


def make_sinkhole_in(freq_hz: float, depth_m: float, radius_m: float,
                     soil_epsr: float, out_path: Path) -> dict:
    """
    컴팩트 sinkhole .in 파일 생성
    도메인: 2m × 1.8m, dx=0.01m
    """
    dx = 0.01
    domain_x = 2.0
    domain_y = max(depth_m * 1.6 + 0.3, 1.2)
    domain_z = dx

    # 공동 위치: 도메인 중앙
    cx, cy = domain_x / 2, depth_m
    tw = _time_window(depth_m + radius_m, soil_epsr)

    # 안테나 위치 (PML 경계 밖: 10셀 = 0.1m)
    ant_y = domain_y - 0.05  # 지표면

    content = f"""#title: FDTD_sinkhole_f{freq_hz/1e6:.0f}MHz_d{depth_m}m_r{radius_m}m_er{soil_epsr}
#domain: {domain_x} {domain_y:.2f} {domain_z}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {tw:.3e}

#material: {soil_epsr} 0.001 1 0 soil

#box: 0 0 0 {domain_x} {domain_y - 0.1:.2f} {domain_z} soil
#cylinder: {cx:.2f} {cy:.2f} 0  {cx:.2f} {cy:.2f} {domain_z}  {radius_m:.3f} free_space

#waveform: ricker 1 {freq_hz:.3e} src_wave
#hertzian_dipole: z 0.01 {ant_y:.2f} 0 src_wave
#rx: 0.02 {ant_y:.2f} 0

#src_steps: {dx} 0 0
#rx_steps: {dx} 0 0
"""
    out_path.write_text(content)
    n_traces = int((domain_x - 0.12) / dx)
    return {
        'object_type': 'sinkhole',
        'freq_hz': freq_hz,
        'depth_m': depth_m,
        'radius_m': radius_m,
        'soil_epsr': soil_epsr,
        'cx': cx,
        'cy': cy,
        'ant_y': ant_y,
        'domain_x': domain_x,
        'domain_y': domain_y,
        'dx': dx,
        'tw': tw,
        'n_traces': n_traces,
    }


def make_pipe_in(freq_hz: float, depth_m: float, pipe_diam: float,
                 soil_epsr: float, out_path: Path) -> dict:
    """컴팩트 pipe .in 파일 생성 (PEC 실린더)"""
    dx = 0.01
    domain_x = 2.0
    domain_y = max(depth_m * 1.6 + 0.3, 1.2)
    domain_z = dx

    cx, cy = domain_x / 2, depth_m
    pipe_r = pipe_diam / 2
    tw = _time_window(depth_m + pipe_r, soil_epsr)
    ant_y = domain_y - 0.05

    content = f"""#title: FDTD_pipe_f{freq_hz/1e6:.0f}MHz_d{depth_m}m_diam{pipe_diam}m_er{soil_epsr}
#domain: {domain_x} {domain_y:.2f} {domain_z}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {tw:.3e}

#material: {soil_epsr} 0.001 1 0 soil

#box: 0 0 0 {domain_x} {domain_y - 0.1:.2f} {domain_z} soil
#cylinder: {cx:.2f} {cy:.2f} 0  {cx:.2f} {cy:.2f} {domain_z}  {pipe_r:.3f} pec

#waveform: ricker 1 {freq_hz:.3e} src_wave
#hertzian_dipole: z 0.01 {ant_y:.2f} 0 src_wave
#rx: 0.02 {ant_y:.2f} 0

#src_steps: {dx} 0 0
#rx_steps: {dx} 0 0
"""
    out_path.write_text(content)
    n_traces = int((domain_x - 0.12) / dx)
    return {
        'object_type': 'pipe',
        'freq_hz': freq_hz,
        'depth_m': depth_m,
        'pipe_diameter': pipe_diam,
        'soil_epsr': soil_epsr,
        'cx': cx,
        'cy': cy,
        'ant_y': ant_y,
        'domain_x': domain_x,
        'domain_y': domain_y,
        'dx': dx,
        'tw': tw,
        'n_traces': n_traces,
    }


def make_rebar_in(freq_hz: float, depth_m: float, spacing_m: float, n_rebars: int,
                  soil_epsr: float, out_path: Path) -> dict:
    """컴팩트 rebar .in 파일 생성 (PEC 다중 실린더)"""
    dx = 0.01
    rebar_r = 0.008  # 8mm 반경

    total_width = (n_rebars - 1) * spacing_m + rebar_r * 2
    domain_x = max(total_width + 0.6, 2.0)
    domain_y = max(depth_m * 1.6 + 0.3, 1.0)
    domain_z = dx

    start_x = (domain_x - total_width) / 2 + rebar_r
    positions = [round(start_x + i * spacing_m, 4) for i in range(n_rebars)]
    ant_y = domain_y - 0.05
    tw = _time_window(depth_m + rebar_r, soil_epsr)

    cylinders = '\n'.join(
        f'#cylinder: {px:.4f} {depth_m:.3f} 0  {px:.4f} {depth_m:.3f} {domain_z}  {rebar_r} pec'
        for px in positions
    )

    content = f"""#title: FDTD_rebar_f{freq_hz/1e6:.0f}MHz_d{depth_m}m_sp{spacing_m}m_n{n_rebars}_er{soil_epsr}
#domain: {domain_x:.2f} {domain_y:.2f} {domain_z}
#dx_dy_dz: {dx} {dx} {dx}
#time_window: {tw:.3e}

#material: {soil_epsr} 0.001 1 0 soil

#box: 0 0 0 {domain_x:.2f} {domain_y - 0.1:.2f} {domain_z} soil
{cylinders}

#waveform: ricker 1 {freq_hz:.3e} src_wave
#hertzian_dipole: z 0.01 {ant_y:.2f} 0 src_wave
#rx: 0.02 {ant_y:.2f} 0

#src_steps: {dx} 0 0
#rx_steps: {dx} 0 0
"""
    out_path.write_text(content)
    n_traces = int((domain_x - 0.12) / dx)
    return {
        'object_type': 'rebar',
        'freq_hz': freq_hz,
        'depth_m': depth_m,
        'spacing_m': spacing_m,
        'n_rebars': n_rebars,
        'rebar_positions': positions,
        'soil_epsr': soil_epsr,
        'ant_y': ant_y,
        'domain_x': domain_x,
        'domain_y': domain_y,
        'dx': dx,
        'tw': tw,
        'n_traces': n_traces,
    }


# ─────────────────────────────────────────────
# 2. gprMax 실행
# ─────────────────────────────────────────────

def run_gprmax(in_path: Path, n_traces: int) -> Path:
    """
    gprMax 실행 → 첫 번째 .out 파일 경로 반환
    gprMax는 stem1.out, stem2.out ... stemN.out 을 생성함
    """
    from gprMax import run
    stem = in_path.with_suffix('')  # 확장자 제거된 경로
    first_out = stem.parent / f"{stem.name}1.out"

    if first_out.exists():
        print(f"    [스킵] 이미 존재: {first_out.name}")
        return stem  # stem 반환 (번호 없이)

    print(f"    실행: {in_path.name} (n={n_traces})")
    t0 = time.perf_counter()
    run(str(in_path), n=n_traces)
    elapsed = time.perf_counter() - t0
    print(f"    완료: {elapsed:.1f}초")
    return stem  # stem 반환


# ─────────────────────────────────────────────
# 3. HDF5 → B-scan
# ─────────────────────────────────────────────

def hdf5_to_bscan(stem_path: Path, n_traces: int, component: str = 'Ez') -> np.ndarray:
    """
    gprMax B-scan HDF5 파일들 → numpy B-scan (n_samples × n_traces)

    gprMax는 n_traces만큼 개별 .out 파일 생성:
      stem1.out, stem2.out, ..., stemN.out
    이를 합쳐서 B-scan을 구성.
    """
    traces = []
    for i in range(1, n_traces + 1):
        fpath = stem_path.parent / f"{stem_path.name}{i}.out"
        if not fpath.exists():
            break
        with h5py.File(str(fpath), 'r') as f:
            rx_keys = sorted(f['rxs'].keys())
            if not rx_keys:
                continue
            rx = f['rxs'][rx_keys[0]]
            if component in rx:
                trace = rx[component][:]
            else:
                comp = list(rx.keys())[0]
                trace = rx[comp][:]
            traces.append(trace)

    if not traces:
        raise ValueError(f"No traces found for {stem_path}")

    return np.column_stack(traces).astype(np.float32)  # (n_samples, n_traces)


# ─────────────────────────────────────────────
# 4. YOLO bbox 계산
# ─────────────────────────────────────────────

def compute_fdtd_bbox(meta: dict, bscan: np.ndarray) -> list:
    """
    FDTD 메타데이터 + B-scan shape로 YOLO bbox 계산
    반환: [(class_id, cx_norm, cy_norm, w_norm, h_norm)]
    """
    n_samples, n_traces = bscan.shape
    obj_type = meta['object_type']

    if obj_type == 'background':
        return []

    dx = meta['dx']
    domain_x = meta['domain_x']
    domain_y = meta['domain_y']
    ant_y = meta['ant_y']
    tw = meta['tw']
    soil_epsr = meta['soil_epsr']
    v = C0 / np.sqrt(soil_epsr)

    if obj_type == 'sinkhole':
        cx_m = meta['cx']
        depth_m = meta['depth_m']
        radius_m = meta['radius_m']

        # 쌍곡선 최정점: (cx_m, depth의 왕복 시간)
        t_apex = 2 * depth_m / v
        t_apex_norm = t_apex / tw
        cx_trace = (cx_m - 0.01) / (domain_x - 0.12)

        # 박스 폭: 쌍곡선 확산 (ant_y 높이에서 radius 기준)
        dist_horiz = np.sqrt((radius_m * 2) ** 2 + depth_m ** 2)
        t_edge = 2 * dist_horiz / v
        dt_half = (t_edge - t_apex) * 0.5
        w_norm = min(radius_m * 4 / (domain_x - 0.12), 0.8)
        h_norm = min((2 * dt_half + tw * 0.03) / tw, 0.6)

        return [(0, cx_trace, t_apex_norm, w_norm, h_norm)]

    elif obj_type == 'pipe':
        cx_m = meta['cx']
        depth_m = meta['depth_m']
        pipe_r = meta['pipe_diameter'] / 2

        t_apex = 2 * depth_m / v
        t_apex_norm = t_apex / tw
        cx_trace = (cx_m - 0.01) / (domain_x - 0.12)
        w_norm = min(pipe_r * 6 / (domain_x - 0.12), 0.6)
        h_norm = min(tw * 0.08 / tw, 0.3)
        return [(1, cx_trace, t_apex_norm, w_norm, h_norm)]

    elif obj_type == 'rebar':
        positions = meta['rebar_positions']
        depth_m = meta['depth_m']
        rebar_r = 0.008

        t_apex = 2 * depth_m / v
        t_apex_norm = t_apex / tw

        x_min_m = positions[0] - rebar_r * 5
        x_max_m = positions[-1] + rebar_r * 5
        cx_m = (x_min_m + x_max_m) / 2
        cx_trace = (cx_m - 0.01) / (domain_x - 0.12)
        w_norm = min((x_max_m - x_min_m) / (domain_x - 0.12), 0.95)
        h_norm = min(tw * 0.12 / tw, 0.3)
        return [(2, cx_trace, t_apex_norm, w_norm, h_norm)]

    return []


# ─────────────────────────────────────────────
# 5. 데이터셋 저장
# ─────────────────────────────────────────────

def bscan_to_png(bscan: np.ndarray, out_path: Path):
    """B-scan → 640×640 grayscale PNG"""
    data = bscan.astype(np.float32)
    p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
    if p98 - p2 < 1e-10:
        img = np.zeros((640, 640), dtype=np.uint8)
    else:
        img = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    img_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(out_path), img_resized)


def save_yolo_label(label_path: Path, bboxes: list):
    """YOLO 라벨 파일 저장"""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, 'w') as f:
        for cls_id, cx, cy, w, h in bboxes:
            # 경계 클리핑
            cx = np.clip(cx, 0.01, 0.99)
            cy = np.clip(cy, 0.01, 0.99)
            w  = np.clip(w,  0.01, 0.99)
            h  = np.clip(h,  0.01, 0.99)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────────
# 6. 비교 시각화
# ─────────────────────────────────────────────

def visualize_fdtd_vs_analytical(fdtd_results: list, save_path: Path):
    """FDTD B-scan vs 해석적 합성 B-scan 비교"""
    from week3_simulation import synthesize_bscan, synthesize_pipe_bscan, synthesize_rebar_bscan

    n = len(fdtd_results)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), facecolor='#1a1a2e')

    for i, res in enumerate(fdtd_results):
        meta = res['meta']
        fdtd_bscan = res['bscan']
        obj_type = meta['object_type']

        # 해석적 합성
        try:
            if obj_type == 'sinkhole':
                analytical_bscan, t, x, _ = synthesize_bscan(
                    freq_hz=meta['freq_hz'],
                    depth_m=meta['depth_m'],
                    radius_m=meta['radius_m'],
                    soil_epsr=meta['soil_epsr'],
                )
            elif obj_type == 'pipe':
                analytical_bscan, t, x, _ = synthesize_pipe_bscan(
                    freq_hz=meta['freq_hz'],
                    depth_m=meta['depth_m'],
                    pipe_diameter=meta['pipe_diameter'],
                    soil_epsr=meta['soil_epsr'],
                )
            elif obj_type == 'rebar':
                analytical_bscan, t, x, _ = synthesize_rebar_bscan(
                    freq_hz=meta['freq_hz'],
                    depth_m=meta['depth_m'],
                    spacing_m=meta['spacing_m'],
                    n_rebars=meta['n_rebars'],
                    soil_epsr=meta['soil_epsr'],
                )
            else:
                analytical_bscan = None
        except Exception as e:
            print(f"  [해석적 합성 실패] {e}")
            analytical_bscan = None

        ax_fdtd = axes[i][0] if n > 1 else axes[0]
        ax_analytical = axes[i][1] if n > 1 else axes[1]

        # FDTD
        fdtd_norm = fdtd_bscan.copy()
        p2, p98 = np.percentile(fdtd_norm, 2), np.percentile(fdtd_norm, 98)
        fdtd_norm = np.clip(fdtd_norm, p2, p98)
        ax_fdtd.imshow(fdtd_norm, cmap='seismic', aspect='auto',
                       vmin=p2, vmax=p98)
        ax_fdtd.set_title(f"FDTD: {obj_type} "
                          f"f={meta['freq_hz']/1e6:.0f}MHz "
                          f"d={meta['depth_m']}m",
                          color='white', fontsize=9)
        ax_fdtd.set_xlabel("Trace #", color='white', fontsize=8)
        ax_fdtd.set_ylabel("Time sample", color='white', fontsize=8)
        ax_fdtd.tick_params(colors='white')

        # 해석적
        if analytical_bscan is not None:
            p2a, p98a = np.percentile(analytical_bscan, 2), np.percentile(analytical_bscan, 98)
            a_norm = np.clip(analytical_bscan, p2a, p98a)
            ax_analytical.imshow(a_norm, cmap='seismic', aspect='auto',
                                  vmin=p2a, vmax=p98a)
            ax_analytical.set_title(f"Analytical: {obj_type}",
                                    color='white', fontsize=9)
        else:
            ax_analytical.text(0.5, 0.5, 'N/A', transform=ax_analytical.transAxes,
                               ha='center', va='center', color='white', fontsize=14)
        ax_analytical.set_xlabel("Trace #", color='white', fontsize=8)
        ax_analytical.tick_params(colors='white')

        for ax in [ax_fdtd, ax_analytical]:
            ax.spines[:].set_color('#444')

    plt.suptitle('Phase B: FDTD vs 해석적 합성 B-scan 비교',
                 fontsize=12, fontweight='bold', color='white', y=0.99)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"\n  [저장] {save_path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

# 실행할 시나리오 (Phase B 시연용 - 각 클래스 2~3개)
FDTD_SCENARIOS = [
    # (타입, (파라미터들))
    ('sinkhole', dict(freq_hz=900e6, depth_m=0.5, radius_m=0.2, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=900e6, depth_m=0.8, radius_m=0.3, soil_epsr=6)),
    ('pipe',     dict(freq_hz=900e6, depth_m=0.5, pipe_diam=0.1, soil_epsr=6)),
    ('pipe',     dict(freq_hz=900e6, depth_m=0.8, pipe_diam=0.2, soil_epsr=6)),
    ('rebar',    dict(freq_hz=900e6, depth_m=0.15, spacing_m=0.15, n_rebars=5, soil_epsr=6)),
    ('rebar',    dict(freq_hz=900e6, depth_m=0.20, spacing_m=0.20, n_rebars=3, soil_epsr=6)),
]


def main():
    print("Phase B: gprMax FDTD 시뮬레이션")
    print(f"  시나리오 수: {len(FDTD_SCENARIOS)}")
    print(f"  .in 저장: {FDTD_IN_DIR}")
    print(f"  B-scan 저장: {FDTD_OUT_DIR}")

    fdtd_results = []

    for scenario_type, params in FDTD_SCENARIOS:
        name_parts = [f"{k}{v}" for k, v in params.items()]
        stem = f"fdtd_{scenario_type}_{'_'.join(name_parts[:3])}"
        stem = stem.replace('.', 'p').replace(' ', '')

        in_path = FDTD_IN_DIR / f"{stem}.in"

        # .in 파일 생성
        print(f"\n[시나리오] {stem}")
        if scenario_type == 'sinkhole':
            meta = make_sinkhole_in(**params, out_path=in_path)
        elif scenario_type == 'pipe':
            meta = make_pipe_in(**params, out_path=in_path)
        elif scenario_type == 'rebar':
            meta = make_rebar_in(**params, out_path=in_path)
        else:
            continue

        n_traces = meta['n_traces']
        print(f"  .in 생성 → n_traces={n_traces}, domain={meta['domain_x']}×{meta['domain_y']:.2f}m")

        # gprMax 실행
        in_stem = FDTD_IN_DIR / stem  # 확장자 없는 경로 (stem)
        try:
            run_gprmax(in_path, n_traces)
        except Exception as e:
            print(f"  [오류] gprMax 실행 실패: {e}")
            continue

        # 첫 트레이스 파일 존재 확인
        first_out = FDTD_IN_DIR / f"{stem}1.out"
        if not first_out.exists():
            print(f"  [오류] .out 파일 없음: {first_out}")
            continue

        # HDF5 → B-scan (stem1.out ... stemN.out 합치기)
        bscan = hdf5_to_bscan(in_stem, n_traces)
        print(f"  B-scan shape: {bscan.shape}")

        # YOLO bbox 계산
        bboxes = compute_fdtd_bbox(meta, bscan)
        print(f"  YOLO bbox: {bboxes}")

        # PNG 저장
        png_path = FDTD_OUT_DIR / f"{stem}.png"
        bscan_to_png(bscan, png_path)

        # YOLO 라벨 저장
        label_path = FDTD_OUT_DIR / f"{stem}.txt"
        save_yolo_label(label_path, bboxes)

        # 메타 저장
        meta_path = FDTD_OUT_DIR / f"{stem}_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        fdtd_results.append({'meta': meta, 'bscan': bscan, 'stem': stem})
        print(f"  저장: {png_path.name}")

    # 비교 시각화
    if fdtd_results:
        compare_path = OUTPUT_DIR / "fdtd_vs_analytical.png"
        print(f"\n비교 시각화 생성...")
        visualize_fdtd_vs_analytical(fdtd_results, compare_path)

    # 결과 요약
    print(f"\n{'='*60}")
    print(f"Phase B 완료")
    print(f"  FDTD B-scan 생성: {len(fdtd_results)}개")
    print(f"  PNG + 라벨: {FDTD_OUT_DIR}")
    print(f"  비교 시각화: {OUTPUT_DIR / 'fdtd_vs_analytical.png'}")
    print(f"{'='*60}")

    return fdtd_results


if __name__ == "__main__":
    main()
