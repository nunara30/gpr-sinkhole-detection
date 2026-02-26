"""
Week 3 - GPR 싱크홀 시뮬레이션 (gprMax .in 생성 + 해석적 합성 B-scan)

gprMax FDTD 빌드 불가 시 해석적 방법으로 합성 B-scan 생성:
  - Ricker wavelet convolution
  - Diffraction hyperbola: t(x) = 2/v * sqrt((x-x0)^2 + d^2)
  - Direct wave, ground reflection, sinkhole reflection

48 시나리오: 2(freq) × 4(depth) × 3(radius) × 2(soil)
"""

import sys
import time
import json
import itertools
import subprocess
from pathlib import Path

import numpy as np
import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# week2 import
sys.path.insert(0, str(Path(__file__).parent))
from week2_preprocessing import (
    dc_removal, dewow, background_removal, bandpass_filter,
    gain_sec, fk_migration, run_pipeline, estimate_dt,
    plot_preprocessing_comparison,
)
from week2_database import GPRDatabase


# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("G:/RAG_system")
MODELS_DIR = BASE_DIR / "models"
SYNTHETIC_DIR = BASE_DIR / "data" / "gpr" / "synthetic"
OUTPUT_DIR = BASE_DIR / "src" / "output"

for d in [MODELS_DIR, SYNTHETIC_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 시뮬레이션 파라미터
# ─────────────────────────────────────────────
SCENARIOS = {
    'freq_hz': [400e6, 900e6],          # 안테나 주파수
    'depth_m': [0.5, 1.0, 1.5, 2.0],   # 싱크홀 중심 깊이
    'radius_m': [0.2, 0.5, 1.0],       # 싱크홀 반경
    'soil_epsr': [6, 12],              # 토양 비유전율 (건조/습윤)
}

# 도메인 설정
DOMAIN_X = 5.0   # m
DOMAIN_Z = 3.0   # m (깊이)
DX = 0.01        # 트레이스 간격 (10mm)
N_TRACES = int(DOMAIN_X / DX)  # 500 traces
SINKHOLE_X = DOMAIN_X / 2  # 싱크홀 중심 x위치 (도메인 중앙)

# 전자기파 속도 계산
C0 = 0.2998  # m/ns (빛의 속도)


def soil_velocity(epsr):
    """비유전율에서 전자기파 속도 계산 (m/ns)"""
    return C0 / np.sqrt(epsr)


# ─────────────────────────────────────────────
# 1. gprMax .in 파일 생성
# ─────────────────────────────────────────────

def generate_sinkhole_in(freq_hz, depth_m, radius_m, soil_epsr,
                          domain_x=5.0, domain_z=3.0,
                          dx_dy_dz=0.005, output_dir=None):
    """
    gprMax .in 파일 생성 (싱크홀 = 공기 채움 원형 공동)

    Returns: .in 파일 경로
    """
    if output_dir is None:
        output_dir = MODELS_DIR

    freq_mhz = freq_hz / 1e6
    soil_sigma = 0.001 if soil_epsr <= 8 else 0.01  # 건조/습윤
    soil_name = f"soil_er{soil_epsr}"

    # 시간창 계산 (왕복 시간 + 여유)
    v = soil_velocity(soil_epsr)  # m/ns
    max_depth = domain_z
    tw_ns = 2 * max_depth / v * 1.5  # 1.5배 여유
    tw_s = tw_ns * 1e-9

    # 트레이스 수 (도메인 폭 / 스텝)
    src_step = DX
    n_scans = int(domain_x / src_step)

    # 안테나 offset (송수신 간격)
    ant_offset = 0.06  # 6cm

    # 파일명
    fname = (f"sinkhole_f{freq_mhz:.0f}MHz_d{depth_m:.1f}m_"
             f"r{radius_m:.1f}m_er{soil_epsr}.in")
    fpath = Path(output_dir) / fname

    # 표면 높이 (지표면 = domain_z - 0.5m padding)
    surface_y = domain_z - 0.5

    lines = [
        f"#title: Sinkhole f={freq_mhz:.0f}MHz d={depth_m:.1f}m "
        f"r={radius_m:.1f}m er={soil_epsr}",
        f"#domain: {domain_x} {domain_z} {dx_dy_dz}",
        f"#dx_dy_dz: {dx_dy_dz} {dx_dy_dz} {dx_dy_dz}",
        f"#time_window: {tw_s:.2e}",
        "",
        f"#material: {soil_epsr} {soil_sigma} 1 0 {soil_name}",
        "",
        f"#box: 0 0 0 {domain_x} {surface_y} {dx_dy_dz} {soil_name}",
        f"#cylinder: {SINKHOLE_X} {surface_y - depth_m} 0  "
        f"{SINKHOLE_X} {surface_y - depth_m} {dx_dy_dz}  "
        f"{radius_m} free_space",
        "",
        f"#waveform: ricker 1 {freq_hz:.2e} my_ricker",
        f"#hertzian_dipole: z {src_step} {surface_y} 0 my_ricker",
        f"#rx: {src_step + ant_offset} {surface_y} 0",
        "",
        f"#src_steps: {src_step} 0 0",
        f"#rx_steps: {src_step} 0 0",
    ]

    fpath.write_text('\n'.join(lines), encoding='utf-8')
    return fpath


def generate_all_in_files():
    """48개 시나리오 .in 파일 일괄 생성"""
    files = []
    combos = list(itertools.product(
        SCENARIOS['freq_hz'],
        SCENARIOS['depth_m'],
        SCENARIOS['radius_m'],
        SCENARIOS['soil_epsr'],
    ))

    for freq_hz, depth_m, radius_m, soil_epsr in combos:
        fpath = generate_sinkhole_in(freq_hz, depth_m, radius_m, soil_epsr)
        files.append({
            'path': str(fpath),
            'freq_hz': freq_hz,
            'depth_m': depth_m,
            'radius_m': radius_m,
            'soil_epsr': soil_epsr,
        })

    return files


# ─────────────────────────────────────────────
# 2. gprMax 실행 (설치되어 있을 때)
# ─────────────────────────────────────────────

def run_gprmax_simulation(in_file, n_scans=500):
    """gprMax 실행 (subprocess)"""
    cmd = [
        sys.executable, '-m', 'gprMax', str(in_file), '-n', str(n_scans)
    ]
    print(f"  실행: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  [오류] {result.stderr[:200]}")
            return False
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [오류] gprMax 실행 실패: {e}")
        return False


def merge_output_files(in_file, n_scans, remove_files=False):
    """gprMax A-scan 출력 → B-scan HDF5 병합"""
    try:
        from tools.outputfiles_merge import merge_files
        merge_files(str(in_file), removefiles=remove_files)
        return True
    except ImportError:
        # gprMax tools 미설치 시 수동 병합
        return _manual_merge(in_file, n_scans)


def _manual_merge(in_file, n_scans):
    """HDF5 A-scan 수동 병합"""
    base = Path(in_file).stem
    parent = Path(in_file).parent

    out_files = sorted(parent.glob(f"{base}*[0-9].out"))
    if not out_files:
        return False

    # 첫 파일에서 메타데이터 읽기
    with h5py.File(str(out_files[0]), 'r') as f:
        n_iterations = f.attrs['Iterations']

    # 병합
    bscan = np.zeros((n_iterations, len(out_files)), dtype=np.float64)
    for i, of in enumerate(out_files):
        with h5py.File(str(of), 'r') as f:
            bscan[:, i] = f['rxs']['rx1']['Ez'][:]

    # 병합 파일 저장
    merged_path = parent / f"{base}_merged.out"
    with h5py.File(str(merged_path), 'w') as f:
        f.attrs['Iterations'] = n_iterations
        f.attrs['n_traces'] = len(out_files)
        f.create_dataset('bscan_Ez', data=bscan)

    return True


def load_gprmax_output(out_file, rx_component='Ez'):
    """gprMax HDF5 출력에서 B-scan 추출"""
    with h5py.File(str(out_file), 'r') as f:
        if 'bscan_Ez' in f:
            return f['bscan_Ez'][:]
        # 단일 A-scan → B-scan 구조
        if 'rxs' in f:
            return f['rxs']['rx1'][rx_component][:]
    return None


# ─────────────────────────────────────────────
# 3. 해석적 B-scan 합성 (gprMax 대안)
# ─────────────────────────────────────────────

def ricker_wavelet(t, freq_hz):
    """
    Ricker (Mexican hat) wavelet
    t: 시간 배열 (초)
    freq_hz: 중심 주파수 (Hz)
    """
    tau = t - 1.5 / freq_hz  # 지연
    a = (np.pi * freq_hz * tau) ** 2
    return (1 - 2 * a) * np.exp(-a)


def synthesize_bscan(freq_hz, depth_m, radius_m, soil_epsr,
                      domain_x=5.0, domain_z=3.0, dx=0.01,
                      n_samples=512, sinkhole_x=None,
                      add_noise=True):
    """
    해석적 GPR B-scan 합성

    물리 모델:
    1. Direct wave (air): t = x_offset / c0
    2. Ground reflection: t = 2 * antenna_height / c0
    3. Sinkhole cavity (diffraction hyperbola):
       t(x) = 2/v * sqrt((x - x0)^2 + d^2)
       여기서 공기-토양 경계의 반사파가 쌍곡선 형태로 나타남

    Returns: (bscan, time_axis, x_axis, metadata)
    """
    if sinkhole_x is None:
        sinkhole_x = domain_x / 2

    v = soil_velocity(soil_epsr)  # m/ns
    v_m_s = v * 1e9  # m/s

    # 시간 설정
    max_tw_ns = 2 * domain_z / v * 1.2
    dt_ns = max_tw_ns / n_samples
    dt_s = dt_ns * 1e-9

    time_axis = np.arange(n_samples) * dt_s  # 초
    time_ns = np.arange(n_samples) * dt_ns

    # 공간 설정
    n_traces = int(domain_x / dx)
    x_axis = np.arange(n_traces) * dx  # m

    # Ricker wavelet 템플릿
    t_wavelet = np.arange(-3.0 / freq_hz, 3.0 / freq_hz, dt_s)
    wavelet = ricker_wavelet(t_wavelet + 3.0 / freq_hz, freq_hz)

    bscan = np.zeros((n_samples, n_traces), dtype=np.float64)

    # ── 1. Direct wave (공기 중 직접파) ──
    direct_t_s = 0.06 / (C0 * 1e9)  # 안테나 간격 6cm, 공기 속도
    direct_sample = int(direct_t_s / dt_s)
    if 0 <= direct_sample < n_samples:
        bscan[direct_sample, :] += 1.0  # 전 트레이스 동일

    # ── 2. Ground reflection (지표면 반사) ──
    # 약한 반사 (비유전율 차이에 비례)
    r_ground = (np.sqrt(soil_epsr) - 1) / (np.sqrt(soil_epsr) + 1)
    ground_t_s = 2 * 0.001 / (C0 * 1e9)  # 지표면 바로 아래
    ground_sample = max(direct_sample + 2, int(ground_t_s / dt_s))
    if 0 <= ground_sample < n_samples:
        bscan[ground_sample, :] += r_ground * 0.8

    # ── 3. 수평 지층 반사 (약한 배경) ──
    for layer_d in [0.8, 1.5, 2.2]:
        layer_t = 2 * layer_d / v_m_s
        layer_sample = int(layer_t / dt_s)
        if 0 <= layer_sample < n_samples:
            amp = 0.05 * np.random.uniform(0.5, 1.5)
            bscan[layer_sample, :] += amp

    # ── 4. 싱크홀 diffraction hyperbola ──
    # 반사 계수: 토양 → 공기 경계
    r_sinkhole = (1 - np.sqrt(soil_epsr)) / (1 + np.sqrt(soil_epsr))

    for ix in range(n_traces):
        x = x_axis[ix]

        # 공동 상부 (가장 강한 반사)
        d_top = depth_m - radius_m
        if d_top > 0:
            # 쌍곡선: t = 2/v * sqrt((x-x0)^2 + d_top^2)
            dist = np.sqrt((x - sinkhole_x) ** 2 + d_top ** 2)
            t_reflect = 2 * dist / v_m_s
            sample = int(t_reflect / dt_s)
            if 0 <= sample < n_samples:
                # 진폭: 거리 감쇠 + 반사계수
                amp = abs(r_sinkhole) * (d_top / dist) ** 0.5
                bscan[sample, ix] += amp

        # 공동 하부 (약한 반사)
        d_bottom = depth_m + radius_m
        dist_b = np.sqrt((x - sinkhole_x) ** 2 + d_bottom ** 2)
        t_bottom = 2 * dist_b / v_m_s
        sample_b = int(t_bottom / dt_s)
        if 0 <= sample_b < n_samples:
            amp_b = abs(r_sinkhole) * 0.3 * (d_bottom / dist_b) ** 0.5
            bscan[sample_b, ix] -= amp_b  # 극성 반전

        # 공동 측면 (원형에 의한 산란)
        for angle in np.linspace(-np.pi / 2, np.pi / 2, 12):
            scatter_x = sinkhole_x + radius_m * np.cos(angle)
            scatter_z = depth_m + radius_m * np.sin(angle)
            if scatter_z > 0:
                dist_s = np.sqrt((x - scatter_x) ** 2 + scatter_z ** 2)
                t_s = 2 * dist_s / v_m_s
                sample_s = int(t_s / dt_s)
                if 0 <= sample_s < n_samples:
                    amp_s = abs(r_sinkhole) * 0.08 * (scatter_z / dist_s) ** 0.5
                    bscan[sample_s, ix] += amp_s

    # ── 5. Ricker wavelet convolution ──
    from scipy.signal import fftconvolve
    for ix in range(n_traces):
        trace = bscan[:, ix]
        convolved = fftconvolve(trace, wavelet, mode='full')
        # 중앙 정렬
        offset = len(wavelet) // 2
        bscan[:, ix] = convolved[offset:offset + n_samples]

    # ── 6. 노이즈 추가 ──
    if add_noise:
        signal_rms = np.sqrt(np.mean(bscan ** 2))
        noise_level = signal_rms * 0.05  # SNR ~20dB
        noise = np.random.randn(n_samples, n_traces) * noise_level
        bscan += noise

    metadata = {
        'freq_hz': freq_hz,
        'freq_mhz': freq_hz / 1e6,
        'depth_m': depth_m,
        'radius_m': radius_m,
        'soil_epsr': soil_epsr,
        'velocity_m_ns': v,
        'domain_x': domain_x,
        'domain_z': domain_z,
        'dx': dx,
        'n_samples': n_samples,
        'n_traces': n_traces,
        'dt_ns': dt_ns,
        'time_window_ns': max_tw_ns,
        'sinkhole_x': sinkhole_x,
        'method': 'analytical_synthesis',
    }

    return bscan.astype(np.float32), time_ns, x_axis, metadata


def synthesize_no_sinkhole(soil_epsr, freq_hz, domain_x=5.0, domain_z=3.0,
                            dx=0.01, n_samples=512):
    """싱크홀 없는 배경 B-scan (비교용)"""
    return synthesize_bscan(
        freq_hz=freq_hz,
        depth_m=100,  # 도메인 밖 (사실상 없음)
        radius_m=0.01,
        soil_epsr=soil_epsr,
        domain_x=domain_x,
        domain_z=domain_z,
        dx=dx,
        n_samples=n_samples,
        add_noise=True,
    )


# ─────────────────────────────────────────────
# 4. 일괄 시뮬레이션 실행
# ─────────────────────────────────────────────

def run_all_simulations(use_gprmax=False, db=None):
    """
    48개 시나리오 일괄 실행

    use_gprmax: True → gprMax FDTD, False → 해석적 합성
    Returns: list of result dicts
    """
    combos = list(itertools.product(
        SCENARIOS['freq_hz'],
        SCENARIOS['depth_m'],
        SCENARIOS['radius_m'],
        SCENARIOS['soil_epsr'],
    ))

    results = []
    total = len(combos)

    print(f"\n총 {total}개 시나리오 실행 "
          f"({'gprMax FDTD' if use_gprmax else '해석적 합성'})")
    print("─" * 60)

    for i, (freq_hz, depth_m, radius_m, soil_epsr) in enumerate(combos):
        freq_mhz = freq_hz / 1e6
        label = (f"f{freq_mhz:.0f}_d{depth_m:.1f}_"
                 f"r{radius_m:.1f}_er{soil_epsr}")

        print(f"  [{i+1:2d}/{total}] {label}...", end="", flush=True)
        t0 = time.perf_counter()

        if use_gprmax:
            # gprMax FDTD 실행
            in_file = MODELS_DIR / f"sinkhole_{label}MHz.in"
            if not in_file.exists():
                generate_sinkhole_in(freq_hz, depth_m, radius_m, soil_epsr)
            success = run_gprmax_simulation(in_file, n_scans=N_TRACES)
            if success:
                merge_output_files(in_file, N_TRACES)
                bscan = load_gprmax_output(
                    in_file.with_suffix('.out'))
                metadata = {'method': 'gprMax_FDTD'}
            else:
                print(" FAILED → 해석적 대체")
                bscan, _, _, metadata = synthesize_bscan(
                    freq_hz, depth_m, radius_m, soil_epsr)
        else:
            # 해석적 합성
            bscan, time_ns, x_axis, metadata = synthesize_bscan(
                freq_hz, depth_m, radius_m, soil_epsr)

        elapsed = time.perf_counter() - t0

        # .npy 저장
        npy_path = SYNTHETIC_DIR / f"{label}.npy"
        np.save(str(npy_path), bscan)

        # 메타데이터 JSON 저장
        meta_path = SYNTHETIC_DIR / f"{label}_meta.json"
        meta_save = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                     for k, v in metadata.items()}
        meta_save['elapsed_s'] = round(elapsed, 3)
        meta_path.write_text(json.dumps(meta_save, indent=2), encoding='utf-8')

        # DB 기록
        if db:
            ds_id = db.register_dataset(
                name=f"Synthetic: {label}",
                file_path=str(npy_path),
                data=bscan,
                format="synthetic_npy",
                frequency_mhz=freq_mhz,
                time_window_ns=metadata.get('time_window_ns', 0),
                dx_m=metadata.get('dx', DX),
            )
            metadata['db_dataset_id'] = ds_id

        results.append({
            'label': label,
            'npy_path': str(npy_path),
            'meta_path': str(meta_path),
            'shape': bscan.shape,
            'elapsed_s': round(elapsed, 3),
            'metadata': metadata,
        })

        print(f" {bscan.shape} {elapsed:.2f}s")

    print("─" * 60)
    print(f"완료: {len(results)}개 시나리오, "
          f"총 {sum(r['elapsed_s'] for r in results):.1f}s")

    return results


# ─────────────────────────────────────────────
# 5. 시각화
# ─────────────────────────────────────────────

def plot_simulation_comparison(results, save_path=None):
    """
    파라미터별 B-scan 격자 비교

    행: depth (0.5, 1.0, 1.5, 2.0m)
    열: radius (0.2, 0.5, 1.0m)
    고정: freq=900MHz, soil_epsr=6
    """
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    fig.suptitle('Sinkhole B-scan: Depth × Radius (900MHz, εr=6)',
                 fontsize=16, fontweight='bold')

    for i, depth in enumerate(SCENARIOS['depth_m']):
        for j, radius in enumerate(SCENARIOS['radius_m']):
            label = f"f900_d{depth:.1f}_r{radius:.1f}_er6"
            npy_path = SYNTHETIC_DIR / f"{label}.npy"

            ax = axes[i, j]
            if npy_path.exists():
                bscan = np.load(str(npy_path))
                vmax = np.percentile(np.abs(bscan), 98)
                if vmax == 0:
                    vmax = 1
                ax.imshow(bscan, aspect='auto', cmap='seismic',
                          vmin=-vmax, vmax=vmax,
                          extent=[0, DOMAIN_X, bscan.shape[0], 0])
                ax.set_title(f'd={depth}m, r={radius}m', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                        ha='center', va='center')
                ax.set_title(f'd={depth}m, r={radius}m', fontsize=10)

            if j == 0:
                ax.set_ylabel('Sample')
            if i == 3:
                ax.set_xlabel('Position (m)')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_sinkhole_detection(freq_hz=900e6, soil_epsr=6, depth_m=1.0,
                             radius_m=0.5, save_path=None):
    """싱크홀 유/무 비교 (동일 조건)"""
    # 싱크홀 있음
    bscan_with, _, _, meta_with = synthesize_bscan(
        freq_hz, depth_m, radius_m, soil_epsr)

    # 싱크홀 없음
    bscan_without, _, _, meta_without = synthesize_no_sinkhole(
        soil_epsr, freq_hz)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title in [
        (ax1, bscan_without, 'No Sinkhole (background)'),
        (ax2, bscan_with, f'Sinkhole d={depth_m}m r={radius_m}m'),
    ]:
        vmax = np.percentile(np.abs(data), 98)
        if vmax == 0:
            vmax = 1
        ax.imshow(data, aspect='auto', cmap='seismic',
                  vmin=-vmax, vmax=vmax,
                  extent=[0, DOMAIN_X, data.shape[0], 0])
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Sample')

    fig.suptitle(f'Sinkhole Detection Comparison '
                 f'({freq_hz/1e6:.0f}MHz, εr={soil_epsr})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_frequency_comparison(save_path=None):
    """400MHz vs 900MHz B-scan 비교 (동일 싱크홀)"""
    depth, radius, epsr = 1.0, 0.5, 6

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, freq_hz in zip(axes, [400e6, 900e6]):
        label = f"f{freq_hz/1e6:.0f}_d{depth:.1f}_r{radius:.1f}_er{epsr}"
        npy_path = SYNTHETIC_DIR / f"{label}.npy"
        if npy_path.exists():
            bscan = np.load(str(npy_path))
        else:
            bscan, _, _, _ = synthesize_bscan(freq_hz, depth, radius, epsr)

        vmax = np.percentile(np.abs(bscan), 98)
        if vmax == 0:
            vmax = 1
        ax.imshow(bscan, aspect='auto', cmap='seismic',
                  vmin=-vmax, vmax=vmax,
                  extent=[0, DOMAIN_X, bscan.shape[0], 0])
        ax.set_title(f'{freq_hz/1e6:.0f} MHz', fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Sample')

    fig.suptitle(f'Frequency Comparison (d={depth}m, r={radius}m, εr={epsr})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_soil_comparison(save_path=None):
    """건조(εr=6) vs 습윤(εr=12) 비교"""
    freq, depth, radius = 900e6, 1.0, 0.5

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, epsr in zip(axes, [6, 12]):
        label = f"f{freq/1e6:.0f}_d{depth:.1f}_r{radius:.1f}_er{epsr}"
        npy_path = SYNTHETIC_DIR / f"{label}.npy"
        if npy_path.exists():
            bscan = np.load(str(npy_path))
        else:
            bscan, _, _, _ = synthesize_bscan(freq, depth, radius, epsr)

        vmax = np.percentile(np.abs(bscan), 98)
        if vmax == 0:
            vmax = 1
        ax.imshow(bscan, aspect='auto', cmap='seismic',
                  vmin=-vmax, vmax=vmax,
                  extent=[0, DOMAIN_X, bscan.shape[0], 0])
        v = soil_velocity(epsr)
        ax.set_title(f'εr={epsr} ({"Dry" if epsr==6 else "Wet"}, '
                     f'v={v:.3f} m/ns)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Sample')

    fig.suptitle(f'Soil Permittivity Comparison '
                 f'(900MHz, d={depth}m, r={radius}m)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# 6. 전처리 적용 + DB 기록
# ─────────────────────────────────────────────

def apply_preprocessing_to_synthetic(bscan, metadata, db=None, dataset_id=None):
    """
    합성 B-scan에 Week 2 파이프라인 적용

    Returns: (processed, intermediates, steps_log)
    """
    freq_mhz = metadata['freq_mhz']
    dt_ns = metadata['dt_ns']
    dt_s = dt_ns * 1e-9
    dx = metadata['dx']

    # 주파수 기반 bandpass 범위 자동 설정
    bp_low = freq_mhz * 0.25
    bp_high = freq_mhz * 2.5

    pipeline = [
        ('DC_Removal', dc_removal, {}),
        ('Dewow', dewow, {'window': 30}),
        ('Background_Removal', background_removal, {}),
        ('Bandpass', bandpass_filter,
         {'dt': dt_s, 'low_mhz': bp_low, 'high_mhz': bp_high}),
        ('Gain_SEC', gain_sec,
         {'tpow': 1.5, 'alpha': 0.1, 'dt': dt_s}),
    ]

    processed, intermediates, steps_log = run_pipeline(
        bscan, dt_s, dx, pipeline,
        db=db, dataset_id=dataset_id,
        description=f"Synthetic preprocessing: "
                    f"DC→Dewow→BGR→BP({bp_low:.0f}-{bp_high:.0f}MHz)→SEC"
    )

    return processed, intermediates, steps_log


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Week 3 - GPR Sinkhole Simulation")
    print("=" * 60)

    db = GPRDatabase()

    # ── Step 1: .in 파일 생성 ──
    print("\n[1] gprMax .in 파일 생성 (48개)...")
    in_files = generate_all_in_files()
    print(f"  생성 완료: {len(in_files)}개 → {MODELS_DIR}")

    # ── Step 2: gprMax 사용 가능 여부 확인 ──
    try:
        import gprMax
        use_gprmax = True
        print("\n[2] gprMax 설치 확인됨 → FDTD 시뮬레이션 사용")
    except ImportError:
        use_gprmax = False
        print("\n[2] gprMax 미설치 → 해석적 합성 방법 사용")

    # ── Step 3: 먼저 4개 테스트 ──
    print("\n[3] 테스트 시뮬레이션 (4개)...")
    test_params = [
        (900e6, 1.0, 0.5, 6),   # 기본 시나리오
        (400e6, 1.0, 0.5, 6),   # 저주파
        (900e6, 0.5, 0.2, 6),   # 얕은 소형
        (900e6, 2.0, 1.0, 12),  # 깊은 대형 습윤
    ]
    for freq, d, r, er in test_params:
        bscan, t_ns, x_ax, meta = synthesize_bscan(freq, d, r, er)
        print(f"  f{freq/1e6:.0f} d{d} r{r} er{er}: "
              f"shape={bscan.shape}, range=[{bscan.min():.4f}, {bscan.max():.4f}]")

    # ── Step 4: 48개 전체 실행 ──
    print("\n[4] 48개 전체 시뮬레이션...")
    results = run_all_simulations(use_gprmax=use_gprmax, db=db)

    # ── Step 5: 시각화 ──
    print("\n[5] 시각화...")

    plot_simulation_comparison(
        results,
        save_path=OUTPUT_DIR / "simulation_grid.png"
    )

    plot_sinkhole_detection(
        save_path=OUTPUT_DIR / "sinkhole_comparison.png"
    )

    plot_frequency_comparison(
        save_path=OUTPUT_DIR / "frequency_comparison.png"
    )

    plot_soil_comparison(
        save_path=OUTPUT_DIR / "soil_comparison.png"
    )

    # ── Step 6: 대표 시나리오 전처리 적용 ──
    print("\n[6] 대표 시나리오 전처리 적용...")
    repr_label = "f900_d1.0_r0.5_er6"
    repr_npy = SYNTHETIC_DIR / f"{repr_label}.npy"
    repr_meta_path = SYNTHETIC_DIR / f"{repr_label}_meta.json"

    if repr_npy.exists():
        repr_bscan = np.load(str(repr_npy))
        repr_meta = json.loads(repr_meta_path.read_text(encoding='utf-8'))

        # 전처리 적용
        print(f"  대표: {repr_label} {repr_bscan.shape}")
        processed, intermediates, steps_log = apply_preprocessing_to_synthetic(
            repr_bscan, repr_meta, db=db,
        )

        # 전처리 전/후 비교
        plot_preprocessing_comparison(
            intermediates,
            title=f"Synthetic Preprocessing: {repr_label}",
            save_path=OUTPUT_DIR / "synthetic_preprocessing.png"
        )

    # ── 최종 요약 ──
    print("\n" + "=" * 60)
    print("  최종 요약")
    print("=" * 60)
    print(f"  .in 파일: {len(in_files)}개 → {MODELS_DIR}")
    print(f"  합성 B-scan: {len(results)}개 → {SYNTHETIC_DIR}")
    print(f"  시각화:")
    for img in ['simulation_grid.png', 'sinkhole_comparison.png',
                'frequency_comparison.png', 'soil_comparison.png',
                'synthetic_preprocessing.png']:
        p = OUTPUT_DIR / img
        print(f"    {'✓' if p.exists() else '✗'} {img}")

    db.print_summary()
    print("완료!")
