"""
Week 2 - GPR Preprocessing Pipeline
전처리: DC Removal → Dewow → Background Removal → Bandpass → Gain(SEC) → FK Migration
"""

import sys
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt
from scipy.interpolate import interp1d

# week1 파서 import
sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_dt1, read_ids_dt, plot_bscan
from week2_database import GPRDatabase


# ─────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────

def estimate_dt(time_window_ns, n_samples):
    """시간창(ns)과 샘플수로 dt(초) 계산"""
    return (time_window_ns * 1e-9) / n_samples


# ─────────────────────────────────────────────
# 1. DC Removal (트레이스별 평균 제거)
# ─────────────────────────────────────────────

def dc_removal(data):
    """각 트레이스에서 DC 오프셋(평균값) 제거"""
    return data - np.mean(data, axis=0, keepdims=True)


# ─────────────────────────────────────────────
# 2. Dewow (저주파 와우 제거)
# ─────────────────────────────────────────────

def dewow(data, window=50):
    """
    이동평균으로 저주파 트렌드 제거 (트레이스별 적용)
    window: 이동평균 윈도우 크기 (샘플 수)
    """
    trend = uniform_filter1d(data, size=window, axis=0, mode='nearest')
    return data - trend


# ─────────────────────────────────────────────
# 3. Background Removal (수평 밴딩 제거)
# ─────────────────────────────────────────────

def background_removal(data):
    """전체 트레이스 평균(mean trace)을 빼서 수평 밴딩 제거"""
    return data - np.mean(data, axis=1, keepdims=True)


# ─────────────────────────────────────────────
# 4. Bandpass Filter (대역통과 필터)
# ─────────────────────────────────────────────

def bandpass_filter(data, dt, low_mhz, high_mhz, order=4):
    """
    Butterworth 대역통과 필터 (제로페이즈)
    dt: 샘플 간격 (초)
    low_mhz, high_mhz: 차단 주파수 (MHz)
    """
    fs = 1.0 / dt  # 샘플링 주파수 (Hz)
    nyq = fs / 2.0

    low_hz = low_mhz * 1e6
    high_hz = high_mhz * 1e6

    # 나이퀴스트 범위 클리핑
    low_norm = max(low_hz / nyq, 0.001)
    high_norm = min(high_hz / nyq, 0.999)

    if low_norm >= high_norm:
        print(f"  [경고] bandpass 범위 무효: {low_mhz}~{high_mhz}MHz "
              f"(fs={fs/1e6:.0f}MHz)")
        return data

    sos = butter(order, [low_norm, high_norm], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=0).astype(np.float32)


# ─────────────────────────────────────────────
# 5. Gain - SEC (Spreading & Exponential Compensation)
# ─────────────────────────────────────────────

def gain_sec(data, tpow=1.0, alpha=0.0, dt=1.0):
    """
    SEC 게인: gain(t) = t^tpow * exp(alpha * t)
    tpow: 시간 거듭제곱 (보통 1.0~2.0)
    alpha: 지수 감쇠 보상 계수 (보통 0~0.5)
    dt: 샘플 간격 (초), 시간축 계산용
    """
    n_samples = data.shape[0]
    t = np.arange(1, n_samples + 1, dtype=np.float64) * dt
    gain = (t ** tpow) * np.exp(alpha * t)
    # 정규화 (첫 샘플 기준)
    gain = gain / gain[0]
    return data * gain[:, np.newaxis].astype(np.float32)


# ─────────────────────────────────────────────
# 5b. Gain - AGC (Automatic Gain Control)
# ─────────────────────────────────────────────

def gain_agc(data, window=50):
    """
    AGC: 슬라이딩 윈도우 RMS로 정규화
    window: 윈도우 크기 (샘플 수)
    """
    eps = 1e-10
    rms = np.sqrt(uniform_filter1d(data.astype(np.float64) ** 2,
                                   size=window, axis=0, mode='nearest'))
    rms = np.maximum(rms, eps)
    # 전체 평균 RMS 수준으로 스케일링
    target_rms = np.mean(rms)
    return (data * (target_rms / rms)).astype(np.float32)


# ─────────────────────────────────────────────
# 6. FK Migration (Stolt Migration, 등속 가정)
# ─────────────────────────────────────────────

def fk_migration(data, dt, dx, velocity=0.1):
    """
    Stolt FK Migration (등속도 매질 가정)
    dt: 시간 샘플 간격 (초)
    dx: 트레이스 간격 (미터)
    velocity: 매질 내 전파 속도 (m/ns), 기본 0.1 m/ns

    알고리즘:
    1. 2D FFT로 f-k 도메인 변환
    2. Stolt 주파수 매핑: f_mig = sqrt(f^2 + (v*kx/2)^2)
    3. 보간으로 매핑 적용
    4. 역 2D FFT
    """
    n_samples, n_traces = data.shape
    v = velocity * 1e9  # m/ns → m/s 변환

    # 2D FFT
    DATA = np.fft.fft2(data)

    # 주파수 축
    freq = np.fft.fftfreq(n_samples, d=dt)      # Hz
    kx = np.fft.fftfreq(n_traces, d=dx)          # 1/m (공간 주파수)

    # 양의 주파수만 처리 (대칭성 이용)
    n_freq_pos = n_samples // 2 + 1
    freq_pos = np.abs(freq[:n_freq_pos])

    DATA_mig = np.zeros_like(DATA)

    for j in range(n_traces):
        kx_j = kx[j]
        # Stolt 매핑: f_mig = sqrt(f^2 + (v*kx/2)^2)
        shift = (v * kx_j / 2.0) ** 2
        f_mig = np.sqrt(freq_pos ** 2 + shift)

        # 원래 주파수 축에서 매핑된 주파수로 보간
        col_pos = DATA[:n_freq_pos, j]

        # 보간 (범위 밖은 0)
        interp_func = interp1d(freq_pos, col_pos,
                               kind='linear', bounds_error=False,
                               fill_value=0.0)
        mapped = interp_func(f_mig)

        # 위상 보정 (Jacobian)
        with np.errstate(divide='ignore', invalid='ignore'):
            jacobian = np.where(f_mig > 0, freq_pos / f_mig, 1.0)
        mapped = mapped * jacobian

        # 양의 주파수 결과 저장
        DATA_mig[:n_freq_pos, j] = mapped

        # 음의 주파수 (켤레 대칭)
        if n_samples % 2 == 0:
            DATA_mig[n_freq_pos:, j] = np.conj(mapped[1:-1][::-1])
        else:
            DATA_mig[n_freq_pos:, j] = np.conj(mapped[1:][::-1])

    # 역 FFT
    result = np.real(np.fft.ifft2(DATA_mig))
    return result.astype(np.float32)


# ─────────────────────────────────────────────
# 파이프라인 실행
# ─────────────────────────────────────────────

def run_pipeline(data, dt, dx, pipeline_config, db=None, dataset_id=None,
                 description="Standard GPR Pipeline"):
    """
    전처리 파이프라인 순차 실행

    pipeline_config: list of (step_name, func, kwargs)
    Returns: (final_data, intermediates dict, steps_log list)
    """
    intermediates = {'0_Raw': data.copy()}
    steps_log = []
    current = data.copy()

    for i, (step_name, func, kwargs) in enumerate(pipeline_config):
        t0 = time.perf_counter()
        current = func(current, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        intermediates[f'{i+1}_{step_name}'] = current.copy()
        steps_log.append({
            'step_name': step_name,
            'parameters': {k: str(v) for k, v in kwargs.items()},
            'elapsed_ms': round(elapsed_ms, 2),
        })
        print(f"  {i+1}. {step_name}: {elapsed_ms:.1f}ms "
              f"(range: {current.min():.1f} ~ {current.max():.1f})")

    # DB 기록
    if db and dataset_id:
        db.log_processing_run(dataset_id, description, steps_log)

    return current, intermediates, steps_log


# ─────────────────────────────────────────────
# 시각화 함수들
# ─────────────────────────────────────────────

def plot_preprocessing_comparison(intermediates, title="Pipeline Comparison",
                                  clip_pct=95, figsize=(20, 12),
                                  save_path=None):
    """6패널 파이프라인 진행 비교"""
    keys = list(intermediates.keys())
    n = len(keys)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    for i, key in enumerate(keys):
        ax = axes[i]
        d = intermediates[key]
        vmax = np.percentile(np.abs(d), clip_pct)
        if vmax == 0:
            vmax = 1
        ax.imshow(d, aspect='auto', cmap='seismic',
                  vmin=-vmax, vmax=vmax)
        ax.set_title(key.replace('_', ' '), fontsize=10, fontweight='bold')
        ax.set_xlabel('Trace')
        ax.set_ylabel('Sample')

    # 빈 패널 숨기기
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_before_after(before, after, title="Before vs After",
                      clip_pct=95, figsize=(14, 6), save_path=None):
    """전/후 2패널 비교"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, d, label in [(ax1, before, 'Before'), (ax2, after, 'After')]:
        vmax = np.percentile(np.abs(d), clip_pct)
        if vmax == 0:
            vmax = 1
        ax.imshow(d, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Trace')
        ax.set_ylabel('Sample')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_trace_comparison(before, after, trace_idx=None,
                          title="Trace Comparison", figsize=(10, 6),
                          save_path=None):
    """단일 트레이스 1D 파형 비교"""
    if trace_idx is None:
        trace_idx = before.shape[1] // 2

    fig, ax = plt.subplots(figsize=figsize)
    samples = np.arange(before.shape[0])

    ax.plot(before[:, trace_idx], samples, 'b-', alpha=0.6,
            label='Before', linewidth=0.8)
    ax.plot(after[:, trace_idx], samples, 'r-', alpha=0.8,
            label='After', linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Sample')
    ax.set_title(f"{title} (Trace #{trace_idx})", fontsize=12,
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


def plot_amplitude_spectrum(before, after, dt, trace_idx=None,
                            title="Amplitude Spectrum",
                            figsize=(10, 6), save_path=None):
    """FFT 주파수 스펙트럼 비교 (bandpass 효과 확인)"""
    if trace_idx is None:
        trace_idx = before.shape[1] // 2

    fig, ax = plt.subplots(figsize=figsize)

    for d, label, color in [(before, 'Before', 'blue'),
                            (after, 'After', 'red')]:
        trace = d[:, trace_idx]
        n = len(trace)
        spectrum = np.abs(np.fft.rfft(trace))
        freq_mhz = np.fft.rfftfreq(n, d=dt) / 1e6  # Hz → MHz

        ax.plot(freq_mhz, spectrum, color=color, alpha=0.7,
                label=label, linewidth=0.8)

    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f"{title} (Trace #{trace_idx})", fontsize=12,
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  저장: {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":

    DATA_DIR = Path("G:/RAG_system/data/gpr")
    OUTPUT_DIR = Path("G:/RAG_system/src/output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Week 2 - GPR Preprocessing Pipeline")
    print("=" * 60)

    db = GPRDatabase()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Dataset 1: Frenke LINE00 (100MHz, DT1)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    frenke_path = str(DATA_DIR / "frenke/2014_04_25_frenke/rawGPR/LINE00.DT1")
    print(f"\n[1] Frenke LINE00 로딩...")

    data_frenke, header_frenke = read_dt1(frenke_path)
    print(f"  Shape: {data_frenke.shape}, "
          f"dtype: {data_frenke.dtype}")

    # 파라미터 설정
    freq_frenke = 100.0  # MHz
    tw_frenke = float(header_frenke.get('TOTAL TIME WINDOW', 50.0))
    dt_frenke = estimate_dt(tw_frenke, data_frenke.shape[0])
    dx_frenke = 0.25  # m (트레이스 간격)

    print(f"  Freq: {freq_frenke}MHz, TW: {tw_frenke}ns, "
          f"dt: {dt_frenke*1e9:.3f}ns, dx: {dx_frenke}m")

    # DB 등록
    frenke_id = db.register_dataset(
        name="Frenke LINE00",
        file_path=frenke_path,
        data=data_frenke,
        header=header_frenke,
        format="DT1",
        frequency_mhz=freq_frenke,
        time_window_ns=tw_frenke,
        dx_m=dx_frenke,
    )

    # 파이프라인 정의
    pipeline_frenke = [
        ('DC_Removal', dc_removal, {}),
        ('Dewow', dewow, {'window': 50}),
        ('Background_Removal', background_removal, {}),
        ('Bandpass', bandpass_filter,
         {'dt': dt_frenke, 'low_mhz': 25, 'high_mhz': 250}),
        ('Gain_SEC', gain_sec,
         {'tpow': 1.5, 'alpha': 0.2, 'dt': dt_frenke}),
        ('FK_Migration', fk_migration,
         {'dt': dt_frenke, 'dx': dx_frenke, 'velocity': 0.1}),
    ]

    print(f"\n  파이프라인 실행 ({len(pipeline_frenke)}단계):")
    processed_frenke, intermediates_frenke, log_frenke = run_pipeline(
        data_frenke, dt_frenke, dx_frenke,
        pipeline_frenke, db=db, dataset_id=frenke_id,
        description="Standard Pipeline: DC→Dewow→BGR→BP(25-250MHz)→SEC→FK"
    )

    # 시각화
    plot_preprocessing_comparison(
        intermediates_frenke,
        title="Frenke LINE00 - Preprocessing Pipeline",
        save_path=OUTPUT_DIR / "frenke_pipeline.png"
    )

    # Bandpass 전/후 스펙트럼
    before_bp = intermediates_frenke['3_Background_Removal']
    after_bp = intermediates_frenke['4_Bandpass']
    plot_amplitude_spectrum(
        before_bp, after_bp, dt_frenke,
        title="Frenke LINE00 - Bandpass Effect",
        save_path=OUTPUT_DIR / "frenke_spectrum.png"
    )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Dataset 2: Guangzhou rebar (2GHz, IDS .dt)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    gz_rebar_dir = DATA_DIR / "guangzhou/Data Set/rebar"
    gz_dt_files = sorted(
        f for f in gz_rebar_dir.rglob("*.dt")
        if 'ASCII' not in str(f)
    ) if gz_rebar_dir.exists() else []

    if gz_dt_files:
        gz_path = str(gz_dt_files[0])
        print(f"\n[2] Guangzhou rebar 로딩...")
        print(f"  파일: {gz_path}")

        data_gz, header_gz = read_ids_dt(gz_path)

        if data_gz is not None:
            print(f"  Shape: {data_gz.shape}, dtype: {data_gz.dtype}")

            # 파라미터 설정 (2GHz IDS)
            freq_gz = 2000.0  # MHz
            # IDS SweepTime에서 시간창 계산 (보통 20~40ns)
            sweep_str = header_gz.get('sweep_time',
                        header_gz.get('SweepTime', ''))
            try:
                tw_gz = float(sweep_str) * 1e9 if float(sweep_str) < 1 else float(sweep_str)
            except (ValueError, TypeError):
                tw_gz = 25.0  # 2GHz 기본 추정값

            dt_gz = estimate_dt(tw_gz, data_gz.shape[0])
            dx_gz = 0.01  # m (1cm 간격)

            print(f"  Freq: {freq_gz}MHz, TW: {tw_gz:.1f}ns, "
                  f"dt: {dt_gz*1e12:.1f}ps, dx: {dx_gz}m")

            # DB 등록
            gz_id = db.register_dataset(
                name="Guangzhou rebar",
                file_path=gz_path,
                data=data_gz,
                header=header_gz,
                format="IDS .dt",
                frequency_mhz=freq_gz,
                time_window_ns=tw_gz,
                dx_m=dx_gz,
            )

            # 파이프라인 정의 (2GHz 고주파)
            pipeline_gz = [
                ('DC_Removal', dc_removal, {}),
                ('Dewow', dewow, {'window': 30}),
                ('Background_Removal', background_removal, {}),
                ('Bandpass', bandpass_filter,
                 {'dt': dt_gz, 'low_mhz': 500, 'high_mhz': 5000}),
                ('Gain_SEC', gain_sec,
                 {'tpow': 1.0, 'alpha': 0.1, 'dt': dt_gz}),
                ('FK_Migration', fk_migration,
                 {'dt': dt_gz, 'dx': dx_gz, 'velocity': 0.1}),
            ]

            print(f"\n  파이프라인 실행 ({len(pipeline_gz)}단계):")
            processed_gz, intermediates_gz, log_gz = run_pipeline(
                data_gz, dt_gz, dx_gz,
                pipeline_gz, db=db, dataset_id=gz_id,
                description="Standard Pipeline: DC→Dewow→BGR→BP(500-5000MHz)→SEC→FK"
            )

            # 시각화
            plot_preprocessing_comparison(
                intermediates_gz,
                title="Guangzhou Rebar - Preprocessing Pipeline",
                save_path=OUTPUT_DIR / "guangzhou_rebar_pipeline.png"
            )

            before_bp_gz = intermediates_gz['3_Background_Removal']
            after_bp_gz = intermediates_gz['4_Bandpass']
            plot_amplitude_spectrum(
                before_bp_gz, after_bp_gz, dt_gz,
                title="Guangzhou Rebar - Bandpass Effect",
                save_path=OUTPUT_DIR / "guangzhou_rebar_spectrum.png"
            )
        else:
            print("  Guangzhou rebar 파싱 실패")
    else:
        print("\n[2] Guangzhou rebar .dt 파일 없음")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DB 최종 요약
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    db.print_summary()

    print("완료! output/ 폴더에서 이미지 확인하세요.")
