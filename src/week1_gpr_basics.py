"""
Week 1 - GPR Data Loading & B-scan Visualization
데이터: Frenke (DT1), NSGeophysics (DZT), Tagliamento (DT1), Guangzhou (IDS .dt)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import struct
import os
from pathlib import Path

# ─────────────────────────────────────────────
# 1. DT1 파일 파서 (Sensors & Software 포맷)
# ─────────────────────────────────────────────

def read_dt1_header(hd_path):
    """
    .HD 헤더 파일 파싱 (: 또는 = 구분자 모두 지원)
    반환: dict (샘플수, 트레이스수, 시간창, 안테나주파수 등)
    """
    header = {}
    with open(hd_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, _, val = line.partition('=')
                header[key.strip()] = val.strip()
            elif ':' in line:
                key, _, val = line.partition(':')
                header[key.strip()] = val.strip()
    return header


def _get_n_samples(header):
    """HD 헤더에서 샘플 수 추출 (키 이름이 다를 수 있음)"""
    for key in ['NUMBER OF PTS/TRC', 'SAMPLES PER TRACE', 'NPTS']:
        if key in header:
            return int(float(header[key]))
    return 512


def read_dt1(dt1_path):
    """
    .DT1 바이너리 데이터 파싱
    반환: (data 2D array, header dict)
      data shape: (n_samples, n_traces)
    """
    hd_path = dt1_path.replace('.DT1', '.HD').replace('.dt1', '.hd')
    header = read_dt1_header(hd_path)

    n_samples = _get_n_samples(header)

    data_list = []
    with open(dt1_path, 'rb') as f:
        while True:
            # 각 트레이스 앞 25×4바이트(100바이트) 헤더 스킵
            trace_header = f.read(25 * 4)
            if len(trace_header) < 100:
                break
            # 트레이스 데이터 (16-bit int)
            raw = f.read(n_samples * 2)
            if len(raw) < n_samples * 2:
                break
            trace = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            data_list.append(trace)

    data = np.array(data_list).T  # (n_samples, n_traces)
    return data, header


# ─────────────────────────────────────────────
# 2. DZT 파일 파서 (GSSI 포맷) - readgssi 활용
# ─────────────────────────────────────────────

def read_dzt(dzt_path):
    """
    .DZT 파일 파싱 (readgssi 라이브러리 활용)
    반환: (data 2D array, header dict)
    """
    try:
        import readgssi.readgssi as rg
        # readgssi로 읽기
        header, data, gps = rg.readgssi(infile=dzt_path, verbose=False)
        return data[0].astype(np.float32), header
    except Exception as e:
        print(f"readgssi 오류: {e}")
        return None, None


# ─────────────────────────────────────────────
# 3. IDS GeoRadar .dt 파서 (Guangzhou 터널 데이터)
# ─────────────────────────────────────────────

def _read_ids_ini(dt_path):
    """IDS .dt 파일과 같은 디렉토리의 Ini0001.ini에서 메타데이터 추출"""
    ini_path = os.path.join(os.path.dirname(dt_path), 'Ini0001.ini')
    info = {}
    if os.path.exists(ini_path):
        with open(ini_path, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if line.startswith(';') or not line:
                    continue
                if '=' in line:
                    key, _, val = line.partition('=')
                    info[key.strip()] = val.strip()
    return info


def read_ids_dt(dt_path):
    """
    IDS GeoRadar .dt 바이너리 파싱
    ini 파일에서 ACQ_SAMPLE 읽고, per-trace 4바이트 헤더 처리
    반환: (data 2D array, header dict)
    """
    with open(dt_path, 'rb') as f:
        raw = f.read()

    file_size = len(raw)
    ini = _read_ids_ini(dt_path)

    # ini에서 샘플 수 가져오기 (기본 512)
    ini_samples = int(ini.get('ACQ_SAMPLE', 0))
    n_samples_candidates = [512, 1024, 2048, 256]
    if ini_samples and ini_samples not in n_samples_candidates:
        n_samples_candidates.insert(0, ini_samples)
    elif ini_samples:
        n_samples_candidates.remove(ini_samples)
        n_samples_candidates.insert(0, ini_samples)

    # 방법 1: per-trace 4바이트 헤더 (IDS 표준)
    for n_samples in n_samples_candidates:
        trace_block = 4 + n_samples * 2  # 4-byte header + int16 data
        if file_size % trace_block == 0:
            n_traces = file_size // trace_block
            if n_traces > 10:
                traces = []
                for i in range(n_traces):
                    offset = i * trace_block + 4  # skip 4-byte header
                    trace = np.frombuffer(raw[offset:offset + n_samples * 2], dtype=np.int16)
                    traces.append(trace)
                arr = np.array(traces, dtype=np.float32).T
                header = {
                    'format': 'IDS GeoRadar .dt (per-trace hdr)',
                    'n_samples': n_samples,
                    'n_traces': n_traces,
                    'dtype': 'int16',
                    'freq_mhz': ini.get('MAX_FREQ', 'N/A'),
                    'sweep_time': ini.get('SweepTime', 'N/A'),
                }
                return arr, header

    # 방법 2: 파일 헤더 + 연속 데이터 (헤더 없음)
    for header_size in [0, 1024, 2048, 4096]:
        data_bytes = raw[header_size:]
        for n_samples in n_samples_candidates:
            bytes_per_trace = n_samples * 2  # 16-bit
            if len(data_bytes) % bytes_per_trace == 0:
                n_traces = len(data_bytes) // bytes_per_trace
                if n_traces > 10:
                    arr = np.frombuffer(data_bytes, dtype=np.int16).reshape(n_traces, n_samples).T
                    header = {
                        'format': 'IDS GeoRadar .dt',
                        'header_size': header_size,
                        'n_samples': n_samples,
                        'n_traces': n_traces,
                        'dtype': 'int16',
                    }
                    return arr.astype(np.float32), header

    # 방법 3: 32-bit float
    for header_size in [0, 1024, 2048, 4096]:
        data_bytes = raw[header_size:]
        for n_samples in n_samples_candidates:
            bytes_per_trace = n_samples * 4  # 32-bit
            if len(data_bytes) % bytes_per_trace == 0:
                n_traces = len(data_bytes) // bytes_per_trace
                if n_traces > 10:
                    arr = np.frombuffer(data_bytes, dtype=np.float32).reshape(n_traces, n_samples).T
                    header = {
                        'format': 'IDS GeoRadar .dt (float32)',
                        'header_size': header_size,
                        'n_samples': n_samples,
                        'n_traces': n_traces,
                        'dtype': 'float32',
                    }
                    return arr, header

    print(f"  ⚠️ .dt 파싱 실패: {dt_path} (size={file_size})")
    return None, None


# ─────────────────────────────────────────────
# 4. B-scan 시각화
# ─────────────────────────────────────────────

def plot_bscan(data, title="B-scan (Radargram)",
               time_window_ns=None, figsize=(14, 6),
               cmap='seismic', clip_pct=95):
    """
    GPR B-scan(레이더그램) 시각화

    Parameters:
        data       : 2D array (n_samples, n_traces)
        title      : 그래프 제목
        time_window_ns : 전체 시간창(ns), None이면 샘플 인덱스로 표시
        cmap       : 컬러맵 ('seismic', 'gray', 'bwr')
        clip_pct   : 클리핑 백분위수 (노이즈 제거)
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 클리핑으로 다이나믹 레인지 조정
    vmax = np.percentile(np.abs(data), clip_pct)

    extent = [0, data.shape[1], data.shape[0], 0]
    if time_window_ns:
        extent[2] = time_window_ns
        extent[3] = 0

    im = ax.imshow(data, aspect='auto', cmap=cmap,
                   vmin=-vmax, vmax=vmax, extent=extent)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Trace Number', fontsize=12)
    ax.set_ylabel('Time (ns)' if time_window_ns else 'Sample Index', fontsize=12)

    plt.colorbar(im, ax=ax, label='Amplitude', fraction=0.02)
    plt.tight_layout()
    return fig


def plot_bscan_comparison(data, title="B-scan Comparison", figsize=(16, 10)):
    """
    3가지 컬러맵으로 동시 비교
    """
    cmaps = [('seismic', 'Seismic (표준)'),
             ('gray', 'Grayscale'),
             ('bwr', 'Blue-White-Red')]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    vmax = np.percentile(np.abs(data), 95)

    for ax, (cmap, label) in zip(axes, cmaps):
        im = ax.imshow(data, aspect='auto', cmap=cmap,
                       vmin=-vmax, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Trace')
        ax.set_ylabel('Sample')
        plt.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 4. 기본 통계 출력
# ─────────────────────────────────────────────

def print_gpr_stats(data, header, name="GPR Data"):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Shape     : {data.shape}  (samples x traces)")
    print(f"  n_samples : {data.shape[0]}")
    print(f"  n_traces  : {data.shape[1]}")
    print(f"  dtype     : {data.dtype}")
    print(f"  min/max   : {data.min():.1f} / {data.max():.1f}")
    print(f"  mean      : {data.mean():.4f}")
    print(f"  std       : {data.std():.2f}")
    if header:
        freq = header.get('ANTENNA FREQUENCY', header.get('antfreq', 'N/A'))
        tw   = header.get('TOTAL TIME WINDOW', header.get('ns_per_zsample', 'N/A'))
        print(f"  Frequency : {freq}")
        print(f"  Time win  : {tw}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":

    DATA_DIR = Path("G:/RAG_system/data/gpr")
    OUTPUT_DIR = Path("G:/RAG_system/src/output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("GPR 1주차 - 데이터 로딩 & B-scan 시각화")
    print("=" * 50)

    # ── Frenke DT1 데이터 ──
    frenke_lines = list((DATA_DIR / "frenke/2014_04_25_frenke/rawGPR").glob("LINE*.DT1"))

    if frenke_lines:
        print(f"\n[Frenke] {len(frenke_lines)}개 라인 발견")

        for dt1_path in sorted(frenke_lines):
            data, header = read_dt1(str(dt1_path))
            print_gpr_stats(data, header, name=dt1_path.stem)

        # LINE00 시각화
        dt1_path = sorted(frenke_lines)[0]
        data, header = read_dt1(str(dt1_path))

        fig = plot_bscan(data, title=f"Frenke {dt1_path.stem} - B-scan")
        fig.savefig(OUTPUT_DIR / "frenke_LINE00_bscan.png", dpi=150, bbox_inches='tight')
        print(f"  저장: output/frenke_LINE00_bscan.png")

        fig2 = plot_bscan_comparison(data, title=f"Frenke {dt1_path.stem} - 컬러맵 비교")
        fig2.savefig(OUTPUT_DIR / "frenke_LINE00_comparison.png", dpi=150, bbox_inches='tight')
        print(f"  저장: output/frenke_LINE00_comparison.png")

        plt.close('all')

    # ── NSGeophysics DZT 데이터 ──
    dune_dzt = DATA_DIR / "NSGeophysics/ExampleDuneProfile/DuneData.DZT"

    if dune_dzt.exists():
        print(f"\n[NSGeophysics] DuneData.DZT 로딩...")
        data_dzt, header_dzt = read_dzt(str(dune_dzt))

        if data_dzt is not None:
            print_gpr_stats(data_dzt, header_dzt, name="DuneData (DZT)")

            fig = plot_bscan(data_dzt, title="NSGeophysics - Dune Profile B-scan")
            fig.savefig(OUTPUT_DIR / "dune_bscan.png", dpi=150, bbox_inches='tight')
            print(f"  저장: output/dune_bscan.png")
            plt.close('all')

    # ── Tagliamento River DT1 데이터 (Zenodo) ──
    tagl_dt1 = DATA_DIR / "tagliamento/yyline3.DT1"

    if tagl_dt1.exists():
        print(f"\n[Tagliamento] yyline3.DT1 로딩...")
        data_tagl, header_tagl = read_dt1(str(tagl_dt1))
        print_gpr_stats(data_tagl, header_tagl, name="Tagliamento yyline3")

        tw = float(header_tagl.get('TOTAL TIME WINDOW', 0)) or None
        fig = plot_bscan(data_tagl, title="Tagliamento River - yyline3 B-scan",
                         time_window_ns=tw)
        fig.savefig(OUTPUT_DIR / "tagliamento_bscan.png", dpi=150, bbox_inches='tight')
        print(f"  저장: output/tagliamento_bscan.png")

        fig2 = plot_bscan_comparison(data_tagl, title="Tagliamento yyline3 - 컬러맵 비교")
        fig2.savefig(OUTPUT_DIR / "tagliamento_comparison.png", dpi=150, bbox_inches='tight')
        print(f"  저장: output/tagliamento_comparison.png")
        plt.close('all')

    # ── Guangzhou 터널 데이터 (IDS .dt) ──
    gz_dir = DATA_DIR / "guangzhou"
    gz_data_dir = gz_dir / "Data Set"
    dt_files = sorted(gz_data_dir.rglob("*.dt")) if gz_data_dir.exists() else []

    if dt_files:
        print(f"\n[Guangzhou] {len(dt_files)}개 .dt 파일 발견")

        # 카테고리별 (pipe, rebar, tunnel) 각 1개씩 시각화
        categories = {}
        for dt_path in dt_files:
            rel = dt_path.relative_to(gz_data_dir)
            cat = rel.parts[0] if rel.parts else "unknown"
            if cat not in categories:
                categories[cat] = dt_path

        for cat, dt_path in categories.items():
            print(f"\n  [{cat}] 로딩: {dt_path.relative_to(gz_dir)}")
            data_gz, header_gz = read_ids_dt(str(dt_path))

            if data_gz is not None:
                print_gpr_stats(data_gz, header_gz, name=f"Guangzhou {cat}/{dt_path.stem}")

                fig = plot_bscan(data_gz, title=f"Guangzhou {cat} - {dt_path.stem} B-scan")
                fig.savefig(OUTPUT_DIR / f"guangzhou_{cat}_bscan.png", dpi=150, bbox_inches='tight')
                print(f"  저장: output/guangzhou_{cat}_bscan.png")
                plt.close('all')
            else:
                print(f"  ⚠️ {cat}/{dt_path.stem} 파싱 실패")
    elif gz_dir.exists():
        print(f"\n[Guangzhou] .dt 파일 없음 - zip 압축 해제 필요")

    print("\n완료! output/ 폴더에서 이미지 확인하세요.")
