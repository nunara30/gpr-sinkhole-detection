"""
Week 3 - GPR 오픈소스 프로젝트 분석 & 비교
GPRPy, gprMax, siina 3개 프로젝트의 핵심 구조를 코드로 분석/비교
"""

import ast
import sys
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
BASE_DIR = Path("G:/RAG_system")
GPRPY_DIR = BASE_DIR / "GPRPy_src" / "gprpy"
GPRMAX_DIR = BASE_DIR / "gprMax"
OUTPUT_DIR = BASE_DIR / "src" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# siina은 설치됨 → 패키지 경로
try:
    import siina
    SIINA_DIR = Path(siina.__file__).parent
except ImportError:
    SIINA_DIR = None


# ─────────────────────────────────────────────
# AST 기반 함수/클래스 추출
# ─────────────────────────────────────────────

def extract_functions_from_file(filepath):
    """Python 소스에서 함수명 + docstring 추출 (AST)"""
    results = []
    try:
        source = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return results

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            doc = ast.get_docstring(node) or ""
            doc_line = doc.split('\n')[0].strip() if doc else ""
            results.append({
                'name': node.name,
                'args': [a.arg for a in node.args.args if a.arg != 'self'],
                'doc': doc_line,
                'lineno': node.lineno,
            })
    return results


def extract_classes_from_file(filepath):
    """Python 소스에서 클래스명 + 메서드 추출"""
    results = []
    try:
        source = Path(filepath).read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return results

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith('_'):
                    doc = ast.get_docstring(item) or ""
                    doc_line = doc.split('\n')[0].strip() if doc else ""
                    methods.append({
                        'name': item.name,
                        'doc': doc_line,
                    })
            doc = ast.get_docstring(node) or ""
            results.append({
                'class_name': node.name,
                'doc': doc.split('\n')[0].strip() if doc else "",
                'methods': methods,
            })
    return results


# ─────────────────────────────────────────────
# GPRPy 분석
# ─────────────────────────────────────────────

def analyze_gprpy():
    """GPRPy 프로젝트 분석"""
    info = {
        'name': 'GPRPy',
        'description': 'GUI 기반 GPR 전처리/시각화 도구',
        'url': 'https://github.com/NSGeophysics/GPRPy',
        'formats': [],
        'processing': [],
        'modules': {},
        'classes': [],
    }

    if not GPRPY_DIR.exists():
        info['error'] = f"소스 디렉토리 없음: {GPRPY_DIR}"
        return info

    # I/O 포맷 추출 (gprIO_*.py 파일)
    toolbox_dir = GPRPY_DIR / "toolbox"
    io_formats = {
        'DT1': 'Sensors & Software (.DT1 + .HD)',
        'DZT': 'GSSI (.DZT)',
        'BSQ': 'ENVI BSQ (.DAT + .GPRhdr)',
        'MALA': 'MALA (.rad + .rd3/.rd7)',
    }
    for fmt_key, fmt_desc in io_formats.items():
        io_file = toolbox_dir / f"gprIO_{fmt_key}.py"
        if io_file.exists():
            info['formats'].append(fmt_desc)

    # 핵심 모듈 분석
    key_files = {
        'gprpy.py': GPRPY_DIR / 'gprpy.py',
        'gprpyTools.py': toolbox_dir / 'gprpyTools.py',
    }

    for name, path in key_files.items():
        if path.exists():
            funcs = extract_functions_from_file(path)
            classes = extract_classes_from_file(path)
            info['modules'][name] = {
                'functions': funcs,
                'n_functions': len(funcs),
                'lines': len(path.read_text(encoding='utf-8', errors='ignore').splitlines()),
            }
            info['classes'].extend(classes)

    # 전처리 기능 분류
    processing_map = {
        'Filtering': ['dewow', 'smooth', 'bandpass'],
        'Noise Removal': ['remMeanTrace', 'alignTraces'],
        'Gain': ['tpowGain', 'agcGain', 'normalize'],
        'Migration': ['fkMigration'],
        'Velocity Analysis': ['linStackedAmplitude', 'hypStackedAmplitude',
                              'setVelocity'],
        'Topography': ['correctTopo', 'topoCorrect', 'prepTopo'],
        'Data Manipulation': ['cut', 'flipProfile', 'adjProfile',
                              'setZeroTime', 'truncateY'],
        'Visualization': ['showProfile', 'printProfile', 'showCWFig'],
        'History': ['showHistory', 'writeHistory', 'undo'],
        'Export': ['exportVTK', 'save'],
    }

    all_func_names = set()
    for mod_info in info['modules'].values():
        for f in mod_info['functions']:
            all_func_names.add(f['name'])
    for cls in info['classes']:
        for m in cls['methods']:
            all_func_names.add(m['name'])

    for category, keywords in processing_map.items():
        matched = [k for k in keywords if k in all_func_names]
        if matched:
            info['processing'].append((category, matched))

    return info


# ─────────────────────────────────────────────
# gprMax 분석
# ─────────────────────────────────────────────

def analyze_gprmax():
    """gprMax 프로젝트 분석"""
    info = {
        'name': 'gprMax',
        'description': 'FDTD 기반 GPR 전자기파 시뮬레이션',
        'url': 'https://github.com/gprMax/gprMax',
        'formats': ['HDF5 (.out) — FDTD 시뮬레이션 출력'],
        'processing': [],
        'modules': {},
        'in_commands': {},
        'materials': [],
        'geometries': [],
        'sources': [],
        'waveforms': [],
        'tools': [],
    }

    if not GPRMAX_DIR.exists():
        info['error'] = f"소스 디렉토리 없음: {GPRMAX_DIR}"
        return info

    # .in 파일 명령어 분류
    info['in_commands'] = {
        'Domain': ['#domain', '#dx_dy_dz', '#time_window', '#pml_cells'],
        'Material': ['#material', '#soil_peplinski', '#add_dispersion_debye',
                     '#add_dispersion_lorentz', '#add_dispersion_drude'],
        'Geometry': ['#box', '#cylinder', '#cylindrical_sector', '#sphere',
                     '#edge', '#plate', '#triangle', '#fractal_box',
                     '#add_surface_roughness'],
        'Source': ['#hertzian_dipole', '#magnetic_dipole', '#voltage_source',
                   '#transmission_line'],
        'Waveform': ['#waveform (types: gaussian, gaussiandot, ricker, '
                     'sine, contsine, impulse, user)'],
        'Receiver': ['#rx', '#rx_array'],
        'Output': ['#snapshot', '#geometry_view', '#geometry_objects_write'],
        'Stepping': ['#src_steps', '#rx_steps'],
        'Config': ['#title', '#num_threads', '#messages', '#output_dir'],
    }

    info['materials'] = [
        'free_space (built-in, εr=1)',
        'pec (built-in, perfect conductor)',
        'Custom isotropic (#material: εr σ μr σ*)',
        'Debye dispersive',
        'Lorentz dispersive',
        'Drude dispersive',
        'Peplinski soil model',
    ]

    info['geometries'] = [
        'Box (rectangular)',
        'Cylinder (circular cross-section)',
        'Cylindrical Sector (partial cylinder)',
        'Sphere',
        'Edge (thin wire)',
        'Plate (thin surface)',
        'Triangle (triangular prism)',
        'Fractal Box (stochastic heterogeneous)',
    ]

    info['sources'] = [
        'Hertzian Dipole (electric)',
        'Magnetic Dipole',
        'Voltage Source',
        'Transmission Line',
    ]

    info['waveforms'] = [
        'gaussian', 'gaussiandot', 'gaussiandotdot',
        'ricker', 'sine', 'contsine', 'impulse', 'user',
    ]

    # tools/ 분석
    tools_dir = GPRMAX_DIR / "tools"
    if tools_dir.exists():
        for py_file in sorted(tools_dir.glob("*.py")):
            if py_file.name.startswith('_'):
                continue
            funcs = extract_functions_from_file(py_file)
            info['tools'].append({
                'file': py_file.name,
                'functions': [f['name'] for f in funcs],
            })
            info['modules'][f'tools/{py_file.name}'] = {
                'functions': funcs,
                'n_functions': len(funcs),
                'lines': len(py_file.read_text(encoding='utf-8',
                             errors='ignore').splitlines()),
            }

    # gprMax/ 소스 파일 통계
    src_dir = GPRMAX_DIR / "gprMax"
    if src_dir.exists():
        py_files = list(src_dir.glob("*.py"))
        total_lines = 0
        for f in py_files:
            total_lines += len(f.read_text(encoding='utf-8',
                               errors='ignore').splitlines())
        info['modules']['gprMax/ (core)'] = {
            'n_files': len(py_files),
            'total_lines': total_lines,
        }

    # 시뮬레이션 기능
    info['processing'] = [
        ('FDTD Simulation', ['2D/3D electromagnetic wave propagation']),
        ('Material Modeling', ['isotropic', 'dispersive', 'soil mixing']),
        ('Geometry Building', ['primitive shapes', 'fractal heterogeneity']),
        ('Post-processing', ['A-scan/B-scan extraction', 'output merging']),
        ('Visualization', ['field snapshots (VTK)', 'geometry view']),
    ]

    return info


# ─────────────────────────────────────────────
# siina 분석
# ─────────────────────────────────────────────

def analyze_siina():
    """siina 프로젝트 분석"""
    info = {
        'name': 'siina',
        'description': 'GPR 데이터 I/O + 기본 필터링 라이브러리',
        'url': 'https://github.com/ahartikainen/siina',
        'formats': ['GSSI DZT (.dzt) — 읽기 전용'],
        'processing': [],
        'modules': {},
        'classes': [],
    }

    if SIINA_DIR is None or not SIINA_DIR.exists():
        info['error'] = "siina 미설치"
        return info

    # 모듈 분석
    for py_file in sorted(SIINA_DIR.glob("*.py")):
        if py_file.name.startswith('_'):
            continue
        funcs = extract_functions_from_file(py_file)
        classes = extract_classes_from_file(py_file)
        info['modules'][py_file.name] = {
            'functions': funcs,
            'n_functions': len(funcs),
            'lines': len(py_file.read_text(encoding='utf-8',
                         errors='ignore').splitlines()),
        }
        info['classes'].extend(classes)

    info['processing'] = [
        ('I/O', ['DZT read (auto header parsing)']),
        ('Filtering', ['Butterworth (low/high/band/bandstop)']),
        ('Preprocessing', ['DC removal (func_dc)']),
        ('Axis Generation', ['sample time', 'profile time', 'profile distance']),
    ]

    return info


# ─────────────────────────────────────────────
# Week 2 파이프라인과 비교
# ─────────────────────────────────────────────

def compare_with_week2():
    """Week 2 파이프라인 기능과 3개 프로젝트 비교"""
    week2_features = [
        'DC Removal',
        'Dewow (low-frequency removal)',
        'Background Removal (mean trace)',
        'Bandpass Filter (Butterworth)',
        'Gain SEC (t-power + exponential)',
        'Gain AGC (automatic)',
        'FK Migration (Stolt)',
        'Amplitude Spectrum',
    ]

    comparison = {
        'Feature': [],
        'Week 2': [],
        'GPRPy': [],
        'gprMax': [],
        'siina': [],
    }

    feature_mapping = {
        'DC Removal':       ('✓', '✗ (dewow 대체)', '✗', '✓ func_dc'),
        'Dewow (low-frequency removal)':
                            ('✓', '✓ dewow()', '✗', '✗'),
        'Background Removal (mean trace)':
                            ('✓', '✓ remMeanTrace()', '✗', '✗'),
        'Bandpass Filter (Butterworth)':
                            ('✓', '✓ smooth()', '✗', '✓ butterworth()'),
        'Gain SEC (t-power + exponential)':
                            ('✓', '✓ tpowGain()', '✗', '✗'),
        'Gain AGC (automatic)':
                            ('✓', '✓ agcGain()', '✗', '✗'),
        'FK Migration (Stolt)':
                            ('✓', '✓ fkMigration()', '✗ (FDTD)', '✗'),
        'Amplitude Spectrum':
                            ('✓', '✗', '✗', '✗'),
    }

    for feat in week2_features:
        w2, gpy, gm, si = feature_mapping.get(feat, ('?', '?', '?', '?'))
        comparison['Feature'].append(feat)
        comparison['Week 2'].append(w2)
        comparison['GPRPy'].append(gpy)
        comparison['gprMax'].append(gm)
        comparison['siina'].append(si)

    return comparison


# ─────────────────────────────────────────────
# 출력 포맷팅
# ─────────────────────────────────────────────

def format_comparison_table(comparison):
    """비교표 → 포맷된 텍스트"""
    lines = []

    # 열 너비 계산
    col_widths = {}
    for col in comparison:
        max_w = len(col)
        for val in comparison[col]:
            max_w = max(max_w, len(str(val)))
        col_widths[col] = max_w + 2

    cols = list(comparison.keys())

    # 헤더
    header = '│'.join(f" {c:<{col_widths[c]-1}}" for c in cols)
    sep = '┼'.join('─' * col_widths[c] for c in cols)
    lines.append('┌' + '┬'.join('─' * col_widths[c] for c in cols) + '┐')
    lines.append('│' + header + '│')
    lines.append('├' + sep + '┤')

    # 행
    n_rows = len(comparison[cols[0]])
    for i in range(n_rows):
        row = '│'.join(
            f" {str(comparison[c][i]):<{col_widths[c]-1}}"
            for c in cols
        )
        lines.append('│' + row + '│')

    lines.append('└' + '┴'.join('─' * col_widths[c] for c in cols) + '┘')
    return '\n'.join(lines)


def generate_summary_text(gprpy_info, gprmax_info, siina_info, comparison):
    """전체 분석 요약 텍스트 생성"""
    lines = []
    lines.append("=" * 80)
    lines.append("  Week 3 - GPR 오픈소스 프로젝트 분석 요약")
    lines.append("  생성일: 2026-02-26")
    lines.append("=" * 80)

    # ── 프로젝트별 요약 ──
    for info in [gprpy_info, gprmax_info, siina_info]:
        lines.append("")
        lines.append("─" * 80)
        lines.append(f"  [{info['name']}] {info['description']}")
        lines.append(f"  URL: {info['url']}")
        lines.append("─" * 80)

        # 지원 포맷
        lines.append("  지원 포맷:")
        for fmt in info['formats']:
            lines.append(f"    • {fmt}")

        # 처리 기능
        lines.append("  처리 기능:")
        for category, items in info['processing']:
            items_str = ', '.join(items)
            lines.append(f"    [{category}] {items_str}")

        # 모듈 통계
        lines.append("  모듈 구성:")
        for mod_name, mod_info in info['modules'].items():
            n_func = mod_info.get('n_functions', '?')
            n_lines = mod_info.get('lines', mod_info.get('total_lines', '?'))
            lines.append(f"    {mod_name}: {n_func} functions, {n_lines} lines")

        # 클래스
        if info.get('classes'):
            lines.append("  핵심 클래스:")
            for cls in info['classes']:
                n_methods = len(cls['methods'])
                lines.append(f"    {cls['class_name']}: {n_methods} public methods"
                             f" — {cls['doc']}")

        # gprMax 전용: .in 명령어
        if 'in_commands' in info and info['in_commands']:
            lines.append("  .in 파일 명령어:")
            for category, cmds in info['in_commands'].items():
                cmds_str = ', '.join(cmds)
                lines.append(f"    [{category}] {cmds_str}")

    # ── 기능 비교표 ──
    lines.append("")
    lines.append("=" * 80)
    lines.append("  전처리 기능 비교 (Week 2 파이프라인 vs 오픈소스)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(format_comparison_table(comparison))

    # ── 종합 분석 ──
    lines.append("")
    lines.append("=" * 80)
    lines.append("  종합 분석")
    lines.append("=" * 80)
    lines.append("")
    lines.append("  1. GPRPy: 가장 완전한 전처리 도구")
    lines.append("     - Week 2 파이프라인의 대부분 기능을 포함 (dewow, gain, migration)")
    lines.append("     - 추가: 토포그래피 보정, 속도 분석(CMP/WARR), VTK 3D 출력")
    lines.append("     - 한계: GUI 위주 설계, 모듈 단독 사용 불편, pip 설치 깨짐")
    lines.append("")
    lines.append("  2. gprMax: 시뮬레이션 전문 (FDTD)")
    lines.append("     - 전처리 기능 없음 → 시뮬레이션 도구")
    lines.append("     - .in 텍스트 파일로 복잡한 지하 모델 정의 가능")
    lines.append("     - 싱크홀 합성 B-scan 생성에 이상적 (Week 4 학습 데이터)")
    lines.append("     - 한계: Windows Cython 빌드 필요 (MSVC Build Tools)")
    lines.append("")
    lines.append("  3. siina: 경량 I/O 라이브러리")
    lines.append("     - DZT 포맷 읽기 + Butterworth 필터만 지원")
    lines.append("     - 최소 기능, 빠른 설치, 의존성 적음")
    lines.append("     - 한계: DZT 전용, 전처리 기능 매우 제한적")
    lines.append("")
    lines.append("  Week 2 파이프라인 위치:")
    lines.append("     - GPRPy보다 체계적 파이프라인 (6단계 순차 실행 + DB 기록)")
    lines.append("     - GPRPy에 없는 Background Removal, Amplitude Spectrum 포함")
    lines.append("     - 4개 포맷 지원 (DT1, DZT 헤더, IDS .dt, gprMax HDF5 예정)")
    lines.append("")

    return '\n'.join(lines)


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Week 3 - GPR 오픈소스 프로젝트 분석")
    print("=" * 60)

    # 분석 실행
    print("\n[1] GPRPy 분석...")
    gprpy_info = analyze_gprpy()
    n_funcs_gpy = sum(m.get('n_functions', 0) for m in gprpy_info['modules'].values())
    print(f"  모듈: {len(gprpy_info['modules'])}개, "
          f"함수: {n_funcs_gpy}개, "
          f"포맷: {len(gprpy_info['formats'])}종")

    print("\n[2] gprMax 분석...")
    gprmax_info = analyze_gprmax()
    n_cmds = sum(len(v) for v in gprmax_info['in_commands'].values())
    print(f"  .in 명령어: {n_cmds}개, "
          f"지오메트리: {len(gprmax_info['geometries'])}종, "
          f"소스: {len(gprmax_info['sources'])}종")

    print("\n[3] siina 분석...")
    siina_info = analyze_siina()
    n_funcs_si = sum(m.get('n_functions', 0) for m in siina_info['modules'].values())
    print(f"  모듈: {len(siina_info['modules'])}개, "
          f"함수: {n_funcs_si}개, "
          f"포맷: {len(siina_info['formats'])}종")

    # 비교표
    print("\n[4] Week 2 파이프라인 비교...")
    comparison = compare_with_week2()

    # 요약 생성
    summary = generate_summary_text(gprpy_info, gprmax_info, siina_info,
                                     comparison)

    # 파일 저장
    summary_path = OUTPUT_DIR / "analysis_summary.txt"
    summary_path.write_text(summary, encoding='utf-8')
    print(f"\n  요약 저장: {summary_path}")

    # JSON 상세 데이터 저장
    detail = {
        'gprpy': {
            'formats': gprpy_info['formats'],
            'processing': gprpy_info['processing'],
            'classes': [{'name': c['class_name'],
                        'n_methods': len(c['methods'])}
                       for c in gprpy_info['classes']],
        },
        'gprmax': {
            'in_commands': gprmax_info['in_commands'],
            'materials': gprmax_info['materials'],
            'geometries': gprmax_info['geometries'],
            'sources': gprmax_info['sources'],
            'waveforms': gprmax_info['waveforms'],
        },
        'siina': {
            'formats': siina_info['formats'],
            'processing': siina_info['processing'],
        },
    }
    detail_path = OUTPUT_DIR / "analysis_detail.json"
    detail_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2),
                           encoding='utf-8')
    print(f"  상세 JSON: {detail_path}")

    # 콘솔 출력
    print("\n" + summary)
    print("\n완료!")
