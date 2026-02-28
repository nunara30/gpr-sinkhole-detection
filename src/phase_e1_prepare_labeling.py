"""
Phase E-1: Guangzhou 실측 데이터 라벨링 준비

Guangzhou 데이터셋 (IDS .dt) 에서 다양한 파일을 선택하여
전처리 후 640×640 PNG로 저장 → 수동 라벨링 준비.

선택 기준:
  - pipe:  15개 (ZON 디렉토리 다양성 기준, 상위 MIS 폴더 분산)
  - rebar: 10개 (ZON 디렉토리별 1개)
  - 합계:  25개

출력 구조:
  guangzhou_labeled/
  ├── images/         ← pipe_000.png ~ pipe_014.png, rebar_000.png ~ rebar_009.png
  ├── labels/         ← 빈 .txt 파일 (라벨링 후 YOLO 형식으로 작성)
  ├── manifest.json   ← 소스 경로 + 클래스 매핑
  ├── dataset.yaml    ← Phase E-1 fine-tuning용 YOLO 설정
  └── preview.png     ← 25개 이미지 미리보기 그리드

사용법:
  python phase_e1_prepare_labeling.py
  → guangzhou_labeled/images/ 의 PNG 파일을 라벨링 툴(LabelImg 등)로 열어
    YOLO 형식(.txt)으로 bbox 라벨 작성
  → Phase E-1 fine-tuning 실행: python phase_e1_finetune.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec

# ── 경로 설정 ──
PROJECT_DIR = Path(__file__).parent.parent        # gpr-sinkhole-detection/
GZ_DATA     = PROJECT_DIR / "data" / "gpr" / "guangzhou" / "Data Set"
GZ_PIPE_DIR  = GZ_DATA / "pipe"
GZ_REBAR_DIR = GZ_DATA / "rebar"
OUT_DIR      = PROJECT_DIR / "guangzhou_labeled"

N_PIPE  = 15
N_REBAR = 10

# Guangzhou 2GHz 데이터 전처리 파라미터
DT_NS    = 8.0 / 512        # ~0.015625 ns/sample (TW=8ns, 512 samples)
DT_SEC   = DT_NS * 1e-9
F_LOW_MHZ  = 500.0
F_HIGH_MHZ = 4000.0

CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']


# ──────────────────────────────────────────────
# 1. .dt 파일 선택
# ──────────────────────────────────────────────

def collect_dt_files(root_dir: Path, n: int, skip_ascii: bool = True) -> list[Path]:
    """
    ZON 디렉토리 다양성 기준으로 .dt 파일 선택.
    ASCII 하위 폴더는 중복이므로 기본적으로 스킵.
    """
    # ZON 디렉토리별로 하나씩 수집
    zon_to_files: dict[str, list[Path]] = defaultdict(list)

    for dt_file in sorted(root_dir.rglob("*.dt")):
        # ASCII 폴더 스킵
        if skip_ascii and "ascii" in dt_file.parent.name.lower():
            continue
        # ZON 디렉토리 키 (부모 디렉토리)
        zon_key = dt_file.parent
        zon_to_files[str(zon_key)].append(dt_file)

    # 각 ZON 에서 첫 번째 파일만 선택
    candidates = []
    for zon_key in sorted(zon_to_files.keys()):
        candidates.append(zon_to_files[zon_key][0])

    # MIS 폴더 분산 (파이프는 여러 MIS 폴더가 있음)
    mis_groups: dict[str, list[Path]] = defaultdict(list)
    for f in candidates:
        # MIS 디렉토리: root_dir 바로 아래
        try:
            rel = f.relative_to(root_dir)
            mis = rel.parts[0]
        except ValueError:
            mis = "unknown"
        mis_groups[mis].append(f)

    # 라운드 로빈으로 MIS 분산 선택
    selected = []
    mis_keys = sorted(mis_groups.keys())
    mis_iters = {k: iter(v) for k, v in mis_groups.items()}
    while len(selected) < n:
        added = False
        for k in mis_keys:
            if len(selected) >= n:
                break
            try:
                selected.append(next(mis_iters[k]))
                added = True
            except StopIteration:
                pass
        if not added:
            break  # 더 이상 파일 없음

    return selected[:n]


# ──────────────────────────────────────────────
# 2. 전처리 + PNG 변환
# ──────────────────────────────────────────────

def preprocess_to_png(dt_path: Path, out_path: Path) -> bool:
    """
    IDS .dt 파일을 읽어 전처리 후 640×640 PNG로 저장.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        data, _ = read_ids_dt(str(dt_path))
        if data is None or data.shape[1] < 10:
            print(f"    [SKIP] {dt_path.name}: 데이터 부족 ({data.shape if data is not None else 'None'})")
            return False

        # 전처리 파이프라인 (Guangzhou 2GHz 기준)
        d = dc_removal(data)
        d = background_removal(d)
        d = bandpass_filter(d, DT_SEC, F_LOW_MHZ, F_HIGH_MHZ)
        d = gain_sec(d, tpow=1.0, alpha=0.0, dt=DT_SEC)

        # 정규화 → uint8
        mn, mx = np.percentile(d, [2, 98])
        norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
        gray = (norm * 255).astype(np.uint8)

        # RGB 변환 + 640×640 리사이즈
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        rgb = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_LINEAR)

        # cv2.imwrite는 한글 경로에서 silent fail → imencode + write로 우회
        ok, buf = cv2.imencode('.png', rgb)
        if not ok:
            return False
        out_path.write_bytes(buf.tobytes())
        return True

    except Exception as e:
        print(f"    [ERROR] {dt_path.name}: {e}")
        return False


# ──────────────────────────────────────────────
# 3. 미리보기 그리드 생성
# ──────────────────────────────────────────────

def make_preview(manifest: list[dict], out_path: Path):
    """25개 이미지를 5×5 그리드로 미리보기."""
    n = len(manifest)
    cols = 5
    rows = (n + cols - 1) // cols

    CLASS_COLORS = {'pipe': '#3498db', 'rebar': '#2ecc71', 'sinkhole': '#e74c3c'}

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 0.8),
                             facecolor='#1a1a2e')
    fig.suptitle('Phase E-1: Guangzhou 라벨링 대상 이미지 (25개)',
                 color='white', fontsize=14, fontweight='bold')

    for idx, entry in enumerate(manifest):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]

        img_path = OUT_DIR / "images" / entry["image"]
        # cv2.imread는 한글 경로에서 silent fail → imdecode로 우회
        raw = img_path.read_bytes() if img_path.exists() else None
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if img is not None:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        color = CLASS_COLORS.get(entry["class"], 'white')
        ax.set_title(f"{entry['image']}\n[{entry['class']}]",
                     color=color, fontsize=7)

    # 빈 칸 채우기
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r][c] if rows > 1 else axes[c]
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  미리보기: {out_path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("Phase E-1: Guangzhou 라벨링 준비")
    print(f"  입력: {GZ_DATA}")
    print(f"  출력: {OUT_DIR}")

    # 디렉토리 생성
    (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    # ─ 1. 파일 선택 ─
    print(f"\n[1/4] 파일 선택 (pipe×{N_PIPE}, rebar×{N_REBAR})...")
    pipe_files  = collect_dt_files(GZ_PIPE_DIR,  N_PIPE)
    rebar_files = collect_dt_files(GZ_REBAR_DIR, N_REBAR)
    print(f"  pipe:  {len(pipe_files)}개 선택")
    print(f"  rebar: {len(rebar_files)}개 선택")

    if len(pipe_files) < N_PIPE:
        print(f"  ⚠ pipe 파일 부족: {len(pipe_files)}/{N_PIPE}")
    if len(rebar_files) < N_REBAR:
        print(f"  ⚠ rebar 파일 부족: {len(rebar_files)}/{N_REBAR}")

    # ─ 2. 전처리 + PNG 저장 ─
    print(f"\n[2/4] 전처리 + PNG 저장...")
    manifest = []
    total_ok = 0

    groups = [
        ("pipe",  pipe_files,  1),   # class_id=1
        ("rebar", rebar_files, 2),   # class_id=2
    ]

    for cls_name, files, cls_id in groups:
        idx = 0
        for dt_path in files:
            img_name = f"{cls_name}_{idx:03d}.png"
            img_out  = OUT_DIR / "images" / img_name
            lbl_out  = OUT_DIR / "labels" / img_name.replace(".png", ".txt")

            ok = preprocess_to_png(dt_path, img_out)
            if ok:
                # 빈 라벨 파일 생성 (라벨링 후 채울 것)
                lbl_out.write_text("")
                manifest.append({
                    "image":    img_name,
                    "label":    lbl_out.name,
                    "class":    cls_name,
                    "class_id": cls_id,
                    "source":   str(dt_path),
                })
                print(f"  ✓ {img_name}  ← {dt_path.parent.name}/{dt_path.name}")
                total_ok += 1
                idx += 1
            else:
                # 실패 시 다음 파일로 대체 시도하지 않음 (단순화)
                pass

    print(f"\n  저장 완료: {total_ok}/25개")

    # ─ 3. manifest.json + dataset.yaml ─
    print(f"\n[3/4] 메타데이터 저장...")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps({"images": manifest, "classes": CLASS_NAMES}, indent=2, ensure_ascii=False)
    )
    print(f"  manifest: {manifest_path}")

    yaml_path = OUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        f"train: images\n"
        f"val:   images\n"
        f"nc: 3\n"
        f"names: ['sinkhole', 'pipe', 'rebar']\n"
    )
    print(f"  dataset.yaml: {yaml_path}")

    # ─ 4. 미리보기 ─
    print(f"\n[4/4] 미리보기 그리드 생성...")
    make_preview(manifest, OUT_DIR / "preview.png")

    # ─ 결과 요약 ─
    print(f"\n{'='*60}")
    print(f"Phase E-1 완료")
    print(f"{'='*60}")
    print(f"  저장된 이미지: {total_ok}개")
    print(f"  출력 폴더:     {OUT_DIR}")
    print(f"\n다음 단계 (수동 라벨링):")
    print(f"  1. guangzhou_labeled/images/ 폴더를 라벨링 툴에서 열기")
    print(f"     권장 툴: LabelImg  →  pip install labelImg  →  labelImg")
    print(f"     또는:   Label Studio, CVAT, Roboflow")
    print(f"  2. YOLO 형식으로 bbox 라벨 작성")
    print(f"     - pipe  → class 1")
    print(f"     - rebar → class 2")
    print(f"  3. 라벨링 완료 후 Phase E-1 fine-tuning 실행:")
    print(f"     python phase_e1_finetune.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
