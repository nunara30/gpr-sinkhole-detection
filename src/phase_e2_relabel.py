"""
Phase E-2: 개선된 자동 라벨링 + Tunnel 클래스 추가

변경 사항 (E-1 대비):
  - bbox 감지: row/col std → connected components (더 타이트한 bbox)
  - 터널 이미지 10개 추가 (NJZ .dt, class_id=3)
  - 클래스 4개: sinkhole=0, pipe=1, rebar=2, tunnel=3
  - 총 35개 이미지

출력:
  - guangzhou_labeled/images/   : tunnel_000~009.png 추가
  - guangzhou_labeled/labels/   : 35개 .txt 재생성 (bbox 개선)
  - guangzhou_labeled/manifest.json : 35 entries, nc=4
  - guangzhou_labeled/dataset.yaml  : nc=4 업데이트
  - guangzhou_labeled/auto_label_review.png : 35개 검토 이미지

사용법:
  python src/phase_e2_relabel.py
"""

import sys, json, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec

# ── 경로 설정 ──
PROJECT_DIR  = Path(__file__).parent.parent
GZ_LABELED   = PROJECT_DIR / "guangzhou_labeled"
IMAGES_DIR   = GZ_LABELED / "images"
LABELS_DIR   = GZ_LABELED / "labels"
GZ_DATA      = PROJECT_DIR / "data" / "gpr" / "guangzhou" / "Data Set"
TUNNEL_NJZ   = GZ_DATA / "tunnel" / "NJZ"

CLASS_NAMES  = ['sinkhole', 'pipe', 'rebar', 'tunnel']
CLASS_ID     = {'sinkhole': 0, 'pipe': 1, 'rebar': 2, 'tunnel': 3}
CLASS_COLORS = {0: '#e74c3c', 1: '#3498db', 2: '#2ecc71', 3: '#f39c12'}

N_TUNNEL     = 10
DT_SEC       = (8.0 / 512) * 1e-9


# ──────────────────────────────────────────────
# 1. 전처리 공통
# ──────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    """IDS .dt → grayscale array (after pipeline). None on failure."""
    try:
        data, _ = read_ids_dt(str(dt_path))
        if data is None or data.shape[1] < 10:
            return None
        d = dc_removal(data)
        d = background_removal(d)
        d = bandpass_filter(d, DT_SEC, 500.0, 4000.0)
        d = gain_sec(d, tpow=1.0, alpha=0.0, dt=DT_SEC)
        mn, mx = np.percentile(d, [2, 98])
        norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
        return (norm * 255).astype(np.uint8)
    except Exception as e:
        print(f"    [ERROR] {dt_path.name}: {e}")
        return None


def save_png(path: Path, img: np.ndarray) -> bool:
    """한글 경로 안전 저장."""
    ok, buf = cv2.imencode('.png', img)
    if ok:
        path.write_bytes(buf.tobytes())
    return ok


def load_gray(path: Path) -> np.ndarray | None:
    """한글 경로 안전 로드 (grayscale)."""
    raw = path.read_bytes()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)


# ──────────────────────────────────────────────
# 2. 개선된 bbox 감지 (connected components)
# ──────────────────────────────────────────────

def detect_bbox_v2(img_gray: np.ndarray, cls_name: str):
    """
    연결 성분(connected components) 기반 bbox.

    E-1 문제점: col std 15th percentile → bw≈0.998 (거의 전폭)
    E-2 개선:   이진화 + 모폴로지 + CC → 타이트한 bbox

    Returns:
      (cx, cy, bw, bh, (x1, y1, x2, y2)) normalized to [0,1]
    """
    H, W = img_gray.shape
    skip_top = max(1, int(H * 0.05))   # 직접파 5% 스킵
    work = img_gray[skip_top:, :].astype(np.float32)

    # 클래스별 분석 구역 높이
    region_ratio = {'tunnel': 0.35, 'rebar': 0.50, 'pipe': 0.60}
    max_y = int(work.shape[0] * region_ratio.get(cls_name, 0.60))
    work_region = work[:max(5, max_y), :]

    # 이진화: 72nd percentile 이상
    thresh_val = np.percentile(work_region, 72)
    binary = (work_region > thresh_val).astype(np.uint8) * 255

    # 모폴로지 팽창 (가로로 불연속 성분 연결)
    kx = max(3, W // 30)
    ky = max(3, H // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # 연결 성분 분석
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dilated)

    if num_labels <= 1:
        return _fallback(img_gray, cls_name, skip_top)

    # 유효 성분 필터 (최소 크기 기준)
    min_w = int(W * 0.06)
    valid = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= min_w and h >= 3:
            valid.append({
                'x': x, 'y': y + skip_top,
                'x2': x + w, 'y2': y + h + skip_top,
                'w': w, 'area': area,
            })

    if not valid:
        return _fallback(img_gray, cls_name, skip_top)

    # 클래스별 성분 선택
    if cls_name == 'pipe':
        # 가장 넓은 성분 (파이프 = 수평 강반사 띠)
        best = max(valid, key=lambda c: c['w'])
        x1, y1, x2, y2 = best['x'], best['y'], best['x2'], best['y2']

    else:
        # rebar / tunnel: 상위 면적 N개 합치기
        n_top = 5 if cls_name == 'rebar' else 4
        top_n = sorted(valid, key=lambda c: c['area'], reverse=True)[:n_top]
        x1 = min(c['x']  for c in top_n)
        y1 = min(c['y']  for c in top_n)
        x2 = max(c['x2'] for c in top_n)
        y2 = max(c['y2'] for c in top_n)

    # 여백 추가
    pad_y = int(H * 0.02)
    pad_x = int(W * 0.02)
    x1 = max(0,     x1 - pad_x)
    y1 = max(0,     y1 - pad_y)
    x2 = min(W - 1, x2 + pad_x)
    y2 = min(H - 1, y2 + pad_y)

    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    bw = (x2 - x1) / W
    bh = (y2 - y1) / H
    return cx, cy, bw, bh, (x1, y1, x2, y2)


def _fallback(img_gray, cls_name, skip_top):
    """CC 실패 시 상단 비율 기반 fallback."""
    H, W = img_gray.shape
    y2_ratio = {'tunnel': 0.30, 'rebar': 0.45, 'pipe': 0.50}
    y1 = skip_top
    y2 = int(H * y2_ratio.get(cls_name, 0.50))
    cx = 0.5
    cy = (y1 + y2) / 2 / H
    bw = 1.0
    bh = (y2 - y1) / H
    return cx, cy, bw, bh, (0, y1, W - 1, y2)


# ──────────────────────────────────────────────
# 3. 터널 파일 선택
# ──────────────────────────────────────────────

def collect_tunnel_files(n: int = 10) -> list[Path]:
    """NJZ .dt 파일을 ZON 다양성 기준으로 n개 선택."""
    zon_map: dict[str, list[Path]] = defaultdict(list)
    for dt_file in sorted(TUNNEL_NJZ.rglob("*.dt")):
        zon_map[str(dt_file.parent)].append(dt_file)

    candidates = sorted(files[0] for files in zon_map.values())
    return candidates[:n]


# ──────────────────────────────────────────────
# 4. 검토 이미지
# ──────────────────────────────────────────────

def make_review(entries: list[dict], out_path: Path):
    n = len(entries)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3, rows * 3.2),
                             facecolor='#1a1a2e')
    axes_flat = np.array(axes).flatten()

    pipe_n   = sum(1 for e in entries if e['class'] == 'pipe')
    rebar_n  = sum(1 for e in entries if e['class'] == 'rebar')
    tunnel_n = sum(1 for e in entries if e['class'] == 'tunnel')
    fig.suptitle(
        f'Phase E-2 라벨링 검토  pipe×{pipe_n}  rebar×{rebar_n}  tunnel×{tunnel_n}',
        color='white', fontsize=11, fontweight='bold'
    )

    for idx, entry in enumerate(entries):
        ax = axes_flat[idx]
        raw = (IMAGES_DIR / entry['image']).read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if entry.get('bbox_px'):
            x1, y1, x2, y2 = entry['bbox_px']
            cls_id = entry['class_id']
            color  = CLASS_COLORS.get(cls_id, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, max(0, y1 - 5),
                    entry['class'], color='white', fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.7, pad=1))

        ax.set_title(entry['image'], color='white', fontsize=6)
        ax.axis('off')

    for idx in range(n, rows * cols):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=90, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  검토 이미지: {out_path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("Phase E-2: 개선된 라벨링 + Tunnel 추가")
    print(f"  이미지: {IMAGES_DIR}")
    print(f"  라벨:   {LABELS_DIR}")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # ─ 기존 manifest 로드 ─
    manifest_path = GZ_LABELED / "manifest.json"
    with open(manifest_path, encoding='utf-8') as f:
        mdata = json.load(f)
    existing = mdata['images']   # pipe×15, rebar×10

    # ─ 1. pipe/rebar bbox 재생성 ─
    print(f"\n[1/3] pipe/rebar bbox 재생성 (connected components, {len(existing)}개)...")
    updated: list[dict] = []
    bw_e1, bw_e2 = [], []

    for entry in existing:
        cls_name = entry['class']
        img_path = IMAGES_DIR / entry['image']
        lbl_path = LABELS_DIR / entry['label']

        if not img_path.exists():
            print(f"  [SKIP] {entry['image']}: 이미지 없음")
            continue

        # E-1 bbox 기록 (비교용)
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            try:
                parts = lbl_path.read_text().split()
                bw_e1.append(float(parts[3]))
            except Exception:
                pass

        img_gray = load_gray(img_path)
        cx, cy, bw, bh, bbox_px = detect_bbox_v2(img_gray, cls_name)
        lbl_path.write_text(f"{entry['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        bw_e2.append(bw)
        entry['bbox_px'] = bbox_px
        updated.append(entry)
        print(f"  OK {entry['image']:22s} [{cls_name:5s}] w={bw:.3f} h={bh:.3f}")

    if bw_e1 and bw_e2:
        print(f"\n  bbox 너비 평균  E-1: {np.mean(bw_e1):.3f} → E-2: {np.mean(bw_e2):.3f}")

    # ─ 2. 터널 이미지 추가 ─
    print(f"\n[2/3] Tunnel 이미지 추가 ({N_TUNNEL}개, NJZ)...")
    tunnel_files = collect_tunnel_files(N_TUNNEL)
    print(f"  선택된 .dt: {len(tunnel_files)}개")

    tunnel_entries: list[dict] = []
    for i, dt_path in enumerate(tunnel_files):
        img_name = f"tunnel_{i:03d}.png"
        img_path = IMAGES_DIR / img_name
        lbl_path = LABELS_DIR / img_name.replace('.png', '.txt')

        try:
            gray = preprocess_dt(dt_path)
            if gray is None:
                print(f"  [SKIP] {dt_path.name}: 전처리 실패")
                continue

            # 640×640 PNG 저장
            rgb = cv2.cvtColor(
                cv2.resize(gray, (640, 640), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_GRAY2RGB
            )
            save_png(img_path, rgb)

            # bbox 감지 (리사이즈 후 grayscale 사용)
            gray_640 = cv2.resize(gray, (640, 640), interpolation=cv2.INTER_LINEAR)
            cx, cy, bw, bh, bbox_px = detect_bbox_v2(gray_640, 'tunnel')

            lbl_path.write_text(f"3 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

            entry = {
                'image':    img_name,
                'label':    lbl_path.name,
                'class':    'tunnel',
                'class_id': 3,
                'source':   str(dt_path),
                'bbox_px':  bbox_px,
            }
            tunnel_entries.append(entry)
            print(f"  OK {img_name:22s} [tunnel] w={bw:.3f} h={bh:.3f}  <- {dt_path.parent.name}")
        except Exception as e:
            print(f"  [ERROR] {dt_path.name}: {e}")

    print(f"  터널 이미지 저장: {len(tunnel_entries)}개")

    # ─ 3. manifest + dataset.yaml 업데이트 ─
    print(f"\n[3/3] 메타데이터 업데이트 (nc=4)...")
    all_entries = updated + tunnel_entries

    # bbox_px 제외하여 JSON 직렬화
    manifest_entries = [
        {k: v for k, v in e.items() if k != 'bbox_px'}
        for e in all_entries
    ]
    manifest_path.write_text(
        json.dumps({'images': manifest_entries, 'classes': CLASS_NAMES},
                   indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"  manifest: {manifest_path} ({len(manifest_entries)}개)")

    yaml_path = GZ_LABELED / "dataset.yaml"
    yaml_path.write_text(
        f"path: {GZ_LABELED.as_posix()}\n"
        f"train: images\n"
        f"val:   images\n"
        f"nc: 4\n"
        f"names: ['sinkhole', 'pipe', 'rebar', 'tunnel']\n"
    )
    print(f"  dataset.yaml: {yaml_path} (nc=4)")

    # 검토 이미지
    make_review(all_entries, GZ_LABELED / "auto_label_review.png")

    # ─ 결과 요약 ─
    pipe_n   = sum(1 for e in all_entries if e['class'] == 'pipe')
    rebar_n  = sum(1 for e in all_entries if e['class'] == 'rebar')
    tunnel_n = sum(1 for e in all_entries if e['class'] == 'tunnel')

    print(f"\n{'='*60}")
    print(f"Phase E-2 라벨링 완료")
    print(f"{'='*60}")
    print(f"  총 이미지:      {len(all_entries)}개")
    print(f"    pipe:   {pipe_n}개  (class 1)")
    print(f"    rebar:  {rebar_n}개  (class 2)")
    print(f"    tunnel: {tunnel_n}개  (class 3)")
    print(f"  bbox 너비 E-2 평균: {np.mean(bw_e2):.3f}  (E-1: {np.mean(bw_e1):.3f})")
    print(f"\n다음 단계:")
    print(f"  python src/phase_e2_finetune.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
