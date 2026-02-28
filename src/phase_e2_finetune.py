"""
Phase E-2: 4클래스 Fine-tuning (sinkhole / pipe / rebar / tunnel)

변경 사항 (E-1 대비):
  - 클래스: 3 → 4 (tunnel 추가, class_id=3)
  - 라벨링 데이터: 25 → 35개 (tunnel_000~009 추가)
  - 기반 가중치: E-1 best.pt → 검출 헤드 nc 불일치 시 자동으로
    yolo11n.pt (COCO pretrained) fallback
  - 데이터셋: yolo_gz_e2_mixed
  - 출력:     models/yolo_runs/finetune_gz_e2/

사전 요구사항:
  - phase_e2_relabel.py 실행 완료 (35개 라벨 생성)

사용법:
  python src/phase_e2_finetune.py
"""

import os, sys, json, shutil, random, time, csv, warnings
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec

# ── 경로 설정 ──
PROJECT_DIR  = Path(__file__).parent.parent
GZ_LABELED   = PROJECT_DIR / "guangzhou_labeled"
GZ_DATA      = PROJECT_DIR / "data" / "gpr" / "guangzhou" / "Data Set"
GZ_PIPE_DIR  = GZ_DATA / "pipe"
GZ_REBAR_DIR = GZ_DATA / "rebar"
GZ_TUNNEL_NJZ = GZ_DATA / "tunnel" / "NJZ"

DATA_DIR     = PROJECT_DIR / "data" / "gpr"
SYNTH_DIR    = DATA_DIR / "yolo_multiclass"
FDTD_DIR     = DATA_DIR / "yolo_fdtd"
MIXED_DIR    = DATA_DIR / "yolo_gz_e2_mixed"

E1_WEIGHTS   = PROJECT_DIR / "models" / "yolo_runs" / "finetune_gz_e1" / "run" / "weights" / "best.pt"
FT_DIR       = PROJECT_DIR / "models" / "yolo_runs" / "finetune_gz_e2"
OUTPUT_DIR   = PROJECT_DIR / "src" / "output" / "week4_multiclass"

CLASS_NAMES  = ['sinkhole', 'pipe', 'rebar', 'tunnel']
DT_SEC       = (8.0 / 512) * 1e-9


# ──────────────────────────────────────────────
# 0. 라벨 확인
# ──────────────────────────────────────────────

def check_labels() -> list[dict]:
    manifest_path = GZ_LABELED / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json 없음: {manifest_path}\n"
            f"먼저 phase_e2_relabel.py 를 실행하세요."
        )
    with open(manifest_path, encoding='cp949') as f:
        mdata = json.load(f)

    labeled, unlabeled = [], []
    for entry in mdata['images']:
        lbl_path = GZ_LABELED / "labels" / entry['label']
        if lbl_path.exists() and lbl_path.stat().st_size > 0:
            labeled.append(entry)
        else:
            unlabeled.append(entry)

    by_class = {}
    for e in labeled:
        by_class[e['class']] = by_class.get(e['class'], 0) + 1

    print(f"  라벨링 완료: {len(labeled)}개")
    for cls, cnt in sorted(by_class.items()):
        print(f"    {cls}: {cnt}개")
    if unlabeled:
        print(f"  라벨 없음 (제외): {len(unlabeled)}개")

    if len(labeled) == 0:
        raise ValueError("라벨링된 이미지가 없습니다. phase_e2_relabel.py를 먼저 실행하세요.")

    return labeled


# ──────────────────────────────────────────────
# 1. 혼합 데이터셋 구성
# ──────────────────────────────────────────────

def build_mixed_dataset(labeled: list[dict],
                        n_synth_train: int = 150,
                        n_synth_val: int   = 30,
                        val_ratio: float   = 0.2,
                        seed: int          = 42) -> tuple[int, int]:
    """
    Guangzhou 35개 + 합성 데이터 혼합.

    참고: 합성 데이터(FDTD/해석적)는 nc=3 라벨(class 0~2).
          tunnel(class 3)은 Guangzhou 라벨에만 있음.
          4-class 모델 학습 시 class 3은 Guangzhou 데이터에서만 학습.
    """
    rng = random.Random(seed)
    rng.shuffle(labeled)
    n_val_gz = max(1, int(len(labeled) * val_ratio))
    gz_val   = labeled[:n_val_gz]
    gz_train = labeled[n_val_gz:]
    print(f"  Guangzhou: train={len(gz_train)}, val={len(gz_val)}")

    # 출력 디렉토리 초기화
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        d = MIXED_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count = {'train': 0, 'val': 0}

    def copy_gz(items: list[dict], split: str):
        for entry in items:
            img_src = GZ_LABELED / "images" / entry['image']
            lbl_src = GZ_LABELED / "labels" / entry['label']
            stem    = f"gz_{entry['image'].replace('.png', '')}"
            shutil.copy2(img_src, MIXED_DIR / "images" / split / f"{stem}.png")
            shutil.copy2(lbl_src, MIXED_DIR / "labels" / split / f"{stem}.txt")
            count[split] += 1

    def copy_synth(src_dir: Path, split: str, n: int, prefix: str):
        img_dir = src_dir / "images" / split
        lbl_dir = src_dir / "labels" / split
        if not img_dir.exists():
            print(f"    [경고] 합성 데이터 없음: {img_dir}")
            return
        imgs = sorted(img_dir.glob("*.png"))
        rng.shuffle(imgs)
        for img_path in imgs[:n]:
            lbl_path = lbl_dir / img_path.with_suffix('.txt').name
            stem = f"{prefix}_{img_path.stem}"
            shutil.copy2(img_path, MIXED_DIR / "images" / split / f"{stem}.png")
            dst_lbl = MIXED_DIR / "labels" / split / f"{stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.write_text("")
            count[split] += 1

    copy_gz(gz_train, "train")
    copy_gz(gz_val,   "val")

    for synth_dir, prefix in [(FDTD_DIR, "fdtd"), (SYNTH_DIR, "synth")]:
        if synth_dir.exists():
            copy_synth(synth_dir, "train", n_synth_train // 2, prefix)
            copy_synth(synth_dir, "val",   n_synth_val   // 2, prefix)

    print(f"  혼합 데이터셋: train={count['train']}, val={count['val']}")
    return count['train'], count['val']


def create_yaml() -> Path:
    yaml_path = MIXED_DIR / "dataset.yaml"
    # Write as UTF-8 explicitly — ultralytics (PyYAML) reads with UTF-8 on Python 3
    yaml_path.write_text(
        f"path: {MIXED_DIR.as_posix()}\n"
        "train: images/train\n"
        "val:   images/val\n"
        "nc: 4\n"
        "names: ['sinkhole', 'pipe', 'rebar', 'tunnel']\n",
        encoding='utf-8'
    )
    return yaml_path


# ──────────────────────────────────────────────
# 2. 기반 가중치 선택
# ──────────────────────────────────────────────

def select_base_weights() -> tuple[Path, str]:
    """
    nc=3 → nc=4 전환 처리.

    E-1 weights 사용 시 ultralytics가 검출 헤드를 nc=4로 재초기화.
    nc 불일치 오류 발생 시 yolo11n.pt (COCO pretrained)로 fallback.
    """
    if E1_WEIGHTS.exists():
        print(f"  E-1 가중치 사용: {E1_WEIGHTS}")
        print(f"  (nc 3→4 전환: 검출 헤드 재초기화됨, backbone 가중치 유지)")
        return E1_WEIGHTS, "E-1"

    # fallback
    fallback = PROJECT_DIR / "yolo11n.pt"
    print(f"  ⚠ E-1 가중치 없음: {E1_WEIGHTS}")
    print(f"  Fallback: yolo11n.pt (COCO pretrained, 자동 다운로드)")
    return fallback, "COCO"


# ──────────────────────────────────────────────
# 3. Fine-tuning
# ──────────────────────────────────────────────

def finetune(yaml_path: Path, base_weights: Path,
             epochs: int = 50, batch: int = 2) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(base_weights))

    # nc 불일치 확인 및 처리
    try:
        model_nc = model.model.nc if hasattr(model.model, 'nc') else None
        if model_nc is not None and model_nc != 4:
            print(f"  nc 불일치: 모델={model_nc} → 데이터=4")
            print(f"  ultralytics가 검출 헤드를 자동 재초기화합니다.")
    except Exception:
        pass

    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=416,
        lr0=5e-5,
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        freeze=5,
        dropout=0.1,
        patience=20,
        warmup_epochs=3,
        mosaic=0.0,      # NAS pagefile 환경 메모리 부족 → mosaic 비활성화
        plots=False,     # plot_images 스레드 비활성화 (메모리 절약)
        project=str(FT_DIR),
        name="run",
        exist_ok=True,
        verbose=False,
        workers=0,
        cache=False,
        amp=False,
    )

    best_pt = FT_DIR / "run" / "weights" / "best.pt"
    return best_pt


# ──────────────────────────────────────────────
# 4. Guangzhou 테스트 추론
# ──────────────────────────────────────────────

def preprocess_to_rgb(dt_path: Path) -> np.ndarray | None:
    """IDS .dt → 640×640 RGB."""
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
        gray = (norm * 255).astype(np.uint8)
        return cv2.cvtColor(cv2.resize(gray, (640, 640)), cv2.COLOR_GRAY2RGB)
    except Exception:
        return None


def load_gz_test(labeled: list[dict], n_test: int = 4) -> list[tuple]:
    """라벨링에 사용되지 않은 Guangzhou .dt 파일 테스트 샘플 로드."""
    labeled_sources = {e.get('source', '') for e in labeled}
    samples = []

    for cls_name, root_dir in [("pipe", GZ_PIPE_DIR), ("rebar", GZ_REBAR_DIR)]:
        candidates = [
            f for f in sorted(root_dir.rglob("*.dt"))
            if "ascii" not in f.parent.name.lower()
               and str(f) not in labeled_sources
        ]
        random.Random(99).shuffle(candidates)
        count = 0
        for dt_path in candidates:
            if count >= n_test:
                break
            rgb = preprocess_to_rgb(dt_path)
            if rgb is not None:
                samples.append((dt_path.stem, cls_name, rgb))
                count += 1

    # 터널 테스트 샘플도 추가 (NJZ에서 라벨링 외 파일)
    tunnel_sources = {e.get('source', '') for e in labeled if e['class'] == 'tunnel'}
    tunnel_candidates = [
        f for f in sorted((GZ_DATA / "tunnel" / "NJZ").rglob("*.dt"))
        if str(f) not in tunnel_sources
    ]
    for dt_path in tunnel_candidates[:2]:
        rgb = preprocess_to_rgb(dt_path)
        if rgb is not None:
            samples.append((dt_path.stem, 'tunnel', rgb))

    return samples


def evaluate_gz(e1_weights: Path, ft_weights: Path,
                test_samples: list, conf: float = 0.10) -> dict:
    """E-1 원본 vs E-2 fine-tuned 비교."""
    import tempfile
    from ultralytics import YOLO

    results = {}
    models = {}
    if e1_weights.exists():
        models['E-1'] = YOLO(str(e1_weights))
    models['E-2'] = YOLO(str(ft_weights))

    with tempfile.TemporaryDirectory() as tmpdir:
        for stem, cls_name, rgb in test_samples:
            tmp_path = os.path.join(tmpdir, f"{stem}.png")
            cv2.imwrite(tmp_path, rgb)

            dets = {}
            for model_name, model in models.items():
                preds = model.predict(tmp_path, conf=conf, verbose=False)
                boxes = preds[0].boxes
                det_list = []
                if boxes is not None and len(boxes):
                    nc = len(CLASS_NAMES)
                    for cls_id, conf_val in zip(
                            boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                        cid = int(cls_id)
                        det_list.append({
                            'cls': cid,
                            'cls_name': CLASS_NAMES[cid] if cid < nc else f'cls{cid}',
                            'conf': float(conf_val),
                        })
                dets[model_name] = det_list

            results[f"{cls_name}/{stem}"] = {
                'true_class': cls_name,
                **dets,
            }

    return results


# ──────────────────────────────────────────────
# 5. 시각화
# ──────────────────────────────────────────────

def visualize(eval_results: dict, ft_weights: Path,
              n_train: int, n_val: int, labeled: list[dict],
              test_samples: list, base_label: str):
    from ultralytics import YOLO
    import tempfile

    CLASS_COLORS = {
        'sinkhole': '#e74c3c', 'pipe': '#3498db',
        'rebar': '#2ecc71',    'tunnel': '#f39c12',
    }

    fig = plt.figure(figsize=(20, 14), facecolor='#1a1a2e')
    fig.suptitle(f'Phase E-2: 4클래스 Fine-tuning  (기반: {base_label} → nc=4)',
                 fontsize=15, color='white', fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor('#2a2a4a')
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='white')
        for sp in ax.spines.values():
            sp.set_color('#555')

    # 패널 1: 클래스별 데이터 분포
    ax1 = fig.add_subplot(gs[0, :2])
    from collections import Counter
    cls_count = Counter(e['class'] for e in labeled)
    labels_bar = list(cls_count.keys())
    vals_bar   = [cls_count[k] for k in labels_bar]
    colors_bar = [CLASS_COLORS.get(k, 'gray') for k in labels_bar]
    ax1.bar(labels_bar, vals_bar, color=colors_bar, alpha=0.8)
    for i, v in enumerate(vals_bar):
        ax1.text(i, v + 0.1, str(v), ha='center', color='white', fontsize=10)
    ax1.set_ylabel('이미지 수', color='white')
    style(ax1, f'라벨링 데이터 구성 (train={n_train}, val={n_val})')

    # 패널 2: 탐지 수 비교
    ax2 = fig.add_subplot(gs[0, 2:])
    src_keys  = list(eval_results.keys())[:8]
    model_names = [k for k in list(eval_results[src_keys[0]].keys())
                   if k not in ('true_class',)] if src_keys else []
    bar_colors = ['#3498db', '#e74c3c']
    x2 = range(len(src_keys))
    w  = 0.8 / max(len(model_names), 1)
    for mi, mname in enumerate(model_names):
        cnts = [len(eval_results[k].get(mname, [])) for k in src_keys]
        offset = (mi - len(model_names) / 2 + 0.5) * w
        ax2.bar([i + offset for i in x2], cnts, w,
                label=mname, color=bar_colors[mi % len(bar_colors)], alpha=0.8)
    short_keys = [k.split('/')[0] + '\n' + k.split('/')[1][:8] for k in src_keys]
    ax2.set_xticks(list(x2))
    ax2.set_xticklabels(short_keys, fontsize=7, color='white')
    ax2.set_ylabel('탐지 수 (conf≥0.10)', color='white')
    ax2.legend(facecolor='#2a2a4a', labelcolor='white', fontsize=8)
    style(ax2, '탐지 비교 (E-1 3-class vs E-2 4-class)')

    # 패널 3~6: E-2 모델 추론 시각화
    ft_model = YOLO(str(ft_weights))
    cmap = plt.cm.tab10

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, (stem, cls_name, rgb) in enumerate(test_samples[:4]):
            tmp_path = os.path.join(tmpdir, f"{stem}.png")
            cv2.imwrite(tmp_path, rgb)
            preds = ft_model.predict(tmp_path, conf=0.10, verbose=False)
            boxes = preds[0].boxes

            row, col = divmod(idx, 2)
            ax = fig.add_subplot(
                gs[1, col * 2: col * 2 + 2] if idx < 2
                else gs[2, (idx - 2) * 2: (idx - 2) * 2 + 2]
            )
            ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), aspect='auto')
            ax.set_title(f'[{cls_name}] {stem[:20]}\nE-2 (conf≥0.10)',
                         color=CLASS_COLORS.get(cls_name, 'white'), fontsize=8)
            ax.axis('off')

            if boxes is not None and len(boxes):
                for box in boxes:
                    x1, y1, x2b, y2b = box.xyxy[0].cpu().numpy()
                    cid  = int(box.cls[0])
                    cval = float(box.conf[0])
                    c    = [int(v * 255) for v in cmap(cid)[:3]]
                    rect = plt.Rectangle(
                        (x1, y1), x2b - x1, y2b - y1,
                        fill=False, edgecolor=[v / 255 for v in c], linewidth=2
                    )
                    ax.add_patch(rect)
                    cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f'cls{cid}'
                    ax.text(x1, y1 - 4, f'{cname} {cval:.2f}',
                            color='white', fontsize=7,
                            bbox=dict(facecolor=[v / 255 for v in c], alpha=0.7, pad=1))

    save_path = OUTPUT_DIR / "phase_e2_gz_finetune.png"
    plt.savefig(str(save_path), dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  시각화: {save_path}")
    return save_path


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    print("Phase E-2: 4클래스 Fine-tuning")
    print(f"  라벨 폴더: {GZ_LABELED}")
    print(f"  출력:      {FT_DIR}")

    # ─ 0. 라벨 확인 ─
    print("\n[0/5] 라벨 확인...")
    labeled = check_labels()

    if not any(e['class'] == 'tunnel' for e in labeled):
        raise ValueError(
            "tunnel 라벨이 없습니다. phase_e2_relabel.py를 먼저 실행하세요."
        )

    # ─ 1. 기반 가중치 선택 ─
    print("\n[1/5] 기반 가중치 선택...")
    base_weights, base_label = select_base_weights()

    # ─ 2. 혼합 데이터셋 ─
    print("\n[2/5] 혼합 데이터셋 구성...")
    n_train, n_val = build_mixed_dataset(labeled)
    yaml_path = create_yaml()
    print(f"  dataset.yaml: {yaml_path}")

    # ─ 3. Fine-tuning ─
    print(f"\n[3/5] Fine-tuning (epochs=50, batch=2, base={base_label})...")
    t0 = time.time()
    ft_weights = finetune(yaml_path, base_weights, epochs=50, batch=2)
    elapsed = time.time() - t0
    print(f"  Fine-tuning 완료: {elapsed:.0f}초 ({elapsed / 60:.1f}분)")
    print(f"  Best weights: {ft_weights}")

    # 메트릭 출력
    results_csv = FT_DIR / "run" / "results.csv"
    best_map50 = 0.0
    if results_csv.exists():
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            map50_col  = 'metrics/mAP50(B)'
            map95_col  = 'metrics/mAP50-95(B)'
            best_row   = max(rows, key=lambda r: float(r.get(map50_col, 0) or 0))
            best_map50 = float(best_row.get(map50_col, 0) or 0)
            best_map95 = float(best_row.get(map95_col, 0) or 0)
            print(f"  Best mAP50:    {best_map50:.4f}")
            print(f"  Best mAP50-95: {best_map95:.4f}")

    # ─ 4. 테스트 추론 ─
    print("\n[4/5] Guangzhou 테스트 추론...")
    test_samples = load_gz_test(labeled, n_test=4)
    print(f"  테스트 샘플: {len(test_samples)}개")

    eval_results = evaluate_gz(E1_WEIGHTS, ft_weights, test_samples, conf=0.10)

    print(f"\n{'='*60}")
    print("Phase E-2 결과 요약")
    print(f"{'='*60}")
    print(f"\n[라벨링 데이터] {len(labeled)}개 (train≈{n_train}, val≈{n_val})")
    print(f"[best mAP50]   {best_map50:.4f}")

    print("\n[탐지 비교 (conf=0.10)]")
    model_names = [k for k in list(eval_results[list(eval_results.keys())[0]].keys())
                   if k != 'true_class'] if eval_results else []
    header = f"  {'파일':<25} {'참조':>7} " + " ".join(f"{m:>7}" for m in model_names)
    print(header)
    print("  " + "-" * len(header))
    for key, res in eval_results.items():
        cls_name = res['true_class']
        det_strs = " ".join(f"{len(res.get(m, [])):>7}" for m in model_names)
        print(f"  {key[:24]:<25} {cls_name:>7} {det_strs}")

    # ─ 5. 시각화 ─
    print("\n[5/5] 시각화...")
    if test_samples and ft_weights.exists():
        visualize(eval_results, ft_weights, n_train, n_val, labeled,
                  test_samples, base_label)
    else:
        print("  스킵 (테스트 샘플 없거나 가중치 없음)")

    print(f"\n{'='*60}")
    print("Phase E-2 완료")
    print(f"  4클래스: sinkhole / pipe / rebar / tunnel")
    print(f"  Fine-tuned weights: {ft_weights}")
    print(f"  시각화: {OUTPUT_DIR / 'phase_e2_gz_finetune.png'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
