"""
Phase C: Pseudo-labeling (Semi-supervised)

실측 Guangzhou 데이터에 다중 클래스 모델을 적용하여
고신뢰도 예측을 pseudo-label로 사용 → 모델 fine-tuning.

단계:
  1. 실측 데이터 로드 → PNG 변환
  2. conf=0.05 (매우 낮은 임계값) 으로 예측
  3. 기대 클래스와 일치하는 검출만 유지 (class-filtered pseudo-label)
  4. Pseudo-label 데이터셋 구성 (기존 합성 + 실측 pseudo-label)
  5. Fine-tuning (소규모 epoch, 낮은 lr)
  6. Fine-tuned 모델 vs 원본 모델 비교 평가
"""

import os
import sys
import time
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings('ignore')

from week1_gpr_basics import read_dt1, read_ids_dt
from week2_preprocessing import dc_removal, dewow, background_removal, bandpass_filter, gain_sec

# ── 경로 ──
BASE_DIR  = Path("G:/RAG_system")
DATA_DIR  = BASE_DIR / "data" / "gpr"
MC_DIR    = DATA_DIR / "yolo_multiclass"          # 기존 합성 데이터셋
PL_DIR    = DATA_DIR / "yolo_pseudolabel"         # pseudo-label 데이터셋
MULTI_WEIGHTS = BASE_DIR / "models/yolo_runs/multiclass_detect/weights/best.pt"
FT_RUNS_DIR   = BASE_DIR / "models/yolo_runs/finetune_pseudo"
OUTPUT_DIR    = BASE_DIR / "src" / "output" / "week4_multiclass"

for d in [PL_DIR / "images/train", PL_DIR / "images/val",
          PL_DIR / "labels/train", PL_DIR / "labels/val",
          OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']
CLASS_COLORS = {'sinkhole': '#ff4444', 'pipe': '#00ccff', 'rebar': '#ffee00'}

# 실측 데이터 설정
REAL_SOURCES = [
    {
        'name': 'GZ_pipe',
        'dir': DATA_DIR / "guangzhou/Data Set/pipe",
        'expected_cls': 1,      # pipe
        'expected_name': 'pipe',
        'max_files': 30,
        'bp': (500e6, 5e9),
        'dt_ns': 0.1,
    },
    {
        'name': 'GZ_rebar',
        'dir': DATA_DIR / "guangzhou/Data Set/rebar",
        'expected_cls': 2,      # rebar
        'expected_name': 'rebar',
        'max_files': 30,
        'bp': (500e6, 5e9),
        'dt_ns': 0.1,
    },
    {
        'name': 'GZ_tunnel',
        'dir': DATA_DIR / "guangzhou/Data Set/tunnel",
        'expected_cls': -1,     # background (탐지 없어야 함)
        'expected_name': 'background',
        'max_files': 10,
        'bp': (500e6, 5e9),
        'dt_ns': 0.1,
    },
]

# Pseudo-label conf 임계값
PSEUDO_CONF_THRESH = 0.05


def bscan_to_img(data: np.ndarray) -> np.ndarray:
    """B-scan → 640×640 grayscale uint8"""
    data = data.astype(np.float32)
    p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
    if p98 - p2 < 1e-10:
        return np.zeros((640, 640), dtype=np.uint8)
    img = np.clip((data - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    return cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)


def preprocess(data: np.ndarray, bp_low_hz: float, bp_high_hz: float,
               dt_ns: float = 0.1) -> np.ndarray:
    dt = dt_ns * 1e-9
    data = dc_removal(data)
    data = dewow(data, window=20)
    data = background_removal(data)
    data = bandpass_filter(data, dt, bp_low_hz / 1e6, bp_high_hz / 1e6)
    data = gain_sec(data, tpow=1.0, alpha=0.0, dt=dt)
    return data


def load_real_files(source: dict) -> list:
    """실측 .dt 파일 로드 → (img_raw, img_proc, stem) 리스트"""
    ds_dir = source['dir']
    if not ds_dir.exists():
        print(f"  [경고] {ds_dir} 없음")
        return []

    dt_files = sorted(f for f in ds_dir.rglob("*.dt") if 'ASCII' not in str(f))
    samples = []
    for dt_file in dt_files[:source['max_files']]:
        data, header = read_ids_dt(str(dt_file))
        if data is None:
            continue
        img_raw  = bscan_to_img(data)
        data_proc = preprocess(data, *source['bp'], source['dt_ns'])
        img_proc = bscan_to_img(data_proc)
        samples.append((img_raw, img_proc, dt_file.stem))
    return samples


# ─────────────────────────────────────────────
# 1. Pseudo-label 생성
# ─────────────────────────────────────────────

def generate_pseudo_labels(model, source: dict, samples: list) -> dict:
    """
    각 실측 이미지에 대해 예측 → expected class 검출만 유지
    반환: {'accepted': [(img, bboxes, stem)], 'rejected': int, 'total': int}
    """
    tmp_path = OUTPUT_DIR / "_tmp_pl.png"
    accepted = []
    total = 0

    for img_raw, img_proc, stem in samples:
        for img, version in [(img_raw, 'raw'), (img_proc, 'proc')]:
            total += 1
            cv2.imwrite(str(tmp_path), img)
            preds = model.predict(str(tmp_path), imgsz=640,
                                  conf=PSEUDO_CONF_THRESH, verbose=False)
            if not preds or preds[0].boxes is None:
                continue

            bboxes = []
            for box in preds[0].boxes:
                cls_id = int(box.cls[0].cpu().numpy())
                conf   = float(box.conf[0].cpu().numpy())
                xyxy   = box.xyxy[0].cpu().numpy()

                # 기대 클래스와 일치하는 검출만 유지
                if source['expected_cls'] == -1:
                    # background: 탐지가 있으면 FP이므로 스킵
                    break
                if cls_id == source['expected_cls']:
                    x1, y1, x2, y2 = xyxy
                    cx = (x1 + x2) / 2 / 640
                    cy = (y1 + y2) / 2 / 640
                    w  = (x2 - x1) / 640
                    h  = (y2 - y1) / 640
                    bboxes.append((cls_id, cx, cy, w, h))

            if bboxes:
                accepted.append((img, bboxes, f"{stem}_{version}"))

    tmp_path.unlink(missing_ok=True)
    return {
        'accepted': accepted,
        'rejected': total - len(accepted),
        'total': total,
    }


def save_pseudo_dataset(pl_results: dict, split_ratio: float = 0.8):
    """
    pseudo-label 데이터를 PL_DIR에 저장 + 기존 합성 데이터 복사
    """
    all_samples = []
    for src_name, result in pl_results.items():
        for img, bboxes, stem in result['accepted']:
            all_samples.append((img, bboxes, f"pl_{src_name}_{stem}"))

    if not all_samples:
        print("  [경고] pseudo-label 데이터 없음 - 합성 데이터만 사용")

    # 섞어서 train/val 분할
    random.shuffle(all_samples)
    n_train = int(len(all_samples) * split_ratio)
    train_pl = all_samples[:n_train]
    val_pl   = all_samples[n_train:]

    # pseudo-label 저장
    for split_name, split_data in [('train', train_pl), ('val', val_pl)]:
        for img, bboxes, stem in split_data:
            img_path = PL_DIR / "images" / split_name / f"{stem}.png"
            lbl_path = PL_DIR / "labels" / split_name / f"{stem}.txt"
            cv2.imwrite(str(img_path), img)
            with open(lbl_path, 'w') as f:
                for cls_id, cx, cy, w, h in bboxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    # 기존 합성 데이터 일부 복사 (과적합 방지를 위해 서브셋)
    synth_train_imgs = list((MC_DIR / "images/train").glob("*.png"))
    synth_val_imgs   = list((MC_DIR / "images/val").glob("*.png"))

    # 합성 200개 + pseudo-label 전부
    random.shuffle(synth_train_imgs)
    random.shuffle(synth_val_imgs)
    copy_train = synth_train_imgs[:200]
    copy_val   = synth_val_imgs[:50]

    for img_path in copy_train:
        lbl_path = MC_DIR / "labels/train" / img_path.with_suffix('.txt').name
        dst_img = PL_DIR / "images/train" / img_path.name
        dst_lbl = PL_DIR / "labels/train" / lbl_path.name
        shutil.copy2(img_path, dst_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            dst_lbl.write_text("")

    for img_path in copy_val:
        lbl_path = MC_DIR / "labels/val" / img_path.with_suffix('.txt').name
        dst_img = PL_DIR / "images/val" / img_path.name
        dst_lbl = PL_DIR / "labels/val" / lbl_path.name
        shutil.copy2(img_path, dst_img)
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl)
        else:
            dst_lbl.write_text("")

    n_train_total = len(list((PL_DIR / "images/train").glob("*.png")))
    n_val_total   = len(list((PL_DIR / "images/val").glob("*.png")))

    print(f"  pseudo-label: train={len(train_pl)}, val={len(val_pl)}")
    print(f"  합성 복사: train={len(copy_train)}, val={len(copy_val)}")
    print(f"  최종 데이터셋: train={n_train_total}, val={n_val_total}")

    return n_train_total, n_val_total


def create_pl_yaml() -> Path:
    yaml_path = PL_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {str(PL_DIR).replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 3\n"
        f"names: ['sinkhole', 'pipe', 'rebar']\n"
    )
    return yaml_path


# ─────────────────────────────────────────────
# 2. Fine-tuning
# ─────────────────────────────────────────────

def finetune(yaml_path: Path, base_weights: Path,
             epochs: int = 30, batch: int = 8) -> Path:
    """기존 모델에서 fine-tuning"""
    from ultralytics import YOLO
    model = YOLO(str(base_weights))

    t0 = time.perf_counter()
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        optimizer='AdamW',
        lr0=1e-4,           # 낮은 초기 lr (fine-tuning)
        lrf=0.01,
        warmup_epochs=2,
        patience=15,
        freeze=0,           # 전체 레이어 학습 허용 (적은 epoch)
        workers=0,
        label_smoothing=0.05,
        dropout=0.1,
        cos_lr=True,
        project=str(FT_RUNS_DIR),
        name='run',
        exist_ok=True,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Fine-tuning 완료: {elapsed:.0f}초")

    ft_weights = FT_RUNS_DIR / "run" / "weights" / "best.pt"
    return ft_weights


# ─────────────────────────────────────────────
# 3. 비교 평가
# ─────────────────────────────────────────────

def compare_models(original_weights: Path, ft_weights: Path,
                   samples_by_source: dict) -> dict:
    """원본 vs fine-tuned 모델을 conf=0.1, 0.25에서 비교"""
    from ultralytics import YOLO

    orig_model = YOLO(str(original_weights))
    ft_model   = YOLO(str(ft_weights)) if ft_weights.exists() else None

    tmp_path = OUTPUT_DIR / "_tmp_cmp.png"
    results = {}

    for src_name, samples in samples_by_source.items():
        results[src_name] = {'original': {}, 'finetune': {}}
        for conf in [0.10, 0.25]:
            orig_dets = {'sinkhole': 0, 'pipe': 0, 'rebar': 0}
            ft_dets   = {'sinkhole': 0, 'pipe': 0, 'rebar': 0}
            for img_raw, img_proc, stem in samples[:10]:
                for img in [img_raw, img_proc]:
                    cv2.imwrite(str(tmp_path), img)
                    # 원본
                    p = orig_model.predict(str(tmp_path), imgsz=640,
                                           conf=conf, verbose=False)
                    if p and p[0].boxes:
                        for box in p[0].boxes:
                            ci = int(box.cls[0].cpu().numpy())
                            if ci < len(CLASS_NAMES):
                                orig_dets[CLASS_NAMES[ci]] += 1
                    # fine-tuned
                    if ft_model:
                        p2 = ft_model.predict(str(tmp_path), imgsz=640,
                                              conf=conf, verbose=False)
                        if p2 and p2[0].boxes:
                            for box in p2[0].boxes:
                                ci = int(box.cls[0].cpu().numpy())
                                if ci < len(CLASS_NAMES):
                                    ft_dets[CLASS_NAMES[ci]] += 1

            results[src_name]['original'][conf] = sum(orig_dets.values())
            results[src_name]['finetune'][conf]  = sum(ft_dets.values())

    tmp_path.unlink(missing_ok=True)
    return results


def visualize_results(pl_results: dict, compare_results: dict,
                      samples_by_source: dict, ft_weights: Path):
    """결과 시각화"""
    from ultralytics import YOLO
    ft_model = YOLO(str(ft_weights)) if ft_weights.exists() else None

    fig = plt.figure(figsize=(18, 10), facecolor='#1a1a2e')
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.3)

    # ── 상단: pseudo-label 통계 ──
    ax_stat = fig.add_subplot(gs[0, :2])
    src_names = list(pl_results.keys())
    accepted = [pl_results[n]['accepted_count'] for n in src_names]
    total    = [pl_results[n]['total'] for n in src_names]
    x = np.arange(len(src_names))
    ax_stat.bar(x - 0.2, total,    0.35, color='#666', alpha=0.7, label='total frames')
    ax_stat.bar(x + 0.2, accepted, 0.35, color='#00cc88', alpha=0.9, label='pseudo-labeled')
    ax_stat.set_xticks(x)
    ax_stat.set_xticklabels(src_names, color='white', fontsize=10)
    ax_stat.set_ylabel('Count', color='white')
    ax_stat.set_title('Pseudo-label 수집 통계', color='white', fontsize=11, fontweight='bold')
    ax_stat.legend(fontsize=9)
    ax_stat.tick_params(colors='white')
    ax_stat.set_facecolor('#0d0d1a')
    for spine in ax_stat.spines.values(): spine.set_color('#444')

    # ── 상단 오른쪽: 비교 막대 ──
    ax_cmp = fig.add_subplot(gs[0, 2])
    src_list = list(compare_results.keys())
    orig_vals = [compare_results[s]['original'].get(0.10, 0) for s in src_list]
    ft_vals   = [compare_results[s]['finetune'].get(0.10, 0)  for s in src_list]
    xx = np.arange(len(src_list))
    ax_cmp.bar(xx - 0.2, orig_vals, 0.35, color='#ff7744', alpha=0.8, label='Original')
    ax_cmp.bar(xx + 0.2, ft_vals,   0.35, color='#44aaff', alpha=0.8, label='Fine-tuned')
    ax_cmp.set_xticks(xx)
    ax_cmp.set_xticklabels(src_list, color='white', fontsize=8, rotation=15)
    ax_cmp.set_title('탐지 수 비교\n(conf=0.10)', color='white', fontsize=10, fontweight='bold')
    ax_cmp.legend(fontsize=8)
    ax_cmp.tick_params(colors='white')
    ax_cmp.set_facecolor('#0d0d1a')
    for spine in ax_cmp.spines.values(): spine.set_color('#444')

    # ── 하단: 실측 이미지 + fine-tuned 예측 ──
    src_order = ['GZ_pipe', 'GZ_rebar', 'GZ_tunnel']
    for ci, src_name in enumerate(src_order):
        if src_name not in samples_by_source or not samples_by_source[src_name]:
            continue
        ax = fig.add_subplot(gs[1, ci])
        img_raw, img_proc, stem = samples_by_source[src_name][0]
        ax.imshow(img_proc, cmap='gray', aspect='auto')

        if ft_model:
            tmp_path = OUTPUT_DIR / "_tmp_vis.png"
            cv2.imwrite(str(tmp_path), img_proc)
            preds = ft_model.predict(str(tmp_path), imgsz=640, conf=0.05, verbose=False)
            tmp_path.unlink(missing_ok=True)
            if preds and preds[0].boxes:
                for box in preds[0].boxes:
                    ci2 = int(box.cls[0].cpu().numpy())
                    sc  = float(box.conf[0].cpu().numpy())
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cname = CLASS_NAMES[ci2] if ci2 < len(CLASS_NAMES) else '?'
                    color = CLASS_COLORS.get(cname, 'white')
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 3, f'{cname} {sc:.2f}', color=color,
                            fontsize=7, fontweight='bold')

        ax.set_title(f"{src_name}\n(fine-tuned, conf=0.05)",
                     color='white', fontsize=8)
        ax.axis('off')

    plt.suptitle('Phase C: Pseudo-labeling + Fine-tuning 결과',
                 fontsize=12, fontweight='bold', color='white', y=0.99)

    save_path = OUTPUT_DIR / "pseudolabel_finetune.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"\n  [저장] {save_path}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    print("Phase C: Pseudo-labeling + Fine-tuning")
    print(f"  base model: {MULTI_WEIGHTS}")
    print(f"  pseudo-label conf threshold: {PSEUDO_CONF_THRESH}")

    model = YOLO(str(MULTI_WEIGHTS))

    # ─── Step 1: 실측 데이터 로드 + pseudo-label 생성 ───
    print("\n[1/4] 실측 데이터 로드 및 pseudo-label 생성...")
    pl_results = {}
    samples_by_source = {}

    for src in REAL_SOURCES:
        print(f"\n  [{src['name']}] 로드 중...")
        samples = load_real_files(src)
        print(f"    파일 로드: {len(samples)}개")
        samples_by_source[src['name']] = samples

        result = generate_pseudo_labels(model, src, samples)
        accepted_count = len(result['accepted'])
        pl_results[src['name']] = {
            'accepted': result['accepted'],
            'accepted_count': accepted_count,
            'total': result['total'],
            'source': src,
        }
        print(f"    pseudo-label 수락: {accepted_count}/{result['total']} "
              f"({accepted_count/max(result['total'],1)*100:.1f}%)")

    # ─── Step 2: 데이터셋 구성 ───
    print("\n[2/4] pseudo-label 데이터셋 구성...")
    # PL_DIR 초기화
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        d = PL_DIR / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n_train, n_val = save_pseudo_dataset(pl_results)
    yaml_path = create_pl_yaml()
    print(f"  dataset.yaml: {yaml_path}")

    # ─── Step 3: Fine-tuning ───
    print("\n[3/4] Fine-tuning...")
    ft_weights = FT_RUNS_DIR / "run" / "weights" / "best.pt"

    if n_train < 10:
        print(f"  [스킵] 학습 데이터 부족 ({n_train}개) - pseudo-label 데이터가 너무 적음")
        print("  → 합성 데이터만으로 fine-tuning 진행")

    ft_weights = finetune(yaml_path, MULTI_WEIGHTS, epochs=30, batch=8)
    print(f"  fine-tuned weights: {ft_weights}")

    # ─── Step 4: 비교 평가 ───
    print("\n[4/4] 원본 vs fine-tuned 비교...")
    compare_results = compare_models(MULTI_WEIGHTS, ft_weights, samples_by_source)

    # 결과 출력
    print("\n" + "=" * 65)
    print("Phase C 결과 요약")
    print("=" * 65)

    print("\n[Pseudo-label 통계]")
    total_pl = 0
    for src_name, res in pl_results.items():
        acc = res['accepted_count']
        tot = res['total']
        total_pl += acc
        print(f"  {src_name:<15} {acc:>4}/{tot:<4} ({acc/max(tot,1)*100:.1f}%)")
    print(f"  {'합계':<15} {total_pl:>4}")

    print("\n[탐지 수 비교 (conf=0.10, raw+proc 합산)]")
    print(f"  {'데이터셋':<15} {'Original':>10} {'Fine-tuned':>12}")
    print("  " + "-" * 40)
    for src_name in compare_results:
        orig = compare_results[src_name]['original'].get(0.10, 0)
        ft   = compare_results[src_name]['finetune'].get(0.10, 0)
        delta = ft - orig
        sign = '+' if delta >= 0 else ''
        print(f"  {src_name:<15} {orig:>10} {ft:>12}  ({sign}{delta})")

    print("\n해석:")
    print("  - pipe/rebar 데이터: fine-tuned 탐지 증가 → 개선")
    print("  - tunnel 데이터: 탐지 없거나 적을수록 좋음 (FP 억제)")

    # 시각화
    visualize_results(pl_results, compare_results, samples_by_source, ft_weights)

    print(f"\n{'='*65}")
    print("Phase C 완료")
    print(f"  Fine-tuned weights: {ft_weights}")
    print(f"  시각화: {OUTPUT_DIR/'pseudolabel_finetune.png'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
