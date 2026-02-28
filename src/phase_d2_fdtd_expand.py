"""
Phase D-2: FDTD 데이터 확장 + Fine-tuning

Phase B의 6개 FDTD 시나리오를 26개로 확장 (20개 신규 추가):
  - Sinkhole: +8개 (freq/depth/radius/soil 다양화)
  - Pipe:     +8개 (freq/depth/diam/soil 다양화)
  - Rebar:    +4개 (freq/depth/spacing/n 다양화)

데이터 증강:
  - 수평 플립 (horizontal flip + bbox 반전) → 2× 데이터
  - 총 26 × 2 = 52개 FDTD B-scan

학습:
  - FDTD 52개 + 해석적 합성 200개 혼합 학습
  - Phase D-1 (Mendeley)과 성능 비교
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path

import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
os.environ['PATH'] = r'C:\Users\jbcgl\miniconda3\envs\gpr_rag\Library\bin;' + os.environ['PATH']
sys.path.insert(0, str(Path(__file__).parent.parent / "gprMax"))

# 경로 임포트 (phase_b_fdtd.py의 함수 재사용)
from phase_b_fdtd import (
    make_sinkhole_in, make_pipe_in, make_rebar_in,
    run_gprmax, hdf5_to_bscan, compute_fdtd_bbox,
    bscan_to_png, save_yolo_label,
    FDTD_IN_DIR, C0
)

# ── 경로 ──
BASE_DIR    = Path("G:/RAG_system")
DATA_DIR    = BASE_DIR / "data" / "gpr"
EXPAND_IN   = BASE_DIR / "models" / "fdtd_expand"    # 신규 .in + .out
EXPAND_OUT  = DATA_DIR / "fdtd_expand"                # 신규 PNG + label
YOLO_FDTD   = DATA_DIR / "yolo_fdtd"                  # 전체 FDTD 데이터셋
SYNTH_DIR   = DATA_DIR / "yolo_multiclass"             # 기존 해석적 합성
FT_DIR      = BASE_DIR / "models" / "yolo_runs" / "finetune_fdtd"
MC_WEIGHTS  = BASE_DIR / "models" / "yolo_runs" / "multiclass_detect" / "weights" / "best.pt"
OUTPUT_DIR  = BASE_DIR / "src" / "output" / "week4_multiclass"

for d in [EXPAND_IN, EXPAND_OUT,
          YOLO_FDTD / "images/train", YOLO_FDTD / "images/val",
          YOLO_FDTD / "labels/train", YOLO_FDTD / "labels/val",
          OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 신규 FDTD 시나리오 정의 (20개)
# ─────────────────────────────────────────────

NEW_SCENARIOS = [
    # Sinkhole 8개: Phase B(900MHz)에서 freq/depth/radius/soil 다양화
    ('sinkhole', dict(freq_hz=400e6,  depth_m=0.5,  radius_m=0.20, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=400e6,  depth_m=0.8,  radius_m=0.30, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=400e6,  depth_m=0.5,  radius_m=0.20, soil_epsr=12)),
    ('sinkhole', dict(freq_hz=400e6,  depth_m=0.8,  radius_m=0.30, soil_epsr=12)),
    ('sinkhole', dict(freq_hz=900e6,  depth_m=1.0,  radius_m=0.25, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=900e6,  depth_m=1.2,  radius_m=0.35, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=2000e6, depth_m=0.3,  radius_m=0.15, soil_epsr=6)),
    ('sinkhole', dict(freq_hz=2000e6, depth_m=0.5,  radius_m=0.20, soil_epsr=6)),

    # Pipe 8개: freq/depth/diam/soil 다양화
    ('pipe', dict(freq_hz=400e6,  depth_m=0.3, pipe_diam=0.10, soil_epsr=6)),
    ('pipe', dict(freq_hz=400e6,  depth_m=0.5, pipe_diam=0.15, soil_epsr=6)),
    ('pipe', dict(freq_hz=400e6,  depth_m=0.8, pipe_diam=0.10, soil_epsr=12)),
    ('pipe', dict(freq_hz=400e6,  depth_m=1.0, pipe_diam=0.20, soil_epsr=12)),
    ('pipe', dict(freq_hz=900e6,  depth_m=0.3, pipe_diam=0.05, soil_epsr=6)),
    ('pipe', dict(freq_hz=900e6,  depth_m=1.0, pipe_diam=0.10, soil_epsr=6)),
    ('pipe', dict(freq_hz=2000e6, depth_m=0.2, pipe_diam=0.05, soil_epsr=6)),
    ('pipe', dict(freq_hz=2000e6, depth_m=0.4, pipe_diam=0.10, soil_epsr=6)),

    # Rebar 4개: freq/depth/spacing/n_rebars 다양화
    ('rebar', dict(freq_hz=900e6,  depth_m=0.15, spacing_m=0.10, n_rebars=3, soil_epsr=12)),
    ('rebar', dict(freq_hz=900e6,  depth_m=0.20, spacing_m=0.20, n_rebars=7, soil_epsr=6)),
    ('rebar', dict(freq_hz=2000e6, depth_m=0.10, spacing_m=0.10, n_rebars=5, soil_epsr=6)),
    ('rebar', dict(freq_hz=2000e6, depth_m=0.15, spacing_m=0.15, n_rebars=3, soil_epsr=6)),
]


def scenario_stem(stype: str, params: dict) -> str:
    """시나리오 고유 파일명 생성 (간결)"""
    f_mhz = int(params['freq_hz'] / 1e6)
    d_cm  = int(params['depth_m'] * 100)
    er    = int(params.get('soil_epsr', 6))
    if stype == 'sinkhole':
        r_cm = int(params['radius_m'] * 100)
        return f"fdtdexp_sk_f{f_mhz}_d{d_cm}_r{r_cm}_er{er}"
    elif stype == 'pipe':
        D_cm = int(params['pipe_diam'] * 100)
        return f"fdtdexp_pi_f{f_mhz}_d{d_cm}_D{D_cm}_er{er}"
    elif stype == 'rebar':
        sp_cm = int(params['spacing_m'] * 100)
        n     = params['n_rebars']
        return f"fdtdexp_rb_f{f_mhz}_d{d_cm}_s{sp_cm}_n{n}_er{er}"
    else:
        return f"fdtdexp_{stype}_f{f_mhz}_d{d_cm}"


# ─────────────────────────────────────────────
# 1. 신규 FDTD 시나리오 실행
# ─────────────────────────────────────────────

def run_new_scenarios() -> list:
    """20개 신규 FDTD 시나리오 생성 및 실행"""
    results = []
    total = len(NEW_SCENARIOS)

    for idx, (stype, params) in enumerate(NEW_SCENARIOS, 1):
        stem = scenario_stem(stype, params)
        in_path = EXPAND_IN / f"{stem}.in"

        print(f"\n[{idx}/{total}] {stem}")

        # .in 파일 생성
        if stype == 'sinkhole':
            meta = make_sinkhole_in(**params, out_path=in_path)
        elif stype == 'pipe':
            meta = make_pipe_in(**params, out_path=in_path)
        elif stype == 'rebar':
            meta = make_rebar_in(**params, out_path=in_path)
        else:
            continue

        n_traces = meta['n_traces']
        print(f"  n_traces={n_traces}, domain={meta['domain_x']:.1f}×{meta['domain_y']:.2f}m")

        # gprMax 실행
        in_stem = EXPAND_IN / stem
        try:
            run_gprmax(in_path, n_traces)
        except Exception as e:
            print(f"  [오류] {e}")
            continue

        # .out 파일 확인
        first_out = EXPAND_IN / f"{stem}1.out"
        if not first_out.exists():
            print(f"  [오류] .out 없음")
            continue

        # B-scan 생성
        bscan = hdf5_to_bscan(in_stem, n_traces)
        print(f"  B-scan: {bscan.shape}")

        # YOLO bbox
        bboxes = compute_fdtd_bbox(meta, bscan)
        print(f"  bbox: {bboxes}")

        # PNG + label 저장
        png_path = EXPAND_OUT / f"{stem}.png"
        label_path = EXPAND_OUT / f"{stem}.txt"
        bscan_to_png(bscan, png_path)
        save_yolo_label(label_path, bboxes)

        # 메타 저장
        (EXPAND_OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))

        results.append({'stem': stem, 'meta': meta, 'bscan': bscan, 'bboxes': bboxes})

    return results


# ─────────────────────────────────────────────
# 3. 수평 플립 증강 (헬퍼)
# ─────────────────────────────────────────────

def _flip_and_save(src_png: Path, src_txt: Path, dst_png: Path, dst_txt: Path):
    """
    B-scan 수평 플립 + bbox cx 반전 저장
    물리적 타당성: GPR 스캔 방향 반전 = 좌우 대칭
    """
    img = cv2.imread(str(src_png))
    if img is None:
        return False
    cv2.imwrite(str(dst_png), cv2.flip(img, 1))

    lines = src_txt.read_text().strip().splitlines() if src_txt.exists() else []
    flipped = []
    for l in lines:
        parts = l.strip().split()
        if len(parts) == 5:
            flipped.append(f"{parts[0]} {1.0 - float(parts[1]):.6f} {parts[2]} {parts[3]} {parts[4]}")
    dst_txt.write_text("\n".join(flipped))
    return True


# ─────────────────────────────────────────────
# 4. YOLO 데이터셋 구성
# ─────────────────────────────────────────────

def build_fdtd_dataset(new_results: list, phase_b_results: list,
                       n_synth_train: int = 200, n_synth_val: int = 50,
                       split_ratio: float = 0.85,
                       seed: int = 42) -> tuple:
    """
    FDTD (신규 + Phase B) + 플립 증강 + 해석적 합성 혼합
    """
    import random
    rng = random.Random(seed)

    # 기존 YOLO 데이터셋 초기화
    for sub in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        d = YOLO_FDTD / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n_train = 0
    n_val = 0
    counter = [0]

    def add_item(src_png, src_txt, split):
        """원본 + 플립 2개를 split에 추가"""
        idx = counter[0]; counter[0] += 2
        # 원본
        dst_stem_orig = f"fdtd_{idx:04d}"
        shutil.copy2(src_png, YOLO_FDTD / "images" / split / f"{dst_stem_orig}.png")
        if src_txt and src_txt.exists():
            shutil.copy2(src_txt, YOLO_FDTD / "labels" / split / f"{dst_stem_orig}.txt")
        else:
            (YOLO_FDTD / "labels" / split / f"{dst_stem_orig}.txt").write_text("")
        # 플립
        dst_stem_flip = f"fdtd_{idx+1:04d}_flip"
        src_txt_safe = src_txt if (src_txt and src_txt.exists()) else Path("/dev/null")
        _flip_and_save(
            src_png, src_txt if (src_txt and src_txt.exists()) else Path(str(src_png)),
            YOLO_FDTD / "images" / split / f"{dst_stem_flip}.png",
            YOLO_FDTD / "labels" / split / f"{dst_stem_flip}.txt",
        )

    # ─ Phase B 기존 6개 ─
    b_out = DATA_DIR / "fdtd_bscan"
    for png in sorted(b_out.glob("*.png")):
        txt = png.with_suffix(".txt")
        split = "train" if rng.random() < split_ratio else "val"
        add_item(png, txt, split)
        if split == "train":
            n_train += 2
        else:
            n_val += 2

    # ─ 신규 FDTD ─
    for res in new_results:
        png = EXPAND_OUT / f"{res['stem']}.png"
        txt = EXPAND_OUT / f"{res['stem']}.txt"
        if not png.exists():
            continue
        split = "train" if rng.random() < split_ratio else "val"
        add_item(png, txt, split)
        if split == "train":
            n_train += 2
        else:
            n_val += 2

    # ─ 해석적 합성 데이터 ─
    synth_train_imgs = sorted((SYNTH_DIR / "images" / "train").glob("*.png"))
    synth_val_imgs   = sorted((SYNTH_DIR / "images" / "val").glob("*.png"))
    rng.shuffle(synth_train_imgs)
    rng.shuffle(synth_val_imgs)
    synth_train_imgs = synth_train_imgs[:n_synth_train]
    synth_val_imgs   = synth_val_imgs[:n_synth_val]

    for img_path in synth_train_imgs:
        lbl = SYNTH_DIR / "labels" / "train" / img_path.with_suffix(".txt").name
        idx = counter[0]; counter[0] += 1
        shutil.copy2(img_path, YOLO_FDTD / "images" / "train" / f"synth_{idx:04d}.png")
        if lbl.exists():
            shutil.copy2(lbl, YOLO_FDTD / "labels" / "train" / f"synth_{idx:04d}.txt")
        else:
            (YOLO_FDTD / "labels" / "train" / f"synth_{idx:04d}.txt").write_text("")
        n_train += 1

    for img_path in synth_val_imgs:
        lbl = SYNTH_DIR / "labels" / "val" / img_path.with_suffix(".txt").name
        idx = counter[0]; counter[0] += 1
        shutil.copy2(img_path, YOLO_FDTD / "images" / "val" / f"synth_{idx:04d}.png")
        if lbl.exists():
            shutil.copy2(lbl, YOLO_FDTD / "labels" / "val" / f"synth_{idx:04d}.txt")
        else:
            (YOLO_FDTD / "labels" / "val" / f"synth_{idx:04d}.txt").write_text("")
        n_val += 1

    print(f"  데이터셋 구성: train={n_train}, val={n_val}")
    return n_train, n_val


def create_fdtd_yaml() -> Path:
    yaml_path = YOLO_FDTD / "dataset.yaml"
    yaml_path.write_text(f"""path: {YOLO_FDTD.as_posix()}
train: images/train
val:   images/val
nc: 3
names: ['sinkhole', 'pipe', 'rebar']
""")
    return yaml_path


# ─────────────────────────────────────────────
# 5. Fine-tuning
# ─────────────────────────────────────────────

def finetune_fdtd(yaml_path: Path, base_weights: Path, epochs: int = 60) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(base_weights))
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        batch=4,
        imgsz=416,
        lr0=1e-4,          # 합성→FDTD 전환 (Phase D-1보다 약간 높게)
        lrf=0.01,
        optimizer='AdamW',
        cos_lr=True,
        freeze=5,
        dropout=0.1,
        patience=25,
        warmup_epochs=3,
        project=str(FT_DIR),
        name="run",
        exist_ok=True,
        verbose=False,
        workers=0,
        cache=False,
        amp=False,
    )
    return FT_DIR / "run" / "weights" / "best.pt"


# ─────────────────────────────────────────────
# 6. 평가 + 시각화
# ─────────────────────────────────────────────

def evaluate_models(orig_w: Path, ft_d1_w: Path, ft_d2_w: Path, conf: float = 0.10):
    """원본/Phase D-1/Phase D-2 모델 3-way 비교"""
    from ultralytics import YOLO
    import tempfile
    from week1_gpr_basics import read_ids_dt
    from week2_preprocessing import dc_removal, background_removal, bandpass_filter, gain_sec

    GZ_PIPE_DIR   = DATA_DIR / "guangzhou" / "Data Set" / "pipe"
    GZ_REBAR_DIR  = DATA_DIR / "guangzhou" / "Data Set" / "rebar"
    GZ_TUNNEL_DIR = DATA_DIR / "guangzhou" / "Data Set" / "tunnel"

    sources = [
        ("GZ_pipe",   GZ_PIPE_DIR,   1),
        ("GZ_rebar",  GZ_REBAR_DIR,  2),
        ("GZ_tunnel", GZ_TUNNEL_DIR, -1),
    ]

    def load_gz(folder, n=5):
        files = sorted(folder.glob("*.dt"))[:n]
        samples = []
        for fp in files:
            try:
                data, dt_ns = read_ids_dt(str(fp))
                if data is None or data.shape[1] < 10:
                    continue
                dt_s = dt_ns * 1e-9
                d = dc_removal(data)
                d = background_removal(d)
                d = bandpass_filter(d, dt_s, 500.0, 4000.0)
                d = gain_sec(d, tpow=1.0, alpha=0.0, dt=dt_s)
                mn, mx = np.percentile(d, [2, 98])
                norm = np.clip((d - mn) / (mx - mn + 1e-8), 0, 1)
                rgb = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                rgb = cv2.resize(rgb, (640, 640))
                samples.append((fp.stem, rgb))
            except Exception:
                continue
        return samples

    models_map = {'original': YOLO(str(orig_w))}
    if ft_d1_w.exists():
        models_map['Phase_D1'] = YOLO(str(ft_d1_w))
    if ft_d2_w.exists():
        models_map['Phase_D2'] = YOLO(str(ft_d2_w))

    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, folder, exp_cls in sources:
            samples = load_gz(folder, n=5)
            src_res = {'expected_cls': exp_cls, 'n_samples': len(samples)}
            for mname, model in models_map.items():
                dets = []
                for stem, rgb in samples:
                    tmp = os.path.join(tmpdir, f"{stem}.png")
                    cv2.imwrite(tmp, rgb)
                    preds = model.predict(tmp, conf=conf, verbose=False)
                    boxes = preds[0].boxes
                    if boxes is not None and len(boxes):
                        for ci, cv_ in zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy()):
                            dets.append({'cls': int(ci), 'conf': float(cv_)})
                src_res[mname] = dets
                print(f"  [{name}][{mname}] {len(dets)} dets")
            results[name] = src_res

    return results, list(models_map.keys())


def visualize_d2(eval_results, model_names, n_train, n_val, n_new_ok):
    """Phase D-2 결과 시각화"""
    fig = plt.figure(figsize=(20, 12), facecolor='#1a1a2e')
    fig.suptitle('Phase D-2: FDTD 확장 Fine-tuning 결과',
                 color='white', fontsize=15, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = ['#3498db', '#e74c3c', '#2ecc71']
    src_names = list(eval_results.keys())

    # ─ 패널 1: 데이터셋 구성 ─
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['Phase B\nFDTD (기존)', 'Phase D-2\nFDTD (신규)', '해석적\n합성']
    sizes = [6 * 2, n_new_ok * 2, n_train - 6 * 2 - n_new_ok * 2]
    sizes = [max(0, s) for s in sizes]
    ax1.bar(labels, sizes, color=colors, alpha=0.85)
    ax1.set_title('학습 데이터 구성\n(원본+플립 포함)', color='white', fontsize=9)
    ax1.set_ylabel('이미지 수', color='white')
    ax1.set_facecolor('#2a2a4a')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_color('#555')
    for i, v in enumerate(sizes):
        ax1.text(i, v + 0.5, str(v), ha='center', color='white', fontsize=9)

    # ─ 패널 2: 모델 비교 (탐지 수) ─
    ax2 = fig.add_subplot(gs[0, 1:])
    x = np.arange(len(src_names))
    w = 0.25
    for mi, mname in enumerate(model_names):
        counts = [len(eval_results[s].get(mname, [])) for s in src_names]
        offset = (mi - len(model_names) / 2 + 0.5) * w
        bars = ax2.bar(x + offset, counts, w, label=mname,
                       color=colors[mi % len(colors)], alpha=0.85)
        for bar, c in zip(bars, counts):
            if c > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         str(c), ha='center', color='white', fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(src_names, color='white')
    ax2.set_ylabel('탐지 수 (conf≥0.10)', color='white')
    ax2.set_title('원본 vs Phase D-1 vs Phase D-2\nGuangzhou 탐지 비교', color='white')
    ax2.legend(facecolor='#2a2a4a', labelcolor='white')
    ax2.set_facecolor('#2a2a4a')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_color('#555')

    # ─ 패널 3: 탐지 클래스 분포 (Phase D-2) ─
    d2_cls = []
    for sdata in eval_results.values():
        d2_cls.extend([d['cls'] for d in sdata.get('Phase_D2', [])])

    ax3 = fig.add_subplot(gs[1, 0])
    CLASS_NAMES = ['sinkhole', 'pipe', 'rebar']
    if d2_cls:
        from collections import Counter
        cc = Counter(d2_cls)
        cls_labels = [CLASS_NAMES[k] for k in sorted(cc)]
        cls_vals = [cc[k] for k in sorted(cc)]
        ax3.bar(cls_labels, cls_vals, color=colors[:len(cls_labels)], alpha=0.85)
        ax3.set_title('Phase D-2 탐지 클래스 분포', color='white', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'FDTD fine-tuning 후\n탐지 0건\n(도메인 갭 지속)',
                 ha='center', va='center', color='#e67e22', fontsize=10,
                 transform=ax3.transAxes)
        ax3.set_title('Phase D-2 탐지 결과', color='white', fontsize=9)
    ax3.set_facecolor('#2a2a4a')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values(): spine.set_color('#555')

    # ─ 패널 4-6: FDTD B-scan 샘플 ─
    sample_types = [('sinkhole', 0), ('pipe', 1), ('rebar', 2)]
    b_out = BASE_DIR / "data" / "gpr" / "fdtd_bscan"
    exp_out = EXPAND_OUT

    for idx, (stype, cls_id) in enumerate(sample_types):
        ax = fig.add_subplot(gs[1, idx])
        # Phase B 또는 신규 중 해당 타입 PNG 찾기
        pngs = sorted(b_out.glob(f"*{stype}*.png")) + sorted(exp_out.glob(f"*{stype}*.png"))
        if pngs:
            img = cv2.imread(str(pngs[0]))
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto', cmap='seismic')
        ax.set_title(f'FDTD: {stype}\n(class {cls_id})', color='white', fontsize=9)
        ax.axis('off')

    save_path = OUTPUT_DIR / "phase_d2_fdtd_expand.png"
    plt.savefig(str(save_path), dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"  시각화: {save_path}")
    return save_path


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("Phase D-2: FDTD 데이터 확장 + Fine-tuning")
    print(f"  신규 시나리오: {len(NEW_SCENARIOS)}개")
    print(f"  Phase B 기존: 6개")
    print(f"  플립 증강: 2× → 총 {(len(NEW_SCENARIOS) + 6) * 2}개 FDTD B-scan\n")

    # ─ 1. 신규 FDTD 시뮬레이션 ─
    print("[1/5] 신규 FDTD 시나리오 실행...")
    t0 = time.time()
    new_results = run_new_scenarios()
    print(f"  완료: {len(new_results)}/{len(NEW_SCENARIOS)}개 성공 "
          f"({time.time() - t0:.0f}초)")

    # ─ 2. 데이터셋 구성 ─
    print("\n[2/5] YOLO 데이터셋 구성 (FDTD + 합성 혼합)...")
    n_train, n_val = build_fdtd_dataset(new_results, [],
                                         n_synth_train=200, n_synth_val=50)
    yaml_path = create_fdtd_yaml()
    print(f"  dataset.yaml: {yaml_path}")
    print(f"  train={n_train}, val={n_val}")

    # ─ 3. Fine-tuning ─
    print("\n[3/5] Fine-tuning (epochs=60)...")
    t0 = time.time()
    ft_weights = finetune_fdtd(yaml_path, MC_WEIGHTS, epochs=60)
    print(f"  완료: {time.time() - t0:.0f}초")
    print(f"  Best weights: {ft_weights}")

    # 메트릭 확인
    results_csv = FT_DIR / "run" / "results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = max(rows, key=lambda r: float(r.get('metrics/mAP50(B)', 0)))
            last = rows[-1]
            print(f"  Best mAP50: {float(best.get('metrics/mAP50(B)', 0)):.4f} "
                  f"(epoch {best.get('epoch', '?')})")
            print(f"  Final mAP50: {float(last.get('metrics/mAP50(B)', 0)):.4f}")

    # ─ 4. 평가 ─
    print("\n[4/5] 실측 Guangzhou 평가 (원본 vs D-1 vs D-2)...")
    orig_weights = MC_WEIGHTS
    d1_weights   = BASE_DIR / "models" / "yolo_runs" / "finetune_real" / "run" / "weights" / "best.pt"
    eval_results, model_names = evaluate_models(orig_weights, d1_weights, ft_weights)

    # ─ 결과 요약 ─
    print(f"\n{'='*65}")
    print("Phase D-2 결과 요약")
    print("="*65)
    print(f"\n[데이터셋] train={n_train}, val={n_val}")
    print(f"  FDTD B-scan: {(len(new_results) + 6) * 2}개 (신규+PhaseB+플립)")
    print(f"  해석적 합성: 250개")

    print("\n[탐지 비교 (conf=0.10, 5샘플)]")
    print(f"  {'소스':<15} " + " ".join(f"{m:>12}" for m in model_names))
    print("  " + "-" * (15 + 14 * len(model_names)))
    for src_name, res in eval_results.items():
        row = f"  {src_name:<15}"
        for mname in model_names:
            row += f" {len(res.get(mname, [])):>12}"
        exp = res['expected_cls']
        row += f"  (기대: {'탐지' if exp >= 0 else '없음'})"
        print(row)

    print("\n해석:")
    print("  - FDTD fine-tuning이 Guangzhou 탐지를 개선했는가?")
    print("  - Phase D-1(Mendeley) vs D-2(FDTD) 어느 쪽이 더 효과적인가?")

    # ─ 5. 시각화 ─
    print("\n[5/5] 시각화...")
    visualize_d2(eval_results, model_names, n_train, n_val, len(new_results))

    print(f"\n{'='*65}")
    print("Phase D-2 완료")
    print(f"  Fine-tuned weights: {ft_weights}")
    print(f"  시각화: {OUTPUT_DIR / 'phase_d2_fdtd_expand.png'}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
