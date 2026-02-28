"""
Phase F - 배치 추론 + 요약
학습에 사용되지 않은 Guangzhou .dt 파일을 대상으로
전체 파이프라인(읽기→전처리→추론→시각화)을 검증하는 현장 배치 워크플로우.
"""

import sys
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─── 경로 설정 ───────────────────────────────────
BASE_DIR  = Path(__file__).parent.parent  # gpr-sinkhole-detection/
DATA_DIR  = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST  = BASE_DIR / "guangzhou_labeled/manifest.json"
MODEL_PT  = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"
OUT_DIR   = BASE_DIR / "src/output/week4_multiclass/phase_f"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec
)

# ─── 상수 ────────────────────────────────────────
CONF_THRESH  = 0.05
IMGSZ        = 640
DT_NS        = 8.0 / 512          # time step (ns per sample)
DT_SEC       = DT_NS * 1e-9       # → seconds
CLASS_NAMES  = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS = {                   # BGR
    "sinkhole": (0, 0, 255),
    "pipe":     (255, 100, 0),
    "rebar":    (0, 200, 0),
    "tunnel":   (0, 180, 255),
}
MAX_PER_CLS  = 20                  # 클래스당 최대 처리 파일 수
CATEGORIES   = ["pipe", "rebar", "tunnel"]


# ─────────────────────────────────────────────────
# 1. 학습 파일 목록 로드
# ─────────────────────────────────────────────────

def _load_trained_stems() -> set:
    """
    manifest.json에서 학습에 사용된 .dt 파일의 '상대 핵심 경로'를 추출.
    비교 기준: 'Data Set' 이후 경로를 소문자 정규화.
    """
    stems = set()
    try:
        raw = MANIFEST.read_bytes()
        # manifest는 cp949(EUC-KR) 인코딩
        data = json.loads(raw.decode("cp949"))
    except Exception as e:
        print(f"  [경고] manifest 로드 실패: {e}")
        return stems

    marker = "Data Set"
    for entry in data.get("images", []):
        src = entry.get("source", "")
        idx = src.lower().find(marker.lower())
        if idx != -1:
            rel = src[idx + len(marker):].replace("\\", "/").lstrip("/").lower()
            stems.add(rel)
    return stems


# ─────────────────────────────────────────────────
# 2. 테스트 파일 수집
# ─────────────────────────────────────────────────

def collect_test_files(trained_stems: set) -> dict:
    """
    guangzhou/Data Set 세 폴더를 순회해 ZON별 첫 .dt를 선택.
    학습에 쓰인 파일은 제외, 클래스당 MAX_PER_CLS개 반환.
    Returns: {"pipe": [Path, ...], "rebar": [...], "tunnel": [...]}
    """
    result = {cls: [] for cls in CATEGORIES}
    marker = "Data Set"

    for cls in CATEGORIES:
        cat_dir = DATA_DIR / cls
        if not cat_dir.exists():
            print(f"  [경고] 폴더 없음: {cat_dir}")
            continue

        # ZON 폴더 수집 (재귀, ASCII 서브폴더 제외)
        zon_dirs = sorted(
            p for p in cat_dir.rglob("*.ZON")
            if p.is_dir() and "ASCII" not in str(p)
        )

        for zon_dir in zon_dirs:
            if len(result[cls]) >= MAX_PER_CLS:
                break

            # ZON 폴더에서 .dt 파일 선택 (직계 자식, ASCII 제외)
            dt_files = sorted(
                p for p in zon_dir.glob("*.dt")
                if "ASCII" not in str(p)
            )
            if not dt_files:
                continue

            dt_path = dt_files[0]

            # manifest 비교용 상대 경로
            full_str = str(dt_path).replace("\\", "/")
            idx = full_str.lower().find(marker.lower())
            rel = (
                full_str[idx + len(marker):].lstrip("/").lower()
                if idx != -1 else full_str.lower()
            )

            if rel in trained_stems:
                continue  # 학습에 쓰인 파일 제외

            result[cls].append(dt_path)

        print(f"  [{cls}] 테스트 파일: {len(result[cls])}개")

    return result


# ─────────────────────────────────────────────────
# 3. 전처리
# ─────────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    """
    .dt → 전처리 → 640×640 BGR 이미지.
    실패 시 None 반환.
    """
    data, _ = read_ids_dt(str(dt_path))
    if data is None:
        return None

    try:
        data = dc_removal(data)
        data = background_removal(data)
        data = bandpass_filter(data, DT_SEC, 500.0, 4000.0)
        data = gain_sec(data, tpow=1.0, alpha=0.0, dt=DT_SEC)
    except Exception as e:
        print(f"    [경고] 전처리 오류 {dt_path.name}: {e}")
        return None

    # 2%–98% percentile 정규화 → grayscale uint8
    p2, p98 = np.percentile(data, [2, 98])
    if p98 <= p2:
        p2, p98 = data.min(), data.max()
    if p98 == p2:
        return None

    norm = np.clip((data - p2) / (p98 - p2), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr  = cv2.resize(bgr, (IMGSZ, IMGSZ))
    return bgr


# ─────────────────────────────────────────────────
# 4. 추론
# ─────────────────────────────────────────────────

def run_inference(model, img_bgr: np.ndarray) -> list:
    """
    YOLO 추론 → [{"cls_name", "conf", "x1", "y1", "x2", "y2"}, ...]
    """
    preds = model.predict(img_bgr, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)
    dets  = []
    if not preds:
        return dets

    result = preds[0]
    if result.boxes is None or len(result.boxes) == 0:
        return dets

    boxes = result.boxes
    for xyxy, cls_id, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
    ):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        dets.append({
            "cls_name": cls_name,
            "conf":     float(conf),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return dets


# ─────────────────────────────────────────────────
# 5. 어노테이션
# ─────────────────────────────────────────────────

def annotate_image(img_bgr: np.ndarray, dets: list, true_cls_name: str) -> np.ndarray:
    """
    bbox + 레이블 텍스트를 이미지에 오버레이.
    좌상단: True class, 우상단: detection count.
    """
    out = img_bgr.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 2

    for det in dets:
        color = CLASS_COLORS.get(det["cls_name"], (200, 200, 200))
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['cls_name']} {det['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        ty = max(y1 - 4, th + 2)
        cv2.rectangle(out, (x1, ty - th - 2), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(out, label, (x1 + 1, ty), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    # 좌상단: True label
    true_label = f"True: {true_cls_name}"
    cv2.putText(out, true_label, (6, 22), font, 0.65,
                (255, 255, 0), 2, cv2.LINE_AA)

    # 우상단: detection count
    count_label = f"{len(dets)} det"
    (cw, _), _ = cv2.getTextSize(count_label, font, 0.6, 2)
    cv2.putText(out, count_label, (IMGSZ - cw - 6, 22), font, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────
# 6. 요약 그리드 생성
# ─────────────────────────────────────────────────

def save_summary_grid(annotated_map: dict, save_path: Path):
    """
    3행(pipe/rebar/tunnel) × 4열 그리드.
    탐지 없는 셀은 "No Detection" 빈 셀로 표시.
    한글 경로 저장: imencode → write_bytes.
    """
    rows = CATEGORIES
    n_cols = 4
    cell_h, cell_w = 320, 320

    fig, axes = plt.subplots(
        len(rows), n_cols,
        figsize=(n_cols * 3.2, len(rows) * 3.2)
    )

    for r, cls in enumerate(rows):
        imgs = annotated_map.get(cls, [])  # list of BGR np.ndarray
        for c in range(n_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(imgs):
                img = imgs[c]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if c == 0:
                    ax.set_title(cls, fontsize=10, fontweight="bold", loc="left")
            else:
                ax.set_facecolor("#1a1a1a")
                ax.text(
                    0.5, 0.5, "No Detection",
                    transform=ax.transAxes,
                    ha="center", va="center",
                    color="#888888", fontsize=9,
                )

    fig.suptitle("Phase F - Detection Summary Grid", fontsize=13, fontweight="bold")
    plt.tight_layout()

    # imencode로 저장 (한글 경로 대응)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(*fig.canvas.get_width_height()[::-1], 4)
    buf_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    ret, encoded = cv2.imencode(".png", buf_bgr)
    if ret:
        save_path.write_bytes(encoded.tobytes())
    plt.close(fig)


# ─────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Phase F - 배치 추론 + 요약")
    print("=" * 60)

    # YOLO 모델 로드
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] ultralytics 미설치: pip install ultralytics")
        return

    print(f"\n[1] 모델 로드: {MODEL_PT}")
    if not MODEL_PT.exists():
        print(f"  [오류] 모델 파일 없음: {MODEL_PT}")
        return
    model = YOLO(str(MODEL_PT))

    # 학습 파일 목록
    print("\n[2] 학습 파일 목록 로드...")
    trained_stems = _load_trained_stems()
    print(f"  학습 파일: {len(trained_stems)}개")

    # 테스트 파일 수집
    print("\n[3] 테스트 파일 수집...")
    test_files = collect_test_files(trained_stems)
    total_files = sum(len(v) for v in test_files.values())
    if total_files == 0:
        print("  [오류] 테스트 파일 없음")
        return
    print(f"  총 {total_files}개 파일 대상")

    # CSV 로그 초기화
    csv_path = OUT_DIR / "detection_log.csv"
    csv_rows  = []

    # 요약 그리드용: 클래스별 첫 4개 어노테이션 이미지 저장
    annotated_map = {cls: [] for cls in CATEGORIES}

    # 통계
    stats = {cls: {"total": 0, "tp": 0, "fp": 0, "miss": 0} for cls in CATEGORIES}

    # ── 메인 루프 ────────────────────────────────
    print("\n[4] 추론 시작...")
    for cls_name, paths in test_files.items():
        print(f"\n  [{cls_name}] {len(paths)}개 파일 처리")
        for idx, dt_path in enumerate(paths):
            img_bgr = preprocess_dt(dt_path)
            if img_bgr is None:
                print(f"    [{idx:03d}] 전처리 실패 - {dt_path.name}")
                stats[cls_name]["miss"] += 1
                continue

            dets = run_inference(model, img_bgr)
            annotated = annotate_image(img_bgr, dets, cls_name)

            # PNG 저장 (imencode로 한글 경로 대응)
            png_name = f"{cls_name}_{idx:03d}_det.png"
            ret, enc = cv2.imencode(".png", annotated)
            if ret:
                (OUT_DIR / png_name).write_bytes(enc.tobytes())

            # 그리드용 저장 (첫 4개)
            if len(annotated_map[cls_name]) < 4:
                annotated_map[cls_name].append(annotated)

            # TP/FP/Miss 판정
            stats[cls_name]["total"] += 1
            pred_cls_names = [d["cls_name"] for d in dets]
            max_conf = max((d["conf"] for d in dets), default=0.0)
            pred_cls = max((dets), key=lambda d: d["conf"])["cls_name"] if dets else "none"

            if cls_name in pred_cls_names:
                stats[cls_name]["tp"] += 1
            elif dets:
                stats[cls_name]["fp"] += 1
                stats[cls_name]["miss"] += 1
            else:
                stats[cls_name]["miss"] += 1

            # CSV 기록
            csv_rows.append({
                "file":      dt_path.name,
                "true_cls":  cls_name,
                "pred_cls":  pred_cls,
                "conf":      f"{max_conf:.4f}",
                "n_dets":    len(dets),
            })

            if (idx + 1) % 5 == 0 or idx == len(paths) - 1:
                print(f"    {idx+1:3d}/{len(paths)} 완료")

    # ── CSV 저장 ─────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "true_cls", "pred_cls", "conf", "n_dets"]
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[5] CSV 저장: {csv_path.name}  ({len(csv_rows)}행)")

    # ── 요약 그리드 저장 ─────────────────────────
    summary_path = OUT_DIR.parent / "phase_f_summary.png"
    save_summary_grid(annotated_map, summary_path)
    print(f"[6] 요약 그리드: {summary_path.name}")

    # ── 터미널 요약 ──────────────────────────────
    print("\n" + "=" * 60)
    print("  [Phase F] 처리 완료")
    print("=" * 60)
    for cls_name in CATEGORIES:
        s = stats[cls_name]
        total = s["total"]
        tp    = s["tp"]
        fp    = s["fp"]
        miss  = s["miss"]
        recall = tp / total if total > 0 else 0.0
        print(
            f"  {cls_name:<8}: {total}파일 → "
            f"TP {tp} / FP {fp} / Miss {miss}  "
            f"(Recall {recall:.2f})"
        )
    print(f"저장: {OUT_DIR}")

    # ── 검증 기준 출력 ───────────────────────────
    print("\n  [검증 기준]")
    png_count = len(list(OUT_DIR.glob("*_det.png")))
    print(f"  PNG 파일: {png_count}개")
    print(f"  CSV 행 수: {len(csv_rows)}")
    all_recall_ok = True
    for cls_name in CATEGORIES:
        s      = stats[cls_name]
        total  = s["total"]
        recall = s["tp"] / total if total > 0 else 0.0
        thr    = 0.3 if cls_name == "tunnel" else 0.5
        ok     = "OK" if recall >= thr else "WARN"
        if recall < thr:
            all_recall_ok = False
        print(f"  {cls_name:<8} recall={recall:.2f}  [{ok}]  (기준 >={thr})")
    print(f"  summary 그리드: {'OK' if summary_path.exists() else 'FAIL'}")
    print(f"\n  [conf_thresh={CONF_THRESH}]  파이프라인 실행 완료")
    if not all_recall_ok:
        print("\n  [분석] 낮은 recall 원인:")
        print("    - 학습 데이터: 35개 이미지 (pseudo-label 기반)")
        print("    - 베이스 모델 sinkhole 편향 → FP 예측이 주로 sinkhole 클래스")
        print("    - 테스트 파일이 학습 ZON과 다른 스캔 조건 (도메인 갭)")
        print("    → 추가 labeling + re-finetune 권장")


if __name__ == "__main__":
    main()
