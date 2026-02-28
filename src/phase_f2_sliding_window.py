"""
Phase F-2 - Sliding Window + GIF Animation
클래스별 대표 .dt 파일 1개를 트레이스 단위로 슬라이딩하며
bbox 탐지 결과가 업데이트되는 GIF 애니메이션 저장.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST = BASE_DIR / "guangzhou_labeled/manifest.json"
MODEL_PT = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"
OUT_DIR  = BASE_DIR / "src/output/week4_multiclass/phase_f2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec
)

# ─── 상수 ────────────────────────────────────────
CONF_THRESH   = 0.05
IMGSZ         = 640
DT_NS         = 8.0 / 512
DT_SEC        = DT_NS * 1e-9
CLASS_NAMES   = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS  = {
    "sinkhole": (0, 0, 255), "pipe": (255, 100, 0),
    "rebar":    (0, 200, 0), "tunnel": (0, 180, 255),
}
CATEGORIES    = ["pipe", "rebar", "tunnel"]

WINDOW_TRACES = 256   # 슬라이딩 윈도우 너비 (트레이스 수)
STRIDE        = 64    # 스텝 크기
MAX_FRAMES    = 50    # GIF 최대 프레임 수
GIF_DURATION  = 120  # 프레임당 ms


# ─────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────

def _load_trained_stems() -> set:
    stems = set()
    try:
        raw  = MANIFEST.read_bytes()
        data = json.loads(raw.decode("cp949"))
    except Exception:
        return stems
    marker = "Data Set"
    for entry in data.get("images", []):
        src = entry.get("source", "")
        idx = src.lower().find(marker.lower())
        if idx != -1:
            rel = src[idx + len(marker):].replace("\\", "/").lstrip("/").lower()
            stems.add(rel)
    return stems


def pick_representative(cls: str, trained_stems: set) -> Path | None:
    """클래스별 가장 많은 트레이스를 가진 미학습 .dt 파일 선택."""
    cat_dir = DATA_DIR / cls
    if not cat_dir.exists():
        return None

    candidates = []
    marker = "Data Set"
    for dt_path in sorted(cat_dir.rglob("*.dt")):
        if "ASCII" in str(dt_path):
            continue
        full_str = str(dt_path).replace("\\", "/")
        idx = full_str.lower().find(marker.lower())
        rel = (full_str[idx + len(marker):].lstrip("/").lower()
               if idx != -1 else full_str.lower())
        if rel in trained_stems:
            continue
        candidates.append(dt_path)

    if not candidates:
        return None

    # 트레이스가 가장 많은 파일 선택
    best, best_n = None, 0
    for p in candidates[:20]:   # 최대 20개 탐색
        data, _ = read_ids_dt(str(p))
        if data is not None and data.shape[1] > best_n:
            best_n = data.shape[1]
            best = p
    return best


def preprocess_raw(dt_path: Path):
    """전처리된 float32 B-scan (n_samples, n_traces) 반환."""
    data, _ = read_ids_dt(str(dt_path))
    if data is None:
        return None
    try:
        data = dc_removal(data)
        data = background_removal(data)
        data = bandpass_filter(data, DT_SEC, 500.0, 4000.0)
        data = gain_sec(data, tpow=1.0, alpha=0.0, dt=DT_SEC)
    except Exception as e:
        print(f"  [경고] 전처리 오류: {e}")
        return None
    return data


def window_to_bgr(window: np.ndarray) -> np.ndarray:
    """슬라이딩 윈도우 → 640×640 BGR."""
    p2, p98 = np.percentile(window, [2, 98])
    if p98 <= p2:
        p2, p98 = window.min(), window.max()
    if p98 == p2:
        return np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    norm = np.clip((window - p2) / (p98 - p2), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr  = cv2.resize(bgr, (IMGSZ, IMGSZ))
    return bgr


def run_inference(model, img_bgr: np.ndarray) -> list:
    preds = model.predict(img_bgr, conf=CONF_THRESH, imgsz=IMGSZ, verbose=False)
    dets  = []
    if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
        return dets
    boxes = preds[0].boxes
    for xyxy, cls_id, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy().astype(int),
        boxes.conf.cpu().numpy(),
    ):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        dets.append({
            "cls_name": cls_name, "conf": float(conf),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return dets


def make_frame(img_bgr, dets, cls_name, start_trace, n_traces_total,
               frame_idx, total_frames) -> np.ndarray:
    """한 프레임 BGR 이미지 생성."""
    out  = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # bbox
    for det in dets:
        color = CLASS_COLORS.get(det["cls_name"], (200, 200, 200))
        cv2.rectangle(out, (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]), color, 2)
        label = f"{det['cls_name']} {det['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(det["y1"] - 4, th + 2)
        cv2.rectangle(out, (det["x1"], ty - th - 2),
                      (det["x1"] + tw + 2, ty + 2), color, -1)
        cv2.putText(out, label, (det["x1"]+1, ty), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    # 좌상단: True class + 트레이스 범위
    end_trace = min(start_trace + WINDOW_TRACES, n_traces_total)
    info = f"True: {cls_name}  [trace {start_trace}~{end_trace}/{n_traces_total}]"
    cv2.putText(out, info, (6, 20), font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    # 우상단: det count
    cnt = f"{len(dets)} det"
    (cw, _), _ = cv2.getTextSize(cnt, font, 0.5, 1)
    cv2.putText(out, cnt, (IMGSZ - cw - 6, 20), font, 0.5,
                (0, 255, 255), 1, cv2.LINE_AA)

    # 하단: 진행 바
    bar_y = IMGSZ - 12
    bar_w = int((frame_idx + 1) / total_frames * IMGSZ)
    cv2.rectangle(out, (0, bar_y), (IMGSZ, IMGSZ), (40, 40, 40), -1)
    cv2.rectangle(out, (0, bar_y), (bar_w, IMGSZ), (0, 200, 100), -1)

    return out


def make_gif(model, cls_name: str, dt_path: Path) -> Path | None:
    print(f"  [{cls_name}] 파일: {dt_path.name}")
    data = preprocess_raw(dt_path)
    if data is None:
        return None

    n_samples, n_traces = data.shape
    print(f"    shape: ({n_samples}, {n_traces})")

    # 윈도우 시작 위치 목록
    if n_traces <= WINDOW_TRACES:
        starts = [0]
    else:
        all_starts = list(range(0, n_traces - WINDOW_TRACES + 1, STRIDE))
        if len(all_starts) > MAX_FRAMES:
            idxs = np.linspace(0, len(all_starts) - 1, MAX_FRAMES, dtype=int)
            starts = [all_starts[i] for i in idxs]
        else:
            starts = all_starts
    print(f"    프레임 수: {len(starts)}")

    frames_pil = []
    for fi, start in enumerate(starts):
        end    = min(start + WINDOW_TRACES, n_traces)
        window = data[:, start:end]
        img    = window_to_bgr(window)
        dets   = run_inference(model, img)
        frame  = make_frame(img, dets, cls_name, start, n_traces, fi, len(starts))
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(rgb))

        if (fi + 1) % 10 == 0 or fi == len(starts) - 1:
            print(f"    {fi+1}/{len(starts)} 프레임 완료")

    gif_path = OUT_DIR / f"{cls_name}_sliding.gif"
    frames_pil[0].save(
        gif_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=GIF_DURATION,
        loop=0,
    )
    print(f"    저장: {gif_path.name}  ({len(frames_pil)} frames)")
    return gif_path


# ─────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Phase F-2 - Sliding Window + GIF Animation")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics")
        return

    if not MODEL_PT.exists():
        print(f"[오류] 모델 없음: {MODEL_PT}")
        return
    model = YOLO(str(MODEL_PT))

    trained_stems = _load_trained_stems()
    print(f"학습 파일 제외: {len(trained_stems)}개\n")

    saved = []
    for cls in CATEGORIES:
        dt_path = pick_representative(cls, trained_stems)
        if dt_path is None:
            print(f"  [{cls}] 대표 파일 없음")
            continue
        gif_path = make_gif(model, cls, dt_path)
        if gif_path:
            saved.append(gif_path)

    print(f"\n완료: GIF {len(saved)}개 저장 → {OUT_DIR}")
    for p in saved:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
