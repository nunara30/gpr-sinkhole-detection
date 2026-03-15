"""
Phase N: YOLO + EfficientNet 앙상블 파이프라인

전략
----
- YOLO (phase_g/run_n20)  : sinkhole / tunnel  → bbox 탐지
- EfficientNet (phase_k)  : pipe / rebar        → ZON 레벨 분류
  * bbox 위치가 ZON마다 달라 YOLO로는 일반화 불가 (Phase J 확인)
  * EfficientNet는 "있다/없다"만 판단 → 이미지 전체 폭 강조 바로 표시

출력
----
  {
    "detections": [                          # YOLO bbox (sinkhole/tunnel)
      {"cls": "tunnel", "conf": 0.87,
       "x1":10, "y1":20, "x2":200, "y2":180},
      ...
    ],
    "classifications": [                     # EfficientNet strip (pipe/rebar)
      {"cls": "pipe",  "conf": 0.91},
      ...
    ],
    "vis_bgr": np.ndarray                    # 시각화 이미지 (BGR, 640×640)
  }

사용법
------
  from phase_n_ensemble import EnsemblePipeline
  pipe = EnsemblePipeline()
  result = pipe.run(dt_path)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec,
)

# ── 경로 ──────────────────────────────────────────────────────────────
YOLO_PT  = BASE_DIR / "models/yolo_runs/phase_g/run_n20/weights/best.pt"
CLS_PT   = BASE_DIR / "models/phase_k_cls.pt"

# ── 상수 ──────────────────────────────────────────────────────────────
YOLO_CLASSES  = ["sinkhole", "pipe", "rebar", "tunnel"]   # 학습 시 순서
YOLO_KEEP     = {"sinkhole"}                              # bbox 신뢰 클래스 (tunnel은 EfficientNet이 더 정확)
CLS_CLASSES   = ["pipe", "rebar", "tunnel"]               # EfficientNet 순서
CLS_KEEP      = {"pipe", "rebar", "tunnel"}               # 분류 담당 클래스
IMG_SIZE      = 640
CLS_SIZE      = 224
DT_SEC        = (8.0 / 512) * 1e-9

CLASS_COLORS  = {
    "sinkhole": (0,   0,   255),
    "tunnel":   (0,   180, 255),
    "pipe":     (255, 100,   0),
    "rebar":    (0,   200,   0),
}

_cls_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((CLS_SIZE, CLS_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── 전처리 ────────────────────────────────────────────────────────────

def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    """IDS .dt → 640×640 BGR uint8. 실패 시 None."""
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
        return cv2.cvtColor(cv2.resize(gray, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_GRAY2BGR)
    except Exception:
        return None


# ── EfficientNet 로더 ─────────────────────────────────────────────────

def _load_cls_model(device: torch.device) -> nn.Module:
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLS_CLASSES))
    state = torch.load(CLS_PT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


# ── 앙상블 파이프라인 ─────────────────────────────────────────────────

class EnsemblePipeline:
    """
    YOLO + EfficientNet 앙상블.
    인스턴스 생성 시 두 모델을 한 번만 로드합니다.
    """

    def __init__(self, yolo_conf: float = 0.25, cls_conf: float = 0.60):
        self.yolo_conf = yolo_conf
        self.cls_conf  = cls_conf
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[EnsemblePipeline] 디바이스: {self.device}")
        print(f"[EnsemblePipeline] YOLO 로딩: {YOLO_PT.name}")
        self.yolo = YOLO(str(YOLO_PT))

        print(f"[EnsemblePipeline] EfficientNet 로딩: {CLS_PT.name}")
        self.cls_model = _load_cls_model(self.device)
        print("[EnsemblePipeline] 준비 완료.")

    # ── 추론 ──────────────────────────────────────────────────────────

    def _run_yolo(self, img_bgr: np.ndarray) -> list[dict]:
        """YOLO → sinkhole/tunnel bbox만 반환."""
        preds = self.yolo.predict(img_bgr, conf=self.yolo_conf,
                                  imgsz=IMG_SIZE, verbose=False)
        dets = []
        if not preds or preds[0].boxes is None:
            return dets
        for xyxy, cls_id, conf in zip(
            preds[0].boxes.xyxy.cpu().numpy(),
            preds[0].boxes.cls.cpu().numpy().astype(int),
            preds[0].boxes.conf.cpu().numpy(),
        ):
            name = YOLO_CLASSES[cls_id] if cls_id < len(YOLO_CLASSES) else str(cls_id)
            if name not in YOLO_KEEP:
                continue
            dets.append({
                "cls": name, "conf": float(conf),
                "x1": int(xyxy[0]), "y1": int(xyxy[1]),
                "x2": int(xyxy[2]), "y2": int(xyxy[3]),
            })
        return dets

    def _run_cls(self, img_bgr: np.ndarray) -> list[dict]:
        """EfficientNet → pipe/rebar 분류 결과만 반환."""
        tensor = _cls_transform(img_bgr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cls_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        results = []
        for i, cls_name in enumerate(CLS_CLASSES):
            if cls_name not in CLS_KEEP:
                continue
            if probs[i] >= self.cls_conf:
                results.append({"cls": cls_name, "conf": float(probs[i])})
        return results

    # ── 시각화 ────────────────────────────────────────────────────────

    def _draw(self, img_bgr: np.ndarray,
              detections: list[dict],
              classifications: list[dict]) -> np.ndarray:
        vis = img_bgr.copy()
        h, w = vis.shape[:2]

        # EfficientNet 결과: 이미지 왼쪽에 세로 컬러 바 + 텍스트
        bar_x = 5
        bar_w = 12
        for i, item in enumerate(classifications):
            color = CLASS_COLORS.get(item["cls"], (200, 200, 200))
            y_start = i * (bar_w + 4) + 5
            y_end   = y_start + bar_w
            # 반투명 세로 바 (이미지 전체 높이)
            overlay = vis.copy()
            cv2.rectangle(overlay, (bar_x, 0), (bar_x + bar_w, h), color, -1)
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
            # 라벨
            label = f"{item['cls']} {item['conf']:.2f}"
            cv2.putText(vis, label,
                        (bar_x + bar_w + 4, y_start + bar_w - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # YOLO 결과: bbox
        for det in detections:
            color = CLASS_COLORS.get(det["cls"], (200, 200, 200))
            cv2.rectangle(vis, (det["x1"], det["y1"]),
                          (det["x2"], det["y2"]), color, 2)
            label = f"{det['cls']} {det['conf']:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(det["y1"] - 4, lh + 2)
            cv2.rectangle(vis, (det["x1"], ty - lh - 2),
                          (det["x1"] + lw + 2, ty + 2), color, -1)
            cv2.putText(vis, label, (det["x1"] + 1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 아무것도 없으면 "미탐지" 표시
        if not detections and not classifications:
            cv2.putText(vis, "No detection", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return vis

    # ── 메인 추론 ─────────────────────────────────────────────────────

    def run(self, dt_path: Path | str,
            img_bgr: np.ndarray | None = None) -> dict:
        """
        .dt 파일 또는 이미 전처리된 BGR 이미지로 앙상블 추론.

        Returns
        -------
        {
          "detections":     list[dict],  # YOLO (sinkhole/tunnel)
          "classifications": list[dict], # EfficientNet (pipe/rebar)
          "vis_bgr":        np.ndarray,  # 시각화 이미지
          "error":          str | None,
        }
        """
        if img_bgr is None:
            img_bgr = preprocess_dt(Path(dt_path))
            if img_bgr is None:
                return {"detections": [], "classifications": [],
                        "vis_bgr": None, "error": "전처리 실패"}

        detections     = self._run_yolo(img_bgr)
        classifications = self._run_cls(img_bgr)
        vis            = self._draw(img_bgr, detections, classifications)

        return {
            "detections":      detections,
            "classifications": classifications,
            "vis_bgr":         vis,
            "error":           None,
        }


# ── 단독 실행 테스트 ──────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    DATA_DIR = BASE_DIR / "data/gpr/guangzhou/Data Set"
    pipeline = EnsemblePipeline()

    # 각 클래스에서 샘플 1개씩 테스트
    for cls in ["pipe", "rebar", "tunnel"]:
        cls_dir = DATA_DIR / cls
        dt_files = sorted(cls_dir.rglob("*.dt"))[:1] if cls_dir.exists() else []
        if not dt_files:
            print(f"[{cls}] .dt 파일 없음")
            continue

        result = pipeline.run(dt_files[0])
        if result["error"]:
            print(f"[{cls}] 오류: {result['error']}")
            continue

        print(f"[{cls}] YOLO: {result['detections']}")
        print(f"[{cls}] EfficientNet: {result['classifications']}")

        out_path = BASE_DIR / f"src/output/phase_n_{cls}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result["vis_bgr"])
        print(f"[{cls}] 저장: {out_path}")
