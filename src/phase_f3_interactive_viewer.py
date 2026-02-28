"""
Phase F-3 - Interactive Viewer (tkinter)
.dt 파일 목록을 좌측 패널에 표시하고, 선택 시 전처리 + 추론 결과를 우측에 실시간 표시.
Confidence 슬라이더로 임계값 조절 가능.
실행: python src/phase_f3_interactive_viewer.py
"""

import sys
import json
import threading
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST = BASE_DIR / "guangzhou_labeled/manifest.json"
MODEL_PT = BASE_DIR / "models/yolo_runs/finetune_gz_e2/run/weights/best.pt"

sys.path.insert(0, str(Path(__file__).parent))
from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec
)

# ─── 상수 ────────────────────────────────────────
IMGSZ       = 640
DT_NS       = 8.0 / 512
DT_SEC      = DT_NS * 1e-9
CLASS_NAMES = ["sinkhole", "pipe", "rebar", "tunnel"]
CLASS_COLORS = {
    "sinkhole": (0, 0, 255), "pipe": (255, 100, 0),
    "rebar":    (0, 200, 0), "tunnel": (0, 180, 255),
}
CATEGORIES  = ["pipe", "rebar", "tunnel"]
MAX_PER_CLS = 20
DISPLAY_W   = 600   # 뷰어 이미지 표시 크기
DISPLAY_H   = 600


# ─────────────────────────────────────────────────
# 데이터 유틸리티
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


def collect_files(trained_stems: set) -> list:
    """(cls_name, dt_path) 리스트 반환."""
    result = []
    marker = "Data Set"
    for cls in CATEGORIES:
        cat_dir = DATA_DIR / cls
        if not cat_dir.exists():
            continue
        zon_dirs = sorted(
            p for p in cat_dir.rglob("*.ZON")
            if p.is_dir() and "ASCII" not in str(p)
        )
        count = 0
        for zon_dir in zon_dirs:
            if count >= MAX_PER_CLS:
                break
            dt_files = sorted(
                p for p in zon_dir.glob("*.dt") if "ASCII" not in str(p)
            )
            if not dt_files:
                continue
            dt_path = dt_files[0]
            full_str = str(dt_path).replace("\\", "/")
            idx = full_str.lower().find(marker.lower())
            rel = (full_str[idx + len(marker):].lstrip("/").lower()
                   if idx != -1 else full_str.lower())
            if rel in trained_stems:
                continue
            result.append((cls, dt_path))
            count += 1
    return result


def preprocess_dt(dt_path: Path) -> np.ndarray | None:
    data, _ = read_ids_dt(str(dt_path))
    if data is None:
        return None
    try:
        data = dc_removal(data)
        data = background_removal(data)
        data = bandpass_filter(data, DT_SEC, 500.0, 4000.0)
        data = gain_sec(data, tpow=1.0, alpha=0.0, dt=DT_SEC)
    except Exception:
        return None
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


def run_inference(model, img_bgr, conf_thresh) -> list:
    preds = model.predict(img_bgr, conf=conf_thresh, imgsz=IMGSZ, verbose=False)
    dets  = []
    if not preds or preds[0].boxes is None or len(preds[0].boxes) == 0:
        return dets
    for xyxy, cls_id, conf in zip(
        preds[0].boxes.xyxy.cpu().numpy(),
        preds[0].boxes.cls.cpu().numpy().astype(int),
        preds[0].boxes.conf.cpu().numpy(),
    ):
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        dets.append({
            "cls_name": cls_name, "conf": float(conf),
            "x1": int(xyxy[0]), "y1": int(xyxy[1]),
            "x2": int(xyxy[2]), "y2": int(xyxy[3]),
        })
    return dets


def annotate(img_bgr, dets, true_cls) -> np.ndarray:
    out  = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for det in dets:
        color = CLASS_COLORS.get(det["cls_name"], (200, 200, 200))
        cv2.rectangle(out, (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]), color, 2)
        label = f"{det['cls_name']} {det['conf']*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(det["y1"] - 4, th + 2)
        cv2.rectangle(out, (det["x1"], ty-th-2),
                      (det["x1"]+tw+2, ty+2), color, -1)
        cv2.putText(out, label, (det["x1"]+1, ty), font, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"True: {true_cls}", (6, 22), font, 0.6,
                (255, 255, 0), 2, cv2.LINE_AA)
    cnt = f"{len(dets)} det"
    (cw, _), _ = cv2.getTextSize(cnt, font, 0.6, 2)
    cv2.putText(out, cnt, (IMGSZ - cw - 6, 22), font, 0.6,
                (0, 255, 255), 2, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────

class GPRViewer:
    def __init__(self, root: tk.Tk, file_list: list, model):
        self.root       = root
        self.file_list  = file_list   # [(cls, Path), ...]
        self.model      = model
        self.cur_idx    = 0
        self._cache     = {}          # dt_path → img_bgr
        self._busy      = False

        root.title("GPR Interactive Viewer  — Phase F-3")
        root.resizable(False, False)
        self._build_ui()
        self._load(0)

    # ── UI 구성 ────────────────────────────────
    def _build_ui(self):
        # ── 좌측 패널 (파일 목록) ──
        left = tk.Frame(self.root, width=260, bg="#1e1e1e")
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        tk.Label(left, text="Test Files", bg="#1e1e1e", fg="#cccccc",
                 font=("Consolas", 11, "bold")).pack(pady=(8, 2))

        # 클래스 필터 버튼
        filter_frame = tk.Frame(left, bg="#1e1e1e")
        filter_frame.pack(fill=tk.X, padx=4, pady=2)
        self._filter = tk.StringVar(value="all")
        for label, val in [("All", "all"), ("Pipe", "pipe"),
                            ("Rebar", "rebar"), ("Tunnel", "tunnel")]:
            tk.Radiobutton(
                filter_frame, text=label, variable=self._filter, value=val,
                bg="#1e1e1e", fg="#aaaaaa", selectcolor="#333333",
                activebackground="#1e1e1e", activeforeground="white",
                command=self._apply_filter,
            ).pack(side=tk.LEFT, padx=2)

        # 목록 박스
        lb_frame = tk.Frame(left, bg="#1e1e1e")
        lb_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        scrollbar = tk.Scrollbar(lb_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(
            lb_frame, yscrollcommand=scrollbar.set,
            bg="#252526", fg="#d4d4d4", selectbackground="#094771",
            font=("Consolas", 8), activestyle="none",
            width=30,
        )
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        self._filtered = list(range(len(self.file_list)))
        self._refresh_listbox()

        # ── 우측 패널 ──
        right = tk.Frame(self.root, bg="#121212")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 이미지
        img_frame = tk.Frame(right, bg="#000000",
                              width=DISPLAY_W, height=DISPLAY_H)
        img_frame.pack(padx=8, pady=8)
        img_frame.pack_propagate(False)

        self.img_label = tk.Label(img_frame, bg="#000000")
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # 상태 텍스트
        self.status_var = tk.StringVar(value="파일을 선택하세요")
        tk.Label(right, textvariable=self.status_var,
                 bg="#121212", fg="#888888",
                 font=("Consolas", 9), anchor="w", wraplength=620,
                 justify="left").pack(fill=tk.X, padx=10)

        # conf 슬라이더
        ctrl = tk.Frame(right, bg="#121212")
        ctrl.pack(fill=tk.X, padx=10, pady=6)

        tk.Label(ctrl, text="Conf threshold:", bg="#121212",
                 fg="#aaaaaa", font=("Consolas", 9)).pack(side=tk.LEFT)
        self.conf_var = tk.DoubleVar(value=0.05)
        conf_slider = tk.Scale(
            ctrl, from_=0.01, to=0.50, resolution=0.01,
            orient=tk.HORIZONTAL, variable=self.conf_var, length=200,
            bg="#121212", fg="#cccccc", troughcolor="#333333",
            highlightthickness=0,
            command=lambda _: self._reload(),
        )
        conf_slider.pack(side=tk.LEFT, padx=6)
        self.conf_lbl = tk.Label(ctrl, textvariable=self.conf_var,
                                 bg="#121212", fg="#cccccc",
                                 font=("Consolas", 9), width=4)
        self.conf_lbl.pack(side=tk.LEFT)

        # 이전/다음
        nav = tk.Frame(right, bg="#121212")
        nav.pack(pady=4)
        btn_style = {"bg": "#333333", "fg": "white", "font": ("Consolas", 9),
                     "relief": "flat", "padx": 12, "pady": 4,
                     "activebackground": "#555555"}
        tk.Button(nav, text="◀ Prev", command=self._prev, **btn_style).pack(side=tk.LEFT, padx=4)
        tk.Button(nav, text="Next ▶", command=self._next, **btn_style).pack(side=tk.LEFT, padx=4)

    # ── 파일 목록 ────────────────────────────
    def _refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        flt = self._filter.get()
        self._filtered = [
            i for i, (cls, _) in enumerate(self.file_list)
            if flt == "all" or cls == flt
        ]
        for i in self._filtered:
            cls, path = self.file_list[i]
            self.listbox.insert(tk.END, f"[{cls[:3]}] {path.name}")

        cls_colors = {"pip": "#4fc3f7", "reb": "#81c784",
                      "tun": "#ffb74d", "sin": "#e57373"}
        for pos, i in enumerate(self._filtered):
            cls = self.file_list[i][0][:3]
            self.listbox.itemconfig(pos, foreground=cls_colors.get(cls, "#cccccc"))

    def _apply_filter(self):
        self._refresh_listbox()
        if self._filtered:
            self._load(self._filtered[0])

    def _on_select(self, event):
        sel = self.listbox.curselection()
        if sel:
            pos = sel[0]
            real_idx = self._filtered[pos]
            self._load(real_idx)

    # ── 로드 + 추론 ──────────────────────────
    def _load(self, idx: int):
        if self._busy:
            return
        self.cur_idx = idx
        cls_name, dt_path = self.file_list[idx]
        self.status_var.set(f"로딩 중... {dt_path.parent.name}/{dt_path.name}")
        self.root.update_idletasks()
        threading.Thread(target=self._load_thread,
                         args=(idx, cls_name, dt_path), daemon=True).start()

    def _load_thread(self, idx, cls_name, dt_path):
        self._busy = True
        key = str(dt_path)
        if key not in self._cache:
            img_bgr = preprocess_dt(dt_path)
            self._cache[key] = img_bgr
        else:
            img_bgr = self._cache[key]

        if img_bgr is None:
            self.root.after(0, lambda: self.status_var.set(
                f"[오류] 전처리 실패: {dt_path.name}"))
            self._busy = False
            return

        conf_thresh = self.conf_var.get()
        dets = run_inference(self.model, img_bgr, conf_thresh)
        annotated = annotate(img_bgr, dets, cls_name)

        # Tk는 메인 스레드에서만 업데이트 가능
        self.root.after(0, lambda: self._update_display(
            annotated, cls_name, dt_path, dets, idx))
        self._busy = False

    def _update_display(self, annotated, cls_name, dt_path, dets, idx):
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.img_label.config(image=self._tk_img)

        pred_names = list({d["cls_name"] for d in dets})
        tp = cls_name in pred_names
        status = (f"[{idx+1}/{len(self.file_list)}]  "
                  f"True: {cls_name}  |  Pred: {pred_names or ['none']}  |  "
                  f"{'✓ TP' if tp else '✗ Miss'} | "
                  f"conf>={self.conf_var.get():.2f}")
        self.status_var.set(status)

        # 목록에서 해당 항목 하이라이트
        if idx in self._filtered:
            pos = self._filtered.index(idx)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(pos)
            self.listbox.see(pos)

    def _reload(self):
        """conf 변경 시 현재 파일 재추론."""
        if not self._busy:
            self._load(self.cur_idx)

    def _prev(self):
        if not self._filtered:
            return
        try:
            pos = self._filtered.index(self.cur_idx)
        except ValueError:
            pos = 0
        if pos > 0:
            self._load(self._filtered[pos - 1])

    def _next(self):
        if not self._filtered:
            return
        try:
            pos = self._filtered.index(self.cur_idx)
        except ValueError:
            pos = -1
        if pos < len(self._filtered) - 1:
            self._load(self._filtered[pos + 1])


# ─────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Phase F-3 - Interactive Viewer")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[오류] pip install ultralytics")
        return

    if not MODEL_PT.exists():
        print(f"[오류] 모델 없음: {MODEL_PT}")
        return

    print("모델 로드 중...")
    model = YOLO(str(MODEL_PT))

    print("테스트 파일 수집 중...")
    trained_stems = _load_trained_stems()
    file_list = collect_files(trained_stems)
    print(f"  {len(file_list)}개 파일 로드됨")
    for cls in CATEGORIES:
        n = sum(1 for c, _ in file_list if c == cls)
        print(f"  {cls}: {n}개")

    if not file_list:
        print("[오류] 파일 없음")
        return

    print("\nGUI 시작 중...")
    root = tk.Tk()
    root.configure(bg="#121212")

    app = GPRViewer(root, file_list, model)
    root.mainloop()


if __name__ == "__main__":
    main()
