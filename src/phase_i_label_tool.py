"""
Phase I: pipe/rebar 수동 라벨링 툴

사용법
------
  /c/Python314/python.exe src/phase_i_label_tool.py

조작
------
  마우스 드래그 : bbox 그리기
  Enter / Space : 현재 bbox 저장 후 다음 이미지
  R             : 현재 bbox 초기화 (다시 그리기)
  S             : bbox 없이 스킵 (라벨 없음으로 저장)
  Q             : 종료

출력
------
  data/gpr/phase_i_manual/images/ : PNG 이미지
  data/gpr/phase_i_manual/labels/ : YOLO 형식 .txt 라벨
"""

import sys
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

import numpy as np
import cv2

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from week1_gpr_basics import read_ids_dt
from week2_preprocessing import (
    dc_removal, background_removal, bandpass_filter, gain_sec,
)

# -- 경로 ------------------------------------------------------------------
GZ_DATA  = BASE_DIR / "data/gpr/guangzhou/Data Set"
MANIFEST = BASE_DIR / "guangzhou_labeled/manifest.json"
OUT_DIR  = BASE_DIR / "data/gpr/phase_i_manual"

CATEGORIES   = ["pipe", "rebar"]
CLASS_IDS    = {"pipe": 1, "rebar": 2}
N_PER_CLASS  = 10
SEED         = 99   # Phase G/H와 다른 seed로 다양성 확보
DT_SEC       = (8.0 / 512) * 1e-9
DISPLAY_SIZE = 640   # 화면에 표시할 이미지 크기


# =========================================================================
# 데이터 준비
# =========================================================================

def preprocess_dt(dt_path: Path):
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
        return cv2.resize(gray, (640, 640))
    except Exception:
        return None


def load_manifest_zon_dirs() -> set[Path]:
    try:
        raw  = MANIFEST.read_bytes()
        data = json.loads(raw.decode("cp949"))
    except Exception:
        return set()
    return {
        Path(e["source"]).parent
        for e in data.get("images", [])
        if e.get("source")
    }


def prepare_images():
    """새 ZON에서 클래스별 N_PER_CLASS 이미지 추출."""
    import random
    rng = random.Random(SEED)

    trained = load_manifest_zon_dirs()
    items   = []  # [(cls, img_gray, stem), ...]

    for cls in CATEGORIES:
        cls_dir   = GZ_DATA / cls
        all_zons  = sorted(
            d for d in cls_dir.rglob("*.ZON")
            if d.is_dir() and "ASCII" not in str(d)
        )
        available = [z for z in all_zons if z not in trained]
        rng.shuffle(available)

        count = 0
        for zon_dir in available:
            if count >= N_PER_CLASS:
                break
            dts = sorted(zon_dir.glob("*.dt"))
            if not dts:
                continue
            gray = preprocess_dt(dts[0])
            if gray is None:
                continue
            stem = f"{cls}_{count:02d}"
            items.append((cls, gray, stem))
            count += 1

        print(f"  {cls}: {count}장 준비")

    return items


# =========================================================================
# 라벨링 툴
# =========================================================================

class LabelTool:
    def __init__(self, items: list):
        self.items   = items
        self.idx     = 0
        self.bbox    = None   # (x1, y1, x2, y2) 픽셀
        self.drawing = False
        self.start   = None
        self.labeled = 0
        self.skipped = 0

        # 출력 디렉토리
        (OUT_DIR / "images").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

        # tkinter 루트
        self.root = tk.Tk()
        self.root.title("Phase I - bbox 라벨링")
        self.root.resizable(False, False)

        # 상단 정보
        self.info_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.info_var,
                 font=("Consolas", 11), bg="#1a1a2e", fg="white",
                 anchor="w", padx=10).pack(fill="x")

        # 캔버스
        self.canvas = tk.Canvas(self.root,
                                width=DISPLAY_SIZE, height=DISPLAY_SIZE,
                                bg="black", cursor="crosshair")
        self.canvas.pack()

        # 하단 안내
        guide = ("드래그: bbox 그리기    Enter/Space: 저장    "
                 "R: 초기화    S: 스킵    Q: 종료")
        tk.Label(self.root, text=guide, font=("Consolas", 9),
                 bg="#2a2a4a", fg="#aaaaaa").pack(fill="x")

        # 이벤트 바인딩
        self.canvas.bind("<ButtonPress-1>",   self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",        self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>",  self.on_mouse_up)
        self.root.bind("<Return>",   self.on_save)
        self.root.bind("<space>",    self.on_save)
        self.root.bind("<r>",        self.on_reset)
        self.root.bind("<R>",        self.on_reset)
        self.root.bind("<s>",        self.on_skip)
        self.root.bind("<S>",        self.on_skip)
        self.root.bind("<q>",        self.on_quit)
        self.root.bind("<Q>",        self.on_quit)

        self.root.configure(bg="#1a1a2e")
        self._load_current()

    # -- 이미지 로드 -------------------------------------------------------

    def _load_current(self):
        if self.idx >= len(self.items):
            self._finish()
            return

        cls, gray, stem = self.items[self.idx]
        self.current_cls  = cls
        self.current_stem = stem
        self.current_gray = gray
        self.bbox = None

        # BGR 변환 후 tkinter PhotoImage 형식으로
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._base_bgr = bgr.copy()
        self._refresh_canvas(bgr)

        total = len(self.items)
        done  = self.labeled + self.skipped
        self.info_var.set(
            f"[{self.idx + 1}/{total}]  cls={cls}  |  "
            f"완료={self.labeled}  스킵={self.skipped}  남은={total - done}"
        )
        self.root.title(f"Phase I - {stem} ({cls})")

    def _refresh_canvas(self, bgr):
        """BGR numpy → tkinter PhotoImage로 캔버스 갱신."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil_img   = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            color_map = {"pipe": "#3498db", "rebar": "#2ecc71"}
            color = color_map.get(self.current_cls, "yellow")
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         outline=color, width=2)
            bw = (x2 - x1) / DISPLAY_SIZE
            bh = (y2 - y1) / DISPLAY_SIZE
            cy = ((y1 + y2) / 2) / DISPLAY_SIZE
            self.canvas.create_text(
                x1 + 4, y1 - 12, anchor="w",
                text=f"cy={cy:.2f}  bh={bh:.2f}",
                fill=color, font=("Consolas", 9, "bold"),
            )

    # -- 마우스 이벤트 -----------------------------------------------------

    def on_mouse_down(self, event):
        self.drawing = True
        self.start   = (event.x, event.y)
        self.bbox    = None

    def on_mouse_drag(self, event):
        if not self.drawing:
            return
        x1, y1 = self.start
        x2, y2 = event.x, event.y
        self.bbox = (min(x1, x2), min(y1, y2),
                     max(x1, x2), max(y1, y2))
        vis = self._base_bgr.copy()
        if self.bbox:
            bx1, by1, bx2, by2 = self.bbox
            color_map = {"pipe": (255, 100, 50), "rebar": (50, 200, 50)}
            color = color_map.get(self.current_cls, (255, 255, 0))
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
        self._refresh_canvas(vis)

    def on_mouse_up(self, event):
        self.drawing = False

    # -- 키 이벤트 ---------------------------------------------------------

    def on_save(self, event=None):
        if self.bbox is None:
            messagebox.showwarning("경고", "bbox를 그려주세요.\n없이 저장하려면 S(스킵)를 누르세요.")
            return

        cls, _, stem = self.items[self.idx]
        gray = self.current_gray
        H, W = gray.shape
        x1, y1, x2, y2 = self.bbox

        # 좌표 클리핑
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W - 1))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H - 1))

        if x2 - x1 < 5 or y2 - y1 < 5:
            messagebox.showwarning("경고", "bbox가 너무 작습니다. 다시 그려주세요.")
            return

        # YOLO 정규화
        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        cls_id = CLASS_IDS[cls]

        # 저장
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if ok:
            (OUT_DIR / "images" / f"{stem}.png").write_bytes(buf.tobytes())
        label = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        (OUT_DIR / "labels" / f"{stem}.txt").write_text(label)

        print(f"  저장: {stem}.txt  "
              f"cls={cls}({cls_id}) cx={cx:.2f} cy={cy:.2f} bw={bw:.2f} bh={bh:.2f}")

        self.labeled += 1
        self.idx     += 1
        self._load_current()

    def on_reset(self, event=None):
        self.bbox = None
        self._refresh_canvas(self._base_bgr.copy())

    def on_skip(self, event=None):
        cls, _, stem = self.items[self.idx]
        # 이미지는 저장하되 라벨 파일은 빈 파일
        gray = self.current_gray
        bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if ok:
            (OUT_DIR / "images" / f"{stem}.png").write_bytes(buf.tobytes())
        (OUT_DIR / "labels" / f"{stem}.txt").write_text("")
        print(f"  스킵: {stem} ({cls})")
        self.skipped += 1
        self.idx     += 1
        self._load_current()

    def on_quit(self, event=None):
        if messagebox.askyesno("종료", "라벨링을 중단하고 종료하시겠습니까?\n(저장된 라벨은 유지됩니다)"):
            self.root.destroy()

    def _finish(self):
        msg = (f"라벨링 완료!\n\n"
               f"저장: {self.labeled}장\n"
               f"스킵: {self.skipped}장\n\n"
               f"이제 phase_i_train.py를 실행하세요.")
        messagebox.showinfo("완료", msg)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# =========================================================================
# 메인
# =========================================================================

def main():
    # Pillow 확인
    try:
        from PIL import Image, ImageTk  # noqa
    except ImportError:
        print("[오류] pip install Pillow")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Phase I - bbox 라벨링 툴")
    print("=" * 60)
    print(f"\n  대상: pipe {N_PER_CLASS}장, rebar {N_PER_CLASS}장")
    print(f"  출력: {OUT_DIR.relative_to(BASE_DIR)}/")

    # 이미 라벨된 파일 확인
    existing = list((OUT_DIR / "labels").glob("*.txt")) if (OUT_DIR / "labels").exists() else []
    if existing:
        print(f"\n  기존 라벨 {len(existing)}개 발견.")

    print("\n  이미지 준비 중...")
    items = prepare_images()

    # 이미 완료된 항목 제외
    done_stems = {p.stem for p in existing if p.stat().st_size > 0} if existing else set()
    items = [(cls, gray, stem) for cls, gray, stem in items
             if stem not in done_stems]

    if not items:
        print("\n  모든 이미지 라벨링 완료!")
        print("  phase_i_train.py를 실행하세요.")
        return

    print(f"\n  남은 이미지: {len(items)}장")
    print("\n  조작 방법:")
    print("    마우스 드래그 : bbox 그리기")
    print("    Enter / Space : 저장 후 다음")
    print("    R             : 현재 bbox 초기화")
    print("    S             : 스킵")
    print("    Q             : 종료")
    print("\n  라벨링 툴 실행...")

    tool = LabelTool(items)
    tool.run()

    # 완료 통계
    labeled_files = list((OUT_DIR / "labels").glob("*.txt"))
    valid = [f for f in labeled_files if f.stat().st_size > 0]
    print(f"\n  총 라벨: {len(valid)}개 / {len(labeled_files)}개")
    print(f"  저장 위치: {OUT_DIR.relative_to(BASE_DIR)}/")

    if len(valid) >= 5:
        print("\n  라벨링 완료 후 실행:")
        print("  /c/Python314/python.exe src/phase_i_train.py")


if __name__ == "__main__":
    main()
