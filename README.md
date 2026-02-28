# GPR Sinkhole Detection

GPR(지중레이더) 데이터 처리부터 YOLOv11 기반 싱크홀/매설물 자동 탐지까지의 실습 프로젝트.

## Overview

| 단계 | 주제 | 핵심 내용 |
|------|------|-----------|
| Week 1 | 데이터 수집 & 시각화 | DT1/DZT/IDS .dt 파서, B-scan 시각화 |
| Week 2 | 전처리 & DB | 6단계 파이프라인 (DC→Dewow→BGRemoval→Bandpass→Gain→Migration), SQLite |
| Week 3 | 오픈소스 분석 & 시뮬레이션 | GPRPy/gprMax/siina 비교, 해석적 합성 B-scan 48개 |
| Week 4 | 싱크홀 탐지 | YOLOv11n 학습, mAP50=0.995, 실제 데이터 zero-shot 추론 |
| Phase A | Confidence Threshold 실험 | conf 조정 → TP=0, 도메인 갭 확인 |
| Phase B | gprMax FDTD 시뮬레이션 | Maxwell 방정식 기반 물리 시뮬레이션 6개 |
| Phase C | Pseudo-labeling | Semi-supervised 시도 → 0% 수락률, 도메인 갭 한계 |
| Phase D-1 | 실측 데이터 Fine-tuning | Mendeley GPR 공개 데이터셋 혼합 학습, mAP50=0.718 |
| Phase D-2 | FDTD 데이터 확장 | gprMax 6→18개 확장 + flip 증강, mAP50=0.939 |
| **Phase E-1** | **Guangzhou 직접 라벨링** | **실측 25개 직접 라벨 + 합성 혼합, mAP50=0.848, 도메인 갭 해소** |
| Phase E-2 | 라벨 품질 개선 + Tunnel 클래스 추가 | CC 기반 bbox 개선, tunnel 10개 추가, 4클래스, mAP50=0.679 |
| **Phase F** | **배치 추론 + 현장 워크플로우 검증** | **미학습 .dt 60개 배치 추론, 슬라이딩 윈도우 GIF, 인터랙티브 뷰어** |

## Results

### Week 4: YOLOv11 Sinkhole Detection (단일 클래스)

| Metric | Value |
|--------|-------|
| mAP50 | **0.995** |
| mAP50-95 | **0.974** |
| Precision | **0.999** |
| Recall | **1.000** |

- 264개 학습 이미지 (합성 B-scan 48개 + 위치 증강 + 전처리 증강)
- YOLOv11n (2.6M params), COCO pretrained, backbone freeze
- 학습 시간: 8.6분 (RTX 3060)
- 실제 GPR 데이터 zero-shot 추론: False Positive 0건

### Multiclass YOLO (3클래스: sinkhole/pipe/rebar + negative)

| Metric | Value |
|--------|-------|
| mAP50 | **0.984** |
| mAP50-95 | **0.843** |
| 데이터 | 합성 219개 × 6 증강 = 1038개 |
| negative sample | background 27개 (FP 억제) |

클래스별 mAP50:

| 클래스 | mAP50 |
|--------|-------|
| sinkhole | 0.989 |
| pipe | 0.988 |
| rebar | 0.974 |

### Phase D-1: Mendeley 실측 GPR Fine-tuning

| Metric | Value |
|--------|-------|
| Best mAP50 (혼합 val) | **0.718** (epoch 47/50) |
| 데이터 | Mendeley 750개 + 합성 375개 |
| Guangzhou 탐지 | 여전히 0건 (장비 간 도메인 갭) |

### Phase D-2: FDTD 데이터 확장 Fine-tuning

| Metric | Value |
|--------|-------|
| Best mAP50 (혼합 val) | **0.939** (epoch 26/45) |
| Best mAP50-95 | 0.747 |
| FDTD 시나리오 | 6 → 18개 (+ flip 증강 = 36개) |
| Guangzhou 탐지 | 여전히 0건 |

### Phase E-1: Guangzhou 실측 직접 라벨링 Fine-tuning ✅ 도메인 갭 해소

| Metric | Value |
|--------|-------|
| **mAP50** | **0.848** |
| **mAP50-95** | **0.672** |
| 학습 시간 | 566초 (~9분) |
| Guangzhou 탐지 | **성공** (Phase D까지 0건 → 탐지 성공) |

- Guangzhou .dt → 640×640 PNG 25개 (pipe×15, rebar×10)
- 신호 에너지 기반 자동 bbox 생성 (`phase_e1_auto_label.py`)
- Phase D-2 weights → 실측 20장 + 합성 150장 혼합 fine-tuning (lr=3e-5)

### Phase E-2: 라벨 품질 개선 + Tunnel 클래스 추가 (4클래스)

| Metric | Value |
|--------|-------|
| **mAP50** | **0.679** |
| **mAP50-95** | **0.502** |
| 학습 시간 | 274초 (~4.5분) |
| epochs | 33 (early stop, best=13) |
| 클래스 | sinkhole / pipe / rebar / tunnel (4클래스) |

클래스별 mAP50:

| 클래스 | mAP50 |
|--------|-------|
| sinkhole | - |
| pipe | ~0.77 |
| rebar | ~0.81 |
| tunnel | ~0.46 |

- E-1의 전체 폭 bbox(w≈0.998) → Connected Components 기반 정밀 bbox로 개선
- NJZ .dt(IDS, 2GHz) → tunnel PNG 10개 추가, class_id=3
- 총 35개 실측 라벨(pipe×15, rebar×10, tunnel×10) + 합성 150장 혼합
- train=178, val=37, imgsz=416, batch=2 (NAS pagefile 메모리 제약)

### Phase F: 배치 추론 + 현장 워크플로우 검증

Phase E-2 모델(mAP50=0.679)을 학습에 사용되지 않은 Guangzhou .dt 파일에 적용해 전체 파이프라인을 검증.

| 항목 | 내용 |
|------|------|
| 대상 파일 | pipe×20 / rebar×20 / tunnel×20 (학습 ZON 제외) |
| conf threshold | 0.05 |
| Recall | pipe 0.00 / rebar 0.00 / tunnel 0.00 |
| FP 패턴 | 탐지 시 대부분 sinkhole 클래스로 오분류 |

**3가지 구현 방식:**

| 스크립트 | 방식 | 출력 |
|----------|------|------|
| `phase_f_realtime_inference.py` | 배치 추론 | PNG 60개 + detection_log.csv + summary 그리드 |
| `phase_f2_sliding_window.py` | 슬라이딩 윈도우 | 클래스별 GIF (pipe 50프레임 / rebar 27프레임 / tunnel 50프레임) |
| `phase_f3_interactive_viewer.py` | 인터랙티브 뷰어 | tkinter GUI (파일 목록 + conf 슬라이더 실시간 탐지) |

**낮은 Recall 원인 분석:**
- 학습 데이터 35개(pseudo-label 기반) → 미학습 ZON에 대한 일반화 부족
- 베이스 모델의 sinkhole 편향 → FP 예측이 주로 sinkhole 클래스로 출력
- 학습 ZON과 테스트 ZON의 스캔 조건 차이 (도메인 갭)
- → 추가 라벨링 + re-finetune으로 개선 가능

### 도메인 적응 전체 비교

| Phase | 방법 | val mAP50 | Guangzhou 탐지 |
|-------|------|----------|---------------|
| 4주차 | 합성 전용 | 0.995 | 0건 |
| C | Pseudo-labeling | 0.985 | 0건 |
| D-1 | Mendeley 실측 fine-tune | 0.718 | 0건 |
| D-2 | FDTD 확장 | 0.939 | 0건 |
| **E-1** | **Guangzhou 직접 라벨** | **0.848** | **✅ 탐지 성공** |
| E-2 | CC bbox 개선 + tunnel 추가 | 0.679 | ✅ (4클래스 확장) |
| F | 배치 추론 (미학습 파일) | - | ⚠️ recall≈0 (도메인 갭, 추가 라벨 필요) |

### Domain Gap 분석 요약

```
합성(해석적/FDTD) → [갭1: conf=0.05에도 탐지 0] → Mendeley 실측 GPR
Mendeley 실측 → [갭2: fine-tuning 후에도 탐지 0] → Guangzhou IDS 실측
                                                          ↑
                                              Phase E-1: 직접 25개 라벨 → 해소 ✅

핵심 교훈: 소량(20장)이라도 타겟 도메인 직접 라벨이 간접 데이터 수천 장보다 효과적
```

## Project Structure

```
├── src/
│   ├── week1_gpr_basics.py        # DT1/DZT/IDS .dt 파서 + B-scan 시각화
│   ├── week2_preprocessing.py     # 6단계 전처리 파이프라인
│   ├── week2_database.py          # SQLAlchemy ORM (Dataset/Run/Step)
│   ├── week3_analysis.py          # GPRPy/gprMax/siina 오픈소스 분석
│   ├── week3_simulation.py        # 합성 B-scan 생성 (sinkhole/pipe/rebar/bg)
│   ├── week4_yolo_detection.py    # YOLOv11 데이터 준비 + 학습 + 평가
│   ├── phase_a_conf_threshold.py  # Confidence threshold 실험
│   ├── phase_b_fdtd.py            # gprMax FDTD 시뮬레이션
│   ├── phase_c_pseudolabel.py     # Pseudo-labeling + fine-tuning
│   ├── phase_d_realdata_finetune.py  # 실측 GPR fine-tuning (Mendeley)
│   ├── phase_d2_fdtd_expand.py    # FDTD 확장 fine-tuning
│   ├── phase_e1_prepare_labeling.py  # Guangzhou .dt → PNG 변환
│   ├── phase_e1_auto_label.py     # 신호 에너지 기반 자동 bbox 생성
│   ├── phase_e1_finetune.py       # Guangzhou 실측 라벨 fine-tuning
│   ├── phase_e2_relabel.py        # CC 기반 bbox 개선 + tunnel 10개 추가
│   ├── phase_e2_finetune.py       # 4클래스 fine-tuning (mAP50=0.679)
│   ├── phase_f_realtime_inference.py  # 배치 추론 + summary 그리드
│   ├── phase_f2_sliding_window.py     # 슬라이딩 윈도우 GIF 애니메이션
│   ├── phase_f3_interactive_viewer.py # tkinter 인터랙티브 뷰어
│   └── output/                    # 시각화 이미지
├── data/gpr/
│   ├── synthetic/                 # 합성 B-scan (.npy + _meta.json)
│   ├── yolo_multiclass/           # 3클래스 YOLO 데이터셋
│   ├── yolo_mixed_real/           # Mendeley+합성 혼합 데이터셋
│   ├── yolo_gz_e2_mixed/          # Phase E-2 혼합 데이터셋 (4클래스)
│   └── mendeley_gpr/              # Mendeley GPR 공개 데이터 (별도 다운로드)
├── models/
│   ├── fdtd_compact/              # gprMax .in + .out 파일
│   └── yolo_runs/
│       ├── multiclass_detect/     # 3클래스 모델 (best.pt, mAP50=0.984)
│       ├── finetune_pseudo/       # Phase C fine-tuned 모델
│       ├── finetune_real/         # Phase D-1 fine-tuned 모델 (mAP50=0.718)
│       ├── finetune_fdtd/         # Phase D-2 fine-tuned 모델 (mAP50=0.939)
│       ├── finetune_gz_e1/        # Phase E-1 fine-tuned 모델 (mAP50=0.848)
│       └── finetune_gz_e2/        # Phase E-2 fine-tuned 모델 (mAP50=0.679, 4클래스)
├── guangzhou_labeled/
│   ├── labels/                    # YOLO 라벨 35개 (pipe×15, rebar×10, tunnel×10)
│   ├── manifest.json              # 소스 경로 매핑 (nc=4)
│   └── auto_label_review.png      # 자동 라벨링 검토 이미지
├── db/gpr_processing.db           # SQLite
└── gpr_env.yml                    # conda 환경 설정
```

## Datasets Used

| 데이터셋 | 포맷 | 출처 | 용도 |
|----------|------|------|------|
| Frenke | Sensors & Software DT1 | Public | 하천 퇴적층 B-scan |
| NSGeophysics | GSSI DZT | GitHub | 사구 탐사 |
| Tagliamento | DT1 | Zenodo 2586189 | 하천 단면 |
| Guangzhou | IDS .dt | Zenodo 14637589 | 파이프/철근/터널 (2GHz) |
| Mendeley GPR | JPEG + YOLO | Mendeley Data (CC BY 4.0) | cavity/utility/intact 2239개 |
| Synthetic | NumPy (.npy) | 자체 생성 | 합성 B-scan (sinkhole/pipe/rebar/bg) |

> 실측 데이터는 용량 문제로 이 저장소에 포함되지 않습니다. 위 출처에서 직접 다운로드하세요.

## Preprocessing Pipeline (Week 2)

```
Raw B-scan
 → DC Removal (trace mean subtraction)
 → Dewow (low-frequency trend removal)
 → Background Removal (mean trace subtraction)
 → Bandpass Filter (Butterworth, freq-adaptive)
 → SEC Gain (spreading & exponential compensation)
 → FK Migration (Stolt, velocity-based)
```

## Setup

```bash
# conda 환경 생성
conda env create -f gpr_env.yml
conda activate gpr_rag

# PyTorch CUDA 확인 (필요 시 재설치)
python -c "import torch; print(torch.cuda.is_available())"
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Usage

```python
# Week 1~4: 기본 파이프라인
python src/week1_gpr_basics.py
python src/week2_preprocessing.py
python src/week3_simulation.py
python src/week4_yolo_detection.py

# Phase A~D: 도메인 적응 실험
python src/phase_a_conf_threshold.py   # conf threshold 실험
python src/phase_b_fdtd.py             # gprMax FDTD 시뮬레이션
python src/phase_c_pseudolabel.py      # pseudo-labeling
python src/phase_d_realdata_finetune.py  # Mendeley fine-tuning

# Phase E-1: Guangzhou 직접 라벨링 fine-tuning
python src/phase_e1_prepare_labeling.py  # .dt → PNG 25개 변환
python src/phase_e1_auto_label.py        # 자동 bbox 라벨 생성
python src/phase_e1_finetune.py          # fine-tuning (mAP50=0.848)

# Phase E-2: CC bbox 개선 + Tunnel 클래스 추가
python src/phase_e2_relabel.py           # CC 기반 bbox 재생성 + tunnel 10개 추가
python src/phase_e2_finetune.py          # 4클래스 fine-tuning (mAP50=0.679)

# Phase F: 배치 추론 + 현장 워크플로우 검증
python src/phase_f_realtime_inference.py    # 배치 추론 (PNG 60개 + CSV + summary)
python src/phase_f2_sliding_window.py       # 슬라이딩 윈도우 GIF (클래스별 애니메이션)
python src/phase_f3_interactive_viewer.py   # tkinter 인터랙티브 뷰어 (GUI)
```

## Troubleshooting (Windows)

| 문제 | 증상 | 해결 |
|------|------|------|
| PyTorch CPU-only | `torch.cuda.is_available()` → `False` | `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126` |
| 의존성 깨짐 | force-reinstall 후 `ModuleNotFoundError` | `pip install sqlalchemy scipy h5py`로 복구 |
| WinError 1455 | `paging file too small`, shm.dll 로드 실패 | `workers=0`, 다른 프로세스 종료로 RAM 확보 |
| CUDA OOM | YOLO 학습 중 GPU 메모리 부족 | `batch=4`, `amp=False`, `imgsz=416` |
| NumPy MemoryError | `(1280,1280,3)` 배열 할당 실패 | `imgsz=416` (mosaic 크기 축소) |
| cv2 채널 불일치 | `img.shape` 언패킹 에러 | `img.shape[:2]` 또는 `cv2.IMREAD_GRAYSCALE` 사용 |
| cv2 한글 경로 silent fail | `cv2.imwrite/imread`가 CJK 경로 미지원 | `imencode`+`write_bytes` / `read_bytes`+`imdecode` 사용 |
| LabelImg 스크롤 크래시 | PyQt5 1.8.6 Windows 버그 | 자동 라벨링 스크립트(`phase_e1_auto_label.py`) 대체 |
| SIGPIPE (pipe 사용 시) | `python script.py \| head -N` 으로 실행 시 훈련 중간 강제 종료 | `python script.py > log.txt 2>&1` 로 리다이렉트 |
| Mendeley 라벨 경로 | 이미지와 같은 폴더에 `.txt` 없음 | `annotations/Yolo_format/` 또는 `annotations/YOLO_format/` 하위 탐색 필요 |
| RAR 압축 해제 (Windows) | `unrar` 미설치 | `C:\Windows\System32\tar.exe -xf file.rar` (bsdtar 3.5.2 내장) |
| gprMax 빌드 (Windows) | MSVC 없음, Cython 컴파일 실패 | `pydistutils.cfg`에 `compiler=mingw32` 설정 후 MinGW gcc 사용 |
| bandpass_filter 단위 | `dt` 파라미터 혼동 (ns vs s) | 반드시 초(s) 단위: `dt_sec = dt_ns * 1e-9` |
| NAS pagefile 메모리 | `numpy._ArrayMemoryError: Unable to allocate 977 KiB` — 충분한 RAM에도 alloc 실패 | E:\pagefile.sys(NAS 마운트) 사용 시 발생. `mosaic=0.0, plots=False, batch=2` 설정, 다른 프로세스 종료 |
| cuBLAS pinned memory | `CUBLAS_STATUS_ALLOC_FAILED` — GPU 학습 초기 즉시 크래시 | CPU RAM 부족 시 cuBLAS가 pinned host memory 할당 실패. Chrome/Discord 등 종료로 5 GB 이상 확보 |
| YOLO nc 불일치 | `nc=3` 모델로 `nc=4` 데이터셋 학습 시 head 오류 | ultralytics 8.4.x는 자동으로 `Overriding model.yaml nc=3 with nc=4` 처리 — 별도 조치 불필요 |
| ultralytics 한글 경로 | dataset.yaml 경로가 'Ϸ' 등 깨진 문자로 읽힘 → FileNotFoundError | `yaml_path.write_text(..., encoding='utf-8')` 로 UTF-8 명시 저장 |

## Requirements

- Python 3.11
- PyTorch 2.x + CUDA 12.x
- ultralytics (YOLOv11)
- NumPy, SciPy, matplotlib, OpenCV
- SQLAlchemy, h5py
- gprMax 3.1.7 (FDTD, optional)

## License

This project is for educational and research purposes.
