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

### Domain Gap 분석 요약

```
합성 B-scan → [갭1: conf=0.05에도 탐지 0] → Mendeley 실측 GPR
Mendeley 실측 → [갭2: fine-tuning 후에도 탐지 0] → Guangzhou IDS 실측

"탐지 없음(background)" 일반화: ✅ (FP=0 유지)
"탐지 있음(객체)"   일반화: ❌ (TP=0, 합성→실측)
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
│   └── output/                    # 시각화 이미지
├── data/gpr/
│   ├── synthetic/                 # 합성 B-scan (.npy + _meta.json)
│   ├── yolo_multiclass/           # 3클래스 YOLO 데이터셋
│   ├── yolo_mixed_real/           # Mendeley+합성 혼합 데이터셋
│   └── mendeley_gpr/              # Mendeley GPR 공개 데이터 (별도 다운로드)
├── models/
│   ├── fdtd_compact/              # gprMax .in + .out 파일
│   └── yolo_runs/
│       ├── multiclass_detect/     # 3클래스 모델 (best.pt, mAP50=0.984)
│       ├── finetune_pseudo/       # Phase C fine-tuned 모델
│       └── finetune_real/         # Phase D-1 fine-tuned 모델 (mAP50=0.718)
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
| SIGPIPE (pipe 사용 시) | `python script.py \| head -N` 으로 실행 시 훈련 중간 강제 종료 | `python script.py > log.txt 2>&1` 로 리다이렉트 |
| Mendeley 라벨 경로 | 이미지와 같은 폴더에 `.txt` 없음 | `annotations/Yolo_format/` 또는 `annotations/YOLO_format/` 하위 탐색 필요 |
| RAR 압축 해제 (Windows) | `unrar` 미설치 | `C:\Windows\System32\tar.exe -xf file.rar` (bsdtar 3.5.2 내장) |
| gprMax 빌드 (Windows) | MSVC 없음, Cython 컴파일 실패 | `pydistutils.cfg`에 `compiler=mingw32` 설정 후 MinGW gcc 사용 |
| bandpass_filter 단위 | `dt` 파라미터 혼동 (ns vs s) | 반드시 초(s) 단위: `dt_sec = dt_ns * 1e-9` |

## Requirements

- Python 3.11
- PyTorch 2.x + CUDA 12.x
- ultralytics (YOLOv11)
- NumPy, SciPy, matplotlib, OpenCV
- SQLAlchemy, h5py
- gprMax 3.1.7 (FDTD, optional)

## License

This project is for educational and research purposes.
