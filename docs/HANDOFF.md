# SHOT Phase 1 인수인계 문서

> 최종 업데이트: 2026-03-18

## 프로젝트 개요

SHOT은 핸드폰 카메라로 테니스 코트 라인을 실시간 인식하는 Android 앱이다.
**키포인트 검출 + 호모그래피 역산** 방식으로 동작한다:

1. TFLite 모델이 카메라 프레임에서 니어코트 8개 키포인트(9~16번) 검출
2. ITF 표준 치수 + 호모그래피로 전체 16개 코트 포인트 역산
3. 코트 라인 오버레이 렌더링

---

## 완료된 작업

### 1. Android 프로젝트 스캐폴딩 (Phase 1A)

멀티모듈 구조 구성 완료:

```
SHOT/
├── app/                    # 메인 앱 (Hilt, Compose, ViewModel)
├── camera/                 # CameraX 관리
├── court-detection/        # TFLite 모델 추론
│   └── src/main/assets/
│       └── court_keypoint.tflite  ← 학습된 모델 (4.25MB)
├── court-model/            # 호모그래피 계산, 검증, 평활화
├── core/                   # 데이터 모델, ITF 코트 스펙
├── ml/                     # Python ML 학습 파이프라인
└── docs/
    └── superpowers/specs/  # 설계 스펙 문서
```

**주요 Kotlin 파일:**

| 파일 | 역할 | 상태 |
|------|------|------|
| `core/.../Keypoint.kt` | 키포인트 데이터 클래스 | 완료 |
| `core/.../CourtDetectionResult.kt` | 검출 결과 모델 | 완료 |
| `core/.../ItfCourtSpec.kt` | ITF 규격 16개 좌표 (미터) | 완료 |
| `camera/.../CameraManager.kt` | CameraX Preview + ImageAnalysis | 완료 |
| `court-detection/.../CourtKeypointDetector.kt` | TFLite 추론 (NHWC) | 완료 |
| `court-model/.../HomographyCalculator.kt` | OpenCV findHomography(RANSAC) | 완료 |
| `court-model/.../CourtProjector.kt` | H⁻¹로 16개 포인트 투영 | 완료 |
| `court-model/.../HomographyValidator.kt` | 재투영 오차, 행렬식 검증 | 완료 |
| `court-model/.../TemporalSmoother.kt` | EMA 시간적 평활화 | 완료 |
| `app/.../CameraViewModel.kt` | 파이프라인 오케스트레이션 | 완료 |
| `app/.../CameraScreen.kt` | 카메라 프리뷰 Compose UI | 완료 |

### 2. ML 학습 파이프라인 (Phase 1B)

**Python 파일 (`ml/src/`):**

| 파일 | 역할 |
|------|------|
| `model.py` | MobileNetV3-Small + 키포인트 회귀 헤드 (1.1M params) |
| `dataset.py` | CourtKeypointDataset (JSON 어노테이션 로더) |
| `augmentations.py` | Albumentations 증강 (색상, 기하, 노이즈, 그림자) |
| `train.py` | 학습 루프 (AdamW, CosineAnnealingLR, early stopping) |
| `export_tflite.py` | PyTorch → ONNX → onnx2tf → TFLite 변환 |
| `convert_dataset.py` | yastrebksv 14-keypoint → SHOT 8-keypoint 변환 |

### 3. 모델 학습 결과

**데이터셋**: yastrebksv/TennisCourtDetector (방송 카메라 영상)
- 8,810장 (14 키포인트 → SHOT 8 키포인트 매핑)
- Train/Val: 80/20 split
- 이미지 해상도: 1280x720

**학습 설정:**
- Optimizer: AdamW (lr=0.001, weight_decay=1e-4)
- Scheduler: CosineAnnealingLR (T_max=100)
- Batch size: 32
- Early stopping: patience=15
- Loss: SmoothL1(좌표) + 0.5 * BCE(신뢰도), 키포인트별 가중치 차등

**결과 (best model, epoch 57, val_loss=0.0053):**

| 키포인트 | 설명 | 오차(px/256) | 목표(px) | 달성 |
|----------|------|-------------|---------|------|
| Pt9  | 서비스라인 좌 (singles left) | 5.32 | <3 | X |
| Pt10 | 서비스라인 중앙 (center mark) | 4.88 | <3 | X |
| Pt11 | 서비스라인 우 (singles right) | 7.70 | <3 | X |
| Pt12 | 베이스라인 복식 좌 (doubles left) | 6.47 | <4 | X |
| Pt13 | 베이스라인 단식 좌 (singles left) | 6.86 | <4 | X |
| Pt14 | 베이스라인 중앙 (center mark) | 5.22 | <2 | X |
| Pt15 | 베이스라인 단식 우 (singles right) | 5.32 | <4 | X |
| Pt16 | 베이스라인 복식 우 (doubles right) | 4.95 | <4 | X |

**TFLite 모델:**
- 크기: 4.25 MB (FP32), 목표 <10MB 달성
- 입력: `[1, 256, 256, 3]` NHWC float32, ImageNet 정규화
- 출력: `[1, 24]` float32 (8 keypoints × [x, y, confidence])
- 배치 위치: `court-detection/src/main/assets/court_keypoint.tflite`

---

## 알려진 이슈 및 한계

### 1. 도메인 갭 (최대 리스크)

현재 모델은 **방송 카메라(높은 앵글, 줌렌즈)** 데이터로만 학습되었다.
실제 사용 환경은 **핸드폰 카메라(낮은 앵글, 광각)** 이므로 성능 저하가 예상된다.

| 항목 | 학습 데이터 (방송) | 실제 환경 (핸드폰) |
|------|------------------|-------------------|
| 카메라 높이 | 5~10m | 0.5~3m |
| 렌즈 | 줌, 좁은 FOV | 광각, 넓은 FOV |
| 앵글 | 위에서 내려다봄 | 거의 수평 |
| 왜곡 | 거의 없음 | 광각 왜곡 있음 |
| 코트 범위 | 전체 코트 | 니어코트 위주 |

**해결 방안:**
1. 핸드폰으로 직접 촬영한 데이터 100~200장 수집 + 라벨링 → fine-tune
2. 기존 방송 데이터에 강한 perspective 증강 적용하여 핸드폰 시점 시뮬레이션
3. 앱이 작동하면 모델 예측값을 초기값으로 사용하여 라벨링 효율화 가능

### 2. 키포인트 정확도 미달

목표 대비 약 1.5~2배 오차. 호모그래피 계산 시 8개 포인트 사용으로 개별 오차 상쇄 가능성 있으나, 검증 필요.

### 3. Android 빌드 미검증

Gradle 빌드가 실제 Android Studio에서 테스트되지 않았다. 의존성 충돌이나 컴파일 에러 가능성 있음.

### 4. albumentations 경고

v2.0.8에서 `OpticalDistortion`의 `shift_limit`, `GaussNoise`의 `var_limit` deprecation 경고 발생. 기능에는 영향 없으나 추후 API 변경 시 수정 필요.

---

## 환경 설정

### Python ML 환경

```bash
# Python 3.14는 TensorFlow/albumentations와 호환 안됨 → 3.12 사용
cd ml
python -m uv venv .venv312 --python 3.12
python -m uv pip install --python .venv312/Scripts/python.exe -r requirements.txt

# 학습 실행
.venv312/Scripts/python.exe src/train.py \
  --data data/raw/annotations.json \
  --image-dir "C:/path/to/tennis_court/images" \
  --epochs 100 --batch-size 32

# TFLite 변환
.venv312/Scripts/python.exe src/export_tflite.py \
  --checkpoint models/best_model.pth \
  --output ../court-detection/src/main/assets/court_keypoint.tflite
```

### TFLite 변환 파이프라인

```
PyTorch (NCHW) → ONNX (opset 18) → onnx2tf → TFLite (NHWC)
```

핵심 파라미터: `keep_ncw_or_nchw_or_ncdhw_input_names=["input"]`
이 옵션 없으면 onnx2tf가 convolution layer에서 에러 발생.

### 데이터셋

yastrebksv/TennisCourtDetector (MIT 라이선스):
- 다운로드: Kaggle에서 `yastrebksv/tennis-court-detector` 검색
- 변환: `python convert_dataset.py --input data/raw/tennis_court --output data/raw/annotations.json --copy-images`

**키포인트 매핑 (yastrebksv → SHOT):**

| SHOT ID | 설명 | yastrebksv index |
|---------|------|-----------------|
| 9 | 서비스라인 singles left | 10 |
| 10 | 서비스라인 center mark | 13 |
| 11 | 서비스라인 singles right | 11 |
| 12 | 베이스라인 doubles left | 2 |
| 13 | 베이스라인 singles left | 5 |
| 14 | 베이스라인 center mark | midpoint(2, 3) |
| 15 | 베이스라인 singles right | 7 |
| 16 | 베이스라인 doubles right | 3 |

---

## 현재 진행 중: YouTube 데이터 수집 파이프라인

### 배경

현재 모델은 **방송 카메라 데이터**로만 학습되어 **핸드폰 카메라 환경에서 도메인 갭** 발생.
이를 해결하기 위해 YouTube에서 동호인 테니스 영상(핸드폰/고정 카메라 촬영)을 수집하여
학습 데이터를 보강하는 파이프라인을 구축.

### 핵심 전략

- **한 영상에서 많은 프레임이 아니라, 다양한 영상에서 각 1~3프레임** 추출
- 고정 카메라 영상이라도 영상 수 자체가 다양성을 보장 (코트, 위치, 높이, 조명 모두 다름)
- 200개 이상 영상에서 각 1장 = **200개 이상 고유 시점** 확보 가능
- **자동화는 수집/추출까지만, 학습 데이터 승인은 반드시 수동 검수 후 진행**

### 데이터 파이프라인 워크플로우

```
[Step 1: 자동] YouTube 검색 → URL 리스트 수집
    python youtube_collect.py --output data/youtube/video_list.json
    python youtube_collect.py --url-file data/youtube/manual_urls.txt --skip-search --output data/youtube/video_list.json

[Step 2: 자동] 영상 다운로드 → 대표 프레임 추출
    python extract_frames.py --input data/youtube/video_list.json --output data/youtube/frames

[Step 3: 자동] 기존 모델로 키포인트 예측 → 미리보기 생성
    python predict_and_preview.py --frames data/youtube/frames --model models/best_model.pth

[Step 4: 수동] 사람이 직접 검수 (승인/거부)
    python review_data.py --predictions data/youtube/predictions.json --preview-dir data/youtube/preview

[Step 5: 학습] 검수 완료된 데이터만으로 fine-tune
    python train.py --data data/youtube/approved_annotations.json --image-dir data/youtube/frames
```

### 추가된 스크립트 (`ml/src/`)

| 파일 | 역할 |
|------|------|
| `youtube_collect.py` | YouTube 검색 + 수동 URL 수집 → video_list.json 생성 |
| `extract_frames.py` | yt-dlp로 다운로드 + OpenCV로 대표 프레임 추출 |
| `predict_and_preview.py` | 기존 모델로 키포인트 예측 + 시각적 미리보기 생성 |
| `review_data.py` | 대화형 검수 도구 (승인/거부/스킵), approved_annotations.json 생성 |
| `labeling_tool.py` | 브라우저 기반 키포인트 라벨링 도구 (클릭으로 8포인트 지정) |
| `sync_delete.py` | previews 폴더 삭제 → frames 폴더 자동 동기화 |
| `train_compare.py` | 3가지 비교 실험 (방송/핸드폰/합산) 학습 + 결과 비교 |
| `prepare_broadcast_data.py` | yastrebksv 방송 데이터 다운로드 + SHOT 포맷 변환 |
| `model_heatmap.py` | **히트맵 기반 키포인트 검출 모델** (MobileNetV3 + decoder) |
| `augmentations_v2.py` | 강한 도메인 갭 축소 augmentation (perspective, 광각, blur 등) |
| `train_3stage.py` | 3단계 학습 파이프라인 (pretrain → mixed → fine-tune) |

### 수동 URL 리스트

`ml/data/youtube/manual_urls.txt`에 확인 완료된 5개 영상 URL 등록:
- 아이언쑨 - 테슬로 복식 (14분)
- 리서땡 - 혼복/잡복 귀뚜라미크린 (30분, 21분)
- 몽돌브라더스 - 나눔귀뚜라미 (1시간56분, 4분)

### 의존성 추가

기존 `requirements.txt` 외 추가 설치 필요:
```bash
pip install yt-dlp opencv-python
```

---

## 완료된 작업: YouTube 데이터 수집 & 라벨링 (2026-03-17)

### 데이터 수집 결과
- [x] YouTube 자동 검색으로 203개 영상 URL 수집 (11개 검색 쿼리)
- [x] 프레임 추출: 203개 영상 → 622 프레임 추출
- [x] 기존 모델로 키포인트 예측 + 미리보기 생성
- [x] 수동 검수: 쓰레기 데이터 262장 삭제 → **360장** 선별
- [x] **수동 키포인트 라벨링 완료**: 339장 (브라우저 기반 라벨링 도구 사용)

### 라벨링 데이터 위치
- 프레임 이미지: `ml/data/youtube/review/frames/` (360장)
- 라벨 어노테이션: `ml/data/youtube/labeled_annotations.json` (339장)
- 포맷: SHOT JSON (image, keypoints{9~16: {x, y, visible}})

### 3가지 비교 실험 (FC regression, 2026-03-18 실행)

| 실험 | 학습 데이터 | 핸드폰 테스트 오차 | 비고 |
|-----|-----------|-----------------|------|
| A: 방송만 | 1,000장 | 37.06px | 도메인 갭 확인 |
| B: 핸드폰만 | 272장 | 24.03px | 데이터 부족 |
| C: 합산 | 1,272장 | 24.95px | 단순 합산 비효과적 |

**결론**: FC regression으로는 목표(2~4px) 달성 불가. 아키텍처 변경 필요.

### 아키텍처 변경: 히트맵 기반 모델 (2026-03-18)

- `model_heatmap.py`: MobileNetV3-Small + ConvTranspose2d decoder → 8채널 64x64 히트맵
- soft-argmax 버그 발견 → argmax + sub-pixel refinement로 수정 (81px → 1.34px)
- Stage 1 (방송 2000장 pretrain) 에폭 10에서 27px까지 감소 확인
- **회사 노트북 GPU(RTX 2070 Super) 불안정** → Google Colab으로 전환

---

## 남은 작업 (우선순위 순)

### 즉시: Google Colab에서 3단계 학습 실행

**Colab 노트북**: `ml/colab_train_3stage.ipynb` (repo에 포함)

**사전 준비:**
1. `ml/data/youtube/youtube_labeled_data.zip` (26MB)을 Google Drive의 `SHOT-AI/` 폴더에 업로드
2. Colab에서 노트북 열기 → Runtime > T4 GPU 선택 → 순서대로 실행
3. 방송 데이터(7.3GB)는 Colab에서 자동 다운로드됨

**3단계 학습 전략:**
- Stage 1: 방송 8,841장으로 pretrain (코트 기하 구조 학습)
- Stage 2: 방송 + 핸드폰 혼합 (50:50 오버샘플링)
- Stage 3: 핸드폰 272장만으로 fine-tune (낮은 LR)

**데이터 위치:**
- YouTube 라벨링 데이터 (339장): `ml/data/youtube/youtube_labeled_data.zip`
- 방송 데이터 (8,841장): [Google Drive](https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu) (Colab에서 자동 다운로드)

### 데이터 추가 확보 (병렬 진행)
- [ ] YouTube 영상 추가 수집 → 1,000장+ 목표
- [ ] 라벨링 도구에 모델 예측 보조 기능 추가 (시간 절약)
- [ ] Optical flow 기반 라벨 자동 전파 구현

### 추가 실험 후
- [ ] 최적 모델 선택 → TFLite 변환
- [ ] INT8 양자화 모델 생성 (현재 FP32 4.25MB → 목표 <5MB INT8)

### Phase 1C: 검출 파이프라인 통합
- [ ] Android 빌드 성공 확인 (Android Studio에서)
- [ ] 카메라 → 검출 → 호모그래피 → 투영 전체 파이프라인 연결
- [ ] CameraViewModel에서 실시간 오케스트레이션

### Phase 1D: UI 및 마무리
- [ ] 코트 라인 오버레이 (Compose Canvas)
- [ ] 키포인트 신뢰도 시각화
- [ ] 디버그 모드 (FPS, 추론시간, 재투영 오차)
- [ ] 설정 화면 (해상도, 단식/복식, 언어)
- [ ] 영상 녹화 (CameraX VideoCapture)

### Phase 1E: 검증
- [ ] 단위 테스트 (호모그래피, 검증, 평활화)
- [ ] 통합 테스트 (전체 파이프라인)
- [ ] 완료 게이트: 10초 연속 95% 프레임 성공

---

## Git 히스토리

| 커밋 | 내용 |
|------|------|
| `2f25931` | Phase 1 초기 구현 (Android 스캐폴딩 + ML 파이프라인) |
| `4b9fed9` | 학습된 TFLite 모델 추가 (4.25MB) |
| `4eabb29` | Phase 1 인수인계 문서 추가 |
| `62e58fc` | 키포인트 라벨링 도구 + TFLite 미리보기 생성기 |
| `74bdb09` | YouTube 데이터 수집 파이프라인 추가 |
| (다음) | 히트맵 모델 + 3단계 학습 + 오답노트 + Colab 노트북 |

**원격 저장소**: https://github.com/kokoro456/SHOT-AI

---

## 오답노트

자세한 실패 분석 및 교훈은 `docs/RETROSPECTIVE.md` 참조.

**핵심 요약:**
1. FC direct regression은 공간 정보 손실 → 히트맵 기반으로 변경
2. 339장은 부족 → 최소 1,000장 필요
3. 단순 합산보다 3단계 학습 (pretrain → mixed → fine-tune)
4. soft-argmax 함정: 평평한 히트맵에서 중심(0.5,0.5)으로 수렴 → argmax 사용
5. 테스트는 반드시 타겟 도메인(핸드폰)으로
