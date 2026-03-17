# SHOT Phase 1 인수인계 문서

> 최종 업데이트: 2026-03-17

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

## 남은 작업 (우선순위 순)

### Phase 1C: 검출 파이프라인 통합
- [ ] Android 빌드 성공 확인 (Android Studio에서)
- [ ] 카메라 → 검출 → 호모그래피 → 투영 전체 파이프라인 연결
- [ ] CameraViewModel에서 실시간 오케스트레이션

### 모델 개선
- [ ] 핸드폰 카메라 데이터 수집 (100~200장)
- [ ] perspective 증강 강화하여 재학습
- [ ] INT8 양자화 모델 생성 (현재 FP32 4.25MB → 목표 <5MB INT8)

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

**원격 저장소**: https://github.com/kokoro456/SHOT-AI
