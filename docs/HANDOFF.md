# SHOT 인수인계 문서

> 최종 업데이트: 2026-03-21

## 프로젝트 개요

SHOT은 핸드폰 카메라로 테니스 코트 라인을 실시간 인식하고, 볼을 추적하여 IN/OUT 판정을 자동화하는 Android 앱이다.

**핵심 기술:**
1. **코트 인식**: MobileNetV3-Small 히트맵 모델 → 니어코트 8개 키포인트 검출 → 호모그래피 역산
2. **볼 추적**: 경량 단일프레임 검출기 + 칼만 필터 (Phase 2b, 개발 중)
3. **IN/OUT 판정**: 볼 궤적 + 코트 좌표계 교차점 (미구현)

---

## 현재 상태 요약 (2026-03-21)

### Phase 2b: 경량 볼 검출기 + 칼만 필터 (진행 중)

기존 TrackNet (3-frame, 148ms, 6fps)을 대체하여 **단일프레임 경량 검출기 (15-25ms) + 칼만 필터로 30-50fps** 목표.

#### 완료된 것
- ✅ **모델 설계** (`ml/src/model_ball.py`) - MobileNetV3-Small + ConvTranspose2d decoder
  - Input: [1, 3, 192, 192] → Output: [1, 1, 48, 48] heatmap
  - 파라미터: 2,271,201 (backbone 927K + decoder 1,344K)
  - Focal loss (alpha=0.97, gamma=2.0), ImageNet normalization
- ✅ **Kaggle 학습 노트북 v2** (`ml/notebooks/BallDetector_Training.ipynb`)
  - Dropout 0.15, AdamW (weight_decay=1e-4), 강화된 augmentation
  - **val_pixel_error 기준 best model 저장** (v1은 val_loss 기준이라 epoch 0 저장됨)
  - TrackNet Dataset-001 (19,835 frames) 사용
- ✅ **v1 학습 완료 (결과)**:
  ```
  Ep 0:  VlErr=9.3px (frozen)     ← val_loss 기준 best (epoch 0) 저장됨
  Ep 5:  VlErr=9.1px (UNFROZEN)
  Ep 13: VlErr=7.8px              ← 실제 best pixel error (저장 안됨!)
  Ep 15: Early stopping
  ```
  - 문제: val_loss 기준 저장이라 epoch 0 모델만 저장됨, Kaggle 세션 만료로 소실
  - → **v2 노트북에서 재학습 필요** (val_pixel_error 기준 저장으로 수정 완료)
- ✅ **칼만 필터** (`court-detection/.../BallKalmanFilter.kt`)
  - 6-state [x, y, vx, vy, ax, ay], diagonal covariance
  - gating 150px, max 3 predict frames
- ✅ **Android 통합 코드** (아직 모델 없이 코드만 준비)
  - `BallTrackingDetector.kt` - 단일프레임 192×192 입력으로 변경
  - `CameraViewModel.kt` - 칼만 필터 기반 볼 추적으로 변경
- ✅ **ONNX export 스크립트** (`ml/src/export_ball_onnx.py`)
- ✅ **핸드폰 촬영 영상 다운로드** (18개 영상, 87,601 프레임 추출)
- ✅ **라벨링 도구** (`ml/src/label_ball.py`) - 브라우저 기반, N키로 Not Visible 자동저장
- ✅ **라벨링용 프레임 서브샘플링** (3,600 프레임, 영상당 200장)

#### 미완료
- ❌ **v2 모델 학습** - Kaggle에서 새 노트북으로 재학습 필요
- ❌ **ONNX 모델 파일** - 학습 완료 후 `court-detection/src/main/assets/ball_detector.onnx`에 배치
- ❌ **핸드폰 영상 볼 라벨링** - 3,600 프레임 라벨링 대기 중
- ❌ **Android 빌드 + 실기기 테스트**
- ❌ **INT8 양자화**

---

## 즉시 해야 할 작업 (우선순위 순)

### 1. Kaggle v2 모델 재학습 ⭐ 최우선

1. Kaggle에서 **새 노트북** 생성
2. GPU P100 활성화
3. TrackNet 데이터셋 추가 (Add Data → `tracknet-tennis-ball-detection`)
4. `ml/notebooks/BallDetector_Training.ipynb` 셀 내용을 복사하여 순서대로 실행
5. **주의**: P100은 CUDA compute 6.0이므로 PyTorch 호환 확인
   ```
   !pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
   !pip install "numpy<2"
   ```
6. 학습 완료 후 ONNX export 셀 실행 → `ball_detector.onnx` 다운로드
7. 다운로드한 파일을 `court-detection/src/main/assets/ball_detector.onnx`에 복사

### 2. 핸드폰 영상 볼 라벨링

```bash
cd C:\Users\kokor\Desktop\SHOT\ml
python src/label_ball.py --frames data/phone_ball/frames_sampled --port 8081
```
- 브라우저에서 http://localhost:8081 접속
- 조작: 클릭(볼 위치), N(안보임→자동저장), Enter(저장+다음), A/D(이전/다음), G/H(영상그룹 이동)
- 결과: `data/phone_ball/ball_annotations.json` 자동 저장
- 라벨링 후 도메인 갭 줄이기 위한 fine-tuning에 사용

### 3. Android 빌드 + 실기기 테스트

ONNX 모델 배치 후:
```bash
cd "C:\Users\kokor\Desktop\SHOT"
JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" \
  "./gradlew" :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.shot.app/.MainActivity
```

---

## 파일 구조 (Phase 2b 관련)

### 새로 생성된 파일

| 파일 | 역할 | 상태 |
|------|------|------|
| `ml/src/model_ball.py` | BallDetector 모델 정의 (MobileNetV3+decoder) | 완료 |
| `ml/src/train_ball.py` | 학습 스크립트 (standalone) | 완료 |
| `ml/src/export_ball_onnx.py` | ONNX 변환 + 검증 | 완료 |
| `ml/src/download_ball_videos.py` | 22개 YouTube 영상 다운로드 + 프레임 추출 | 완료 |
| `ml/notebooks/BallDetector_Training.ipynb` | **Kaggle 학습 노트북 v2** (인라인, 파일 import 불필요) | 완료 |
| `court-detection/.../BallKalmanFilter.kt` | 6-state 칼만 필터 | 완료 |
| `docs/superpowers/specs/2026-03-21-phase2b-ball-detector-design.md` | Phase 2b 설계 스펙 | 완료 |

### 수정된 파일

| 파일 | 변경 내용 |
|------|-----------|
| `court-detection/.../BallTrackingDetector.kt` | 3-frame→1-frame, 9ch→3ch, 128×320→192×192, ImageNet normalization 추가 |
| `app/.../CameraViewModel.kt` | BallKalmanFilter import, updateBallWithKalman() 추가, unlockCourt()에서 kalman reset |

### 데이터 (git에 포함 안됨)

```
ml/data/phone_ball/
├── videos/           # 18개 영상 (01-22, 일부 누락)
├── frames/           # 87,601 프레임 (5fps 추출)
├── frames_sampled/   # 3,600 프레임 (라벨링용, 영상당 200장)
└── ball_annotations.json  # 라벨링 결과 (생성 예정)
```

---

## 기술 상세

### BallDetector 아키텍처
```
Input [1, 3, 192, 192]
  ↓ MobileNetV3-Small features (pretrained, 927K params)
  ↓ [1, 576, 6, 6]
  ↓ ConvTranspose2d 576→128 (stride 2) + BN + ReLU + Dropout(0.15)
  ↓ ConvTranspose2d 128→64  (stride 2) + BN + ReLU + Dropout(0.15)
  ↓ ConvTranspose2d 64→32   (stride 2) + BN + ReLU
  ↓ Conv2d 32→1 (1×1)
  ↓ Sigmoid
Output [1, 1, 48, 48] heatmap
```

### 학습 전략 (v2)
- **Freeze backbone** 5 epochs → **Unfreeze**
- **Separate LR**: backbone 1e-4, decoder 1e-3
- **Optimizer**: AdamW (weight_decay=1e-4)
- **Loss**: Focal loss (alpha=0.97, gamma=2.0)
- **Augmentation**: horizontal flip, Gaussian blur, brightness, color shift, noise, contrast
- **Early stopping**: patience 20, **val_pixel_error 기준** (v1은 val_loss 기준이었음)
- **Best model 이중 저장**: `ball_best.pth` (pixel error), `ball_best_loss.pth` (val_loss)
- **Heatmap**: 48×48, Gaussian sigma=2.5, argmax extraction

### 칼만 필터 (BallKalmanFilter.kt)
```
State: [x, y, vx, vy, ax, ay]
PROCESS_NOISE_POS = 5.0
MEASUREMENT_NOISE = 8.0
GATE_DISTANCE = 150px
MAX_PREDICT_FRAMES = 3
```
- detect 성공 → update(measurement) → 보정된 위치
- detect 실패 → predict() → 물리 기반 예측 (최대 3프레임)
- 3프레임 초과 미검출 → lost 상태

### 핸드폰 촬영 영상 (22개 URL)

메모리 파일 참조: `~/.claude/projects/.../memory/reference_ball_training_videos.md`

다운로드된 영상: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 12, 14, 15, 16, 18, 19, 21, 22 (18개)
누락된 영상: 11, 13, 17, 20 (30분 초과 duration filter)

---

## 이전 Phase 완료 사항

### Phase 1: 코트 인식 (완료)
- MobileNetV3-Small 히트맵 모델, 0.68px 오차
- 961장 라벨링 데이터 (YouTube 339 + SNTC 622)
- Android 앱: 카메라 프리뷰 + 코트 라인 오버레이 + 키포인트 시각화
- ONNX Runtime 배포 (TFLite 변환 실패로 전환)

### Phase 2a: TrackNet v1 (완료 → 폐기)
- 3-frame TrackNet, 148ms 추론, 6fps → **모바일 부적합**
- 코트 잠금 모드, 스크린샷 캡처, 네온 코트 오버레이 등 UI 완성

---

## 환경 설정

### Android 빌드 환경
- **Gradle**: 8.9, **AGP**: 8.7.3, **Kotlin**: 2.0.21
- **compileSdk**: 35, **minSdk**: 26, **targetSdk**: 35
- **Java**: 17 (Android Studio JBR)
- `gradle.properties`: `android.overridePathCheck=true` (한글 경로 대응)

### Python ML 환경
```bash
cd ml
python -m uv venv .venv312 --python 3.12
python -m uv pip install --python .venv312/Scripts/python.exe -r requirements.txt
```

### Kaggle 학습 환경 (P100 호환)
```
pip install torch==2.2.0+cu118 torchvision==0.17.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2"
```
P100 = CUDA compute 6.0 (sm_60), PyTorch 2.10+ (cu128)는 호환 안됨.

---

## Git 히스토리

| 커밋 | 내용 |
|------|------|
| `2f25931` | Phase 1 초기 구현 |
| `4b9fed9` | TFLite 모델 추가 |
| `4eabb29` | Phase 1 인수인계 |
| `62e58fc` | 키포인트 라벨링 도구 |
| `74bdb09` | YouTube 데이터 수집 |
| `dbf8ad2` | 비교 학습 스크립트 |
| `75e12bb` | 학습 셋업 + 데스크탑 전환 |
| `d87bb81` | 히트맵 모델 + 3단계 학습 |
| `ec065a7` | 히트맵 3-stage v2 (0.68px) |
| `0fa22a1` | Android 통합 |
| (이전) | Phase 2a TrackNet + UI완성 |
| **(현재)** | **Phase 2b 경량 볼 검출기 + 칼만 필터 + v2 학습 노트북** |

**원격 저장소**: https://github.com/kokoro456/SHOT-AI

---

## 오답노트 (Phase 2b)

1. **val_loss ≠ 실제 성능**: val_loss 기준 best model(epoch 0, 9.3px)보다 val_pixel_error 기준(epoch 13, 7.8px)이 실제로 더 좋음 → v2에서 이중 저장으로 해결
2. **과적합**: backbone unfreeze 후 val_loss 증가하지만 pixel error는 개선됨 → Dropout + weight decay 추가
3. **Kaggle 세션 만료**: 학습 결과가 세션 종료 시 소실됨 → ONNX export까지 한 세션에서 완료해야 함
4. **yt-dlp 호환성**: `--js-runtimes nodejs`는 무효, `node`가 올바름. `--quiet` 플래그가 에러 숨김
5. **Kaggle P100**: CUDA compute 6.0, PyTorch 2.2.0+cu118까지만 호환
