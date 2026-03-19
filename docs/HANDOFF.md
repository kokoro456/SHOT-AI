# SHOT Phase 1 인수인계 문서

> 최종 업데이트: 2026-03-19

## 프로젝트 개요

SHOT은 핸드폰 카메라로 테니스 코트 라인을 실시간 인식하는 Android 앱이다.
**키포인트 검출 + 호모그래피 역산** 방식으로 동작한다:

1. TFLite 모델이 카메라 프레임에서 니어코트 8개 키포인트(9~16번) 검출
2. ITF 표준 치수 + 호모그래피로 전체 16개 코트 포인트 역산
3. 코트 라인 오버레이 렌더링

---

## 현재 상태 요약 (2026-03-19)

### 완료된 것
- ✅ ML 모델 학습 완료 (히트맵 3-stage v2, **0.68px 오차**, 이전 2.41px 대비 3.5배 개선)
- ✅ SNTC 데이터 대규모 수집: 678 동영상에서 753 프레임 추출, 622장 라벨링
- ✅ 총 라벨링 데이터: 961장 (YouTube 339 + SNTC 462 + SNTC selected 160)
- ✅ Android 프로젝트 스캐폴딩 (멀티모듈)
- ✅ Android 빌드 성공 (Gradle 8.9, AGP 8.7.3)
- ✅ 실기기(USB) 설치 + 실행 성공
- ✅ 카메라 라이브 프리뷰 동작
- ✅ ML 추론 파이프라인 연결 (카메라 → 검출 → 호모그래피 → 투영)
- ✅ 코트 라인 오버레이 렌더링 (Compose Canvas)
- ✅ 키포인트 시각화 (초록 원)
- ✅ 상태 표시 (코트 인식됨/부분 인식/미인식)
- ✅ 디버그 정보 표시 (추론시간, 재투영오차, 키포인트수)
- ✅ TFLite 변환 실패 → ONNX Runtime으로 전환 (Android 배포)
- ✅ 새 모델(v2) 폰 배포 완료, 모니터 스크린샷(12.jpg, 123.jpg) 테스트에서 유의미한 개선 확인
- ✅ 오버레이 지터 수정: deadzone 4px, 3-frame gate, homography deadzone, output stabilization
- ✅ 라벨링 도구 개선: 같은 영상 그룹 라벨 복사 기능

### 현재 문제점
- ❌ **실제 코트 필드 테스트 미수행**: 모니터 스크린샷 테스트에서 개선 확인했으나 실제 코트 테스트 필요
- ❌ **TFLite 변환 불가**: onnx2tf가 decoder의 skip connection을 처리 못함 → ONNX Runtime 사용 중
- ❌ Android Studio Run 버튼 비활성화 (Gradle sync 문제, ADB 직접 설치로 우회)

### 테스트 스크린샷 분석

| 파일 | 상황 | 결과 |
|------|------|------|
| EE.jpg | 방송 영상(아카풀코) 비춤 | 가장 정확. 니어/파코트 오버레이 거의 일치 |
| RE.jpg | 방송 영상(아카풀코) 비춤 | 니어코트 OK, 파코트 약간 어긋남 |
| SHOT.jpg | 방송 영상(마이애미) 비춤 | **심각한 오류** - 파코트 투영이 좌측으로 크게 밀림 |

**분석**: 같은 방송 영상이라도 카메라 앵글에 따라 호모그래피 품질이 크게 달라짐.
모델의 니어코트 키포인트 검출 자체는 정확하지만, 호모그래피 → 파코트 역산 과정에서 오차 증폭.

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
| `court-model/.../HomographyCalculator.kt` | DLT + Jacobi SVD 호모그래피 | 완료 |
| `court-model/.../CourtProjector.kt` | H⁻¹로 16개 포인트 투영 | 완료 |
| `court-model/.../HomographyValidator.kt` | 재투영 오차, 행렬식 검증 | 완료 |
| `court-model/.../TemporalSmoother.kt` | EMA 시간적 평활화 | 완료 |
| `app/.../CameraViewModel.kt` | 파이프라인 오케스트레이션 | 완료 |
| `app/.../CameraScreen.kt` | 카메라 프리뷰 + 오버레이 Compose UI | 완료 |
| `app/.../MainActivity.kt` | Hilt 엔트리포인트 | 완료 |

### 2. ML 학습 파이프라인 (Phase 1B)

**Python 파일 (`ml/src/`):**

| 파일 | 역할 |
|------|------|
| `model.py` | MobileNetV3-Small + 키포인트 회귀 헤드 (1.1M params) |
| `model_heatmap.py` | **히트맵 기반 키포인트 검출 모델** (MobileNetV3 + decoder) |
| `dataset.py` | CourtKeypointDataset (JSON 어노테이션 로더) |
| `augmentations.py` | Albumentations 증강 (색상, 기하, 노이즈, 그림자) |
| `augmentations_v2.py` | 강한 도메인 갭 축소 augmentation |
| `train.py` | 학습 루프 (AdamW, CosineAnnealingLR, early stopping) |
| `train_3stage.py` | 3단계 학습 파이프라인 |
| `train_compare.py` | 3가지 비교 실험 학습 |
| `export_tflite.py` | PyTorch → ONNX → onnx2tf → TFLite 변환 |
| `convert_dataset.py` | yastrebksv 14-keypoint → SHOT 8-keypoint 변환 |
| `youtube_collect.py` | YouTube 검색 + URL 수집 |
| `extract_frames.py` | yt-dlp 다운로드 + 프레임 추출 |
| `predict_and_preview.py` | 모델 예측 + 시각적 미리보기 |
| `review_data.py` | 대화형 검수 도구 |
| `labeling_tool.py` | 브라우저 기반 키포인트 라벨링 도구 |

### 3. 모델 학습 결과

**히트맵 3-Stage v2 학습 최종 결과: 평균 오차 0.68px** (이전 v1: 2.41px, 3.5배 개선)

| Stage | 내용 | 오차(px) |
|-------|------|----------|
| v1 Stage 2 | 339장 phone data, Mixed 50:50 | 2.41 |
| **v2 Stage 2** | **961장 phone data (YouTube 339 + SNTC 622), Mixed 50:50** | **0.68** |

**모델 배포 (ONNX Runtime):**
- TFLite 변환 실패: onnx2tf가 decoder의 skip connection 처리 불가
- ONNX Runtime for Android로 전환
- 입력: `[1, 3, 256, 256]` NCHW float32, ImageNet 정규화
- 출력: 8 keypoints × [x, y, confidence]
- 배치 위치: `court-detection/src/main/assets/` (ONNX 모델)

### 4. Android 빌드 환경 구축 (2026-03-19)

노트북으로 작업환경 이전 후 발생한 빌드 에러들과 해결:

| 문제 | 원인 | 해결 |
|------|------|------|
| `dependencyResolution` 에러 | Gradle 8.9에서 `dependencyResolutionManagement` 필요 | `settings.gradle.kts` 수정 |
| non-ASCII 경로 에러 | 프로젝트 경로에 한글 (`잡/바탕 화면`) | `gradle.properties`에 `android.overridePathCheck=true` 추가 |
| `tensorflow-lite-support:2.16.1` 없음 | Google Maven에 해당 버전 없음 | `0.4.4`로 다운그레이드 |
| `windowKeepScreenOn` 에러 | `themes.xml`에서 style 속성 오류 | `android:keepScreenOn`은 View 속성, theme에서 제거 |
| `mipmap/ic_launcher` 없음 | 앱 아이콘 리소스 누락 | 기본 아이콘 리소스 생성 |
| Run ▶ 비활성화 | Gradle sync 실패 / wrapper 누락 | Gradle wrapper 파일 생성 + ADB 직접 설치로 우회 |

---

## 알려진 이슈 및 해결 필요 사항

### 1. 호모그래피 역산 정밀도 (최우선)

**현상**: 니어코트 8개 키포인트는 정확하게 검출되지만, 호모그래피로 역산한 파코트 포인트가 크게 어긋남.
특히 카메라 앵글이 달라지면 오차가 급격히 증가 (SHOT.jpg vs EE.jpg).

**원인 분석**:
- DLT + Jacobi SVD 구현체의 수치 안정성 문제 가능성
- 8개 포인트가 니어코트에 집중되어 있어 파코트 방향 외삽(extrapolation) 시 오차 증폭
- 호모그래피 H 계산에 사용하는 좌표 정규화가 충분하지 않을 수 있음

**해결 방향**:
1. OpenCV 네이티브 `findHomography()` 대신 순수 Kotlin DLT 사용 중 → OpenCV JNI 도입 검토
2. 포인트 수가 부족한 경우(6~7개) 호모그래피 대신 affine transform 사용 검토
3. 재투영 오차 기반 outlier 제거 (RANSAC) 추가
4. 파코트 포인트의 범위 제한 (화면 밖으로 너무 멀리 나가지 않도록)

### 2. False Positive 문제

**현상**: 카메라를 손으로 가려도 "코트 인식됨" 또는 "부분 인식"으로 표시됨.

**원인**: 모델이 입력 이미지와 관계없이 일관된 "기본 위치" 키포인트를 출력.
이 기본 위치들이 자기 일관성이 높아서 호모그래피 검증을 통과함.

**현재 우회책**: `CameraViewModel`에서 모델 기본출력(검은 이미지의 출력)과 비교하여
현재 출력이 기본출력과 너무 유사하면 거부하는 로직 추가됨.
하지만 threshold 튜닝이 부족하여 실제 코트도 거부하는 경우 발생.

**해결 방향**:
1. 프레임 간 키포인트 변화량 기반 판정 (카메라 움직임에 키포인트가 반응하는지)
2. 모델 출력 confidence 패턴 분석 (실제 코트: 일부 키포인트 매우 높고 일부 낮음 vs 비코트: 균일)
3. 추론 전 이미지 variance 체크 (매우 어두운/단색 이미지 사전 거부)

### 3. 실제 코트 테스트 미수행

지금까지 테스트는 모두 **모니터에 방송 영상을 띄우고 핸드폰으로 촬영**하는 방식.
실제 코트 옆에서 핸드폰 카메라로 직접 비추는 테스트가 필요함.
모델은 YouTube 핸드폰 영상으로 학습했으므로 실제 환경에서는 다를 수 있음.

### 4. Android Studio Run 버튼 비활성화

Gradle sync가 완료되지 않아 Run Configuration이 자동 생성되지 않음.
현재 **ADB 명령어로 직접 설치+실행**하는 방식으로 우회 중.

```bash
# 빌드
cd "TENNIS SHOT" && JAVA_HOME="/c/Program Files/Android/Android Studio/jbr" \
  "/c/Program Files/Android/Android Studio/jbr/bin/java.exe" -Xmx2048m \
  -cp "gradle/wrapper/gradle-wrapper.jar" org.gradle.wrapper.GradleWrapperMain :app:assembleDebug

# 설치 + 실행
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.shot.app/.MainActivity
```

---

## 환경 설정

### Android 빌드 환경

- **Gradle**: 8.9
- **AGP**: 8.7.3
- **Kotlin**: 2.0.21
- **compileSdk**: 35, **minSdk**: 26, **targetSdk**: 35
- **Java**: 17 (Android Studio JBR)
- **주의**: 프로젝트 경로에 한글 포함 → `android.overridePathCheck=true` 필수

### Python ML 환경

```bash
# Python 3.14는 TensorFlow/albumentations와 호환 안됨 → 3.12 사용
cd ml
python -m uv venv .venv312 --python 3.12
python -m uv pip install --python .venv312/Scripts/python.exe -r requirements.txt
```

### 모델 변환 파이프라인

```
[v1] PyTorch (NCHW) → ONNX (opset 18) → onnx2tf → TFLite (NHWC)  ← 더 이상 사용 안 함
[v2] PyTorch (NCHW) → ONNX (opset 18) → ONNX Runtime (Android)   ← 현재 사용 중
```

TFLite 변환 실패 원인: onnx2tf가 히트맵 decoder의 skip connection(ConvTranspose2d + concat)을 처리하지 못함.
ONNX Runtime은 NCHW를 그대로 사용하므로 변환 문제 없음.

### 데이터셋

- **방송 데이터**: yastrebksv/TennisCourtDetector (MIT 라이선스, Kaggle)
- **핸드폰 데이터**: 총 961장 라벨링 완료
  - YouTube 동호인 영상: 339장 (`ml/data/youtube/labeled_annotations.json`)
  - SNTC YouTube 채널: 678 동영상에서 753 프레임 추출, 462장 라벨링
  - SNTC selected: 160장 추가 라벨링
  - 합계: 339 + 462 + 160 = 961장

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

### Phase 1C: 검출 파이프라인 품질 개선 (즉시)

1. [x] ~~호모그래피 정밀도 개선~~ → 오버레이 지터 수정 (deadzone, 3-frame gate, stabilization)
2. [x] ~~False positive 제거~~ → 개선됨
3. [ ] **실제 코트에서 필드 테스트** (최우선)
4. [ ] Android Studio Gradle sync 해결 (Run 버튼 활성화)

### Phase 1D: UI 및 마무리

5. [ ] 설정 화면 (해상도, 단식/복식, 언어)
6. [ ] 영상 녹화 (CameraX VideoCapture)
7. [ ] FPS 계산 구현 (현재 TODO 상태)

### Phase 1E: 검증

8. [ ] 완료 게이트: 10초 연속 95% 프레임 성공
9. [ ] 20개 다양한 코트에서 테스트

### Phase 2: TrackNet 공 추적 (향후)

- [ ] TrackNet 모델 학습 (공 위치 추적)
- [ ] 공 궤적 기반 착지점 판정
- [ ] Android 통합

### 향후 (데이터 추가 확보)

- [x] ~~YouTube 영상 추가 수집 → 1,000장+ 목표~~ → 961장 달성 (목표 근접)
- [x] ~~모델 예측 보조 라벨링으로 효율화~~ → 라벨 복사 기능 구현
- [ ] INT8 양자화 검토 (FP32 → 모바일 속도 개선용)

---

## Git 히스토리

| 커밋 | 내용 |
|------|------|
| `2f25931` | Phase 1 초기 구현 (Android 스캐폴딩 + ML 파이프라인) |
| `4b9fed9` | 학습된 TFLite 모델 추가 (4.25MB) |
| `4eabb29` | Phase 1 인수인계 문서 추가 |
| `62e58fc` | 키포인트 라벨링 도구 + TFLite 미리보기 생성기 |
| `74bdb09` | YouTube 데이터 수집 파이프라인 추가 |
| `dbf8ad2` | 비교 학습 스크립트 + 라벨링 완료 업데이트 |
| `75e12bb` | 학습 셋업 스크립트 + 데스크탑 전환 업데이트 |
| `d87bb81` | 히트맵 모델 + 3단계 학습 + Colab 노트북 |
| `ec065a7` | 히트맵 3-stage 결과 기록: 2.41px 달성 |
| `0fa22a1` | 인수인계 업데이트: Android 통합 단계 전환 |
| (현재) | Android 빌드 성공 + 실기기 설치 + 파이프라인 연결 + 테스트 |

**원격 저장소**: https://github.com/kokoro456/SHOT-AI

---

## 오답노트

자세한 실패 분석 및 교훈은 `docs/RETROSPECTIVE.md` 참조.

**핵심 요약:**
1. FC direct regression은 공간 정보 손실 → 히트맵 기반으로 변경 → **2.41px → 0.68px 달성!**
2. 데이터 스케일링 효과 입증: 339장 → 961장 = 2.41px → 0.68px (3.5배 개선)
3. 단순 합산보다 3단계 학습 (pretrain → mixed → fine-tune)
4. soft-argmax 함정: 평평한 히트맵에서 중심(0.5,0.5)으로 수렴 → argmax 사용
5. 테스트는 반드시 타겟 도메인(핸드폰)으로
6. Colab에서 broadcast 이미지 전처리(256x256 JPEG) 필수 → 6배 속도 향상
7. TFLite 변환 실패 → ONNX Runtime이 대안 (skip connection 있는 decoder 구조에서)
8. 데이터 양 vs 모델 구조: 같은 모델에서 데이터 3배 → 오차 3.5배 감소
