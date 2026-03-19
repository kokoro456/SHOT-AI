# Phase 1a: TrackNet 공 추적 모델 구현 계획

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 삼각대 고정 촬영 영상에서 테니스 공을 실시간 추적하는 TrackNet 모델을 학습하고 ONNX로 변환하여 Android 앱에 통합할 준비를 완료한다.

**Architecture:** TrackNet v2 기반 encoder-decoder (연속 3프레임 입력 → 공 위치 히트맵 출력). 기존 코드(`model_tracknet.py`, `train_tracknet.py`)를 기반으로 데이터 수집→라벨링→학습→검증→ONNX 변환 파이프라인을 완성한다.

**Tech Stack:** PyTorch, ONNX Runtime, Google Colab (학습), Python (데이터 처리), 기존 `label_ball.py` (라벨링 도구)

**입력 해상도:** 128×320 (스펙 문서의 360×640 대비 축소). 이유: (1) 모바일 추론 속도 25ms 목표 달성을 위해 4배 축소 필요, (2) 테니스 공은 히트맵에서 가우시안 피크로 표현되므로 고해상도가 필수적이지 않음, (3) 기존 `model_tracknet.py`가 128×320으로 설계됨. 85% 검출률 미달 시 해상도를 192×480 또는 360×640으로 단계적 상향.

**목표 미달 시 대응 전략:**
1. 85% 미달 → 데이터 추가 수집 (특히 미검출 환경 집중)
2. 여전히 미달 → 해상도 192×480으로 상향 + 재학습
3. 여전히 미달 → base_filters 64로 증가 (모델 크기 ~12MB)
4. 최종 수단 → 360×640 원본 해상도 사용 (추론 속도 타협)

---

## 파일 구조

### 기존 파일 (수정)
| 파일 | 역할 | 수정 내용 |
|------|------|----------|
| `ml/src/train_tracknet.py` | TrackNet 학습 스크립트 | 어그멘테이션 강화, 비디오별 split, early stopping 추가 |
| `ml/src/model_tracknet.py` | TrackNet 모델 아키텍처 | 변경 없음 (현재 구조 유지) |
| `ml/src/label_ball.py` | 공 라벨링 도구 | 비디오 그룹 네비게이션 + 이전 프레임 공 위치 가이드 추가 |
| `ml/src/export_tflite.py` | 코트 모델 ONNX 변환 | 변경 없음 (TrackNet은 별도 파일 사용) |

### 신규 파일 (생성)
| 파일 | 역할 |
|------|------|
| `ml/src/extract_ball_frames.py` | 촬영 영상에서 30fps 프레임 추출 + 비디오 ID 기반 파일명 생성 |
| `ml/src/review_ball_data.py` | 라벨링 데이터 검수 도구 (랜덤 샘플 시각화 + 통계) |
| `ml/src/augmentations_ball.py` | 공 추적 전용 어그멘테이션 파이프라인 |
| `ml/src/export_tracknet_onnx.py` | TrackNet → ONNX 변환 전용 스크립트 |
| `ml/notebooks/TrackNet_Training.ipynb` | Colab 학습 노트북 |

---

## Task 1: 학습 데이터 수집용 프레임 추출 도구

**Files:**
- Create: `ml/src/extract_ball_frames.py`

- [ ] **Step 1: 프레임 추출 스크립트 작성**

```python
"""
촬영 영상에서 TrackNet 학습용 프레임을 추출한다.

사용법:
    python extract_ball_frames.py --video path/to/video.mp4 --output data/ball/frames --fps 30
    python extract_ball_frames.py --video-dir path/to/videos --output data/ball/frames --fps 30
"""

import argparse
import os
from pathlib import Path
import cv2


def extract_frames(video_path: str, output_dir: str, fps: int = 30):
    """단일 영상에서 프레임을 추출한다."""
    video_path = Path(video_path)
    vid_id = video_path.stem  # 파일명을 비디오 ID로 사용

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, round(src_fps / fps))

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            filename = f"{vid_id}_frame_{count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            count += 1
        frame_idx += 1

    cap.release()
    print(f"  {vid_id}: {count} frames extracted (source: {src_fps:.0f}fps → {fps}fps)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract frames for ball labeling")
    parser.add_argument("--video", type=str, help="Single video file")
    parser.add_argument("--video-dir", type=str, help="Directory of videos")
    parser.add_argument("--output", required=True, help="Output frames directory")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    total = 0
    if args.video:
        total = extract_frames(args.video, args.output, args.fps)
    elif args.video_dir:
        exts = {".mp4", ".mov", ".avi", ".mkv"}
        videos = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix.lower() in exts)
        print(f"Found {len(videos)} videos")
        for v in videos:
            total += extract_frames(str(v), args.output, args.fps)
    else:
        print("ERROR: --video or --video-dir required")
        return

    print(f"\nTotal: {total} frames → {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 테스트 실행**

Run: `cd ml/src && python extract_ball_frames.py --video ../data/sntc/tmp/sample.mp4 --output ../data/ball_test/frames --fps 30`
Expected: 프레임이 `{vid_id}_frame_XXXX.jpg` 형식으로 추출됨

- [ ] **Step 3: Commit**

```bash
git add ml/src/extract_ball_frames.py
git commit -m "feat: add frame extraction tool for TrackNet training data"
```

---

## Task 2: 라벨링 도구 개선

**Files:**
- Modify: `ml/src/label_ball.py`

현재 `label_ball.py`는 기능이 충분하지만 대량 라벨링 효율을 위해 2가지 개선이 필요하다:
1. 비디오별 그룹 네비게이션 (같은 비디오 프레임 간 빠른 이동)
2. 이전 프레임의 공 위치를 현재 프레임에 가이드 점으로 표시 (공이 조금씩 이동하므로)

- [ ] **Step 1: 비디오 그룹 네비게이션 추가**

JavaScript `init()` 함수에 비디오별 그룹 인덱스를 생성하고, `G`/`H` 키로 이전/다음 비디오 그룹으로 점프하는 기능을 추가한다.

```javascript
// init() 안에 추가:
// Build video group index
let videoGroups = {};
images.forEach((name, idx) => {
    const match = name.match(/^(.+?)_frame_/);
    const vid = match ? match[1] : 'unknown';
    if (!videoGroups[vid]) videoGroups[vid] = [];
    videoGroups[vid].push(idx);
});
let videoIds = Object.keys(videoGroups);
```

키보드 핸들러에 추가:
```javascript
case 'g': case 'G':  // prev video group
    jumpToVideoGroup(-1);
    break;
case 'h': case 'H':  // next video group
    jumpToVideoGroup(1);
    break;
```

```javascript
function jumpToVideoGroup(direction) {
    const currentName = images[currentIdx];
    const match = currentName.match(/^(.+?)_frame_/);
    const currentVid = match ? match[1] : 'unknown';
    const vidIdx = videoIds.indexOf(currentVid);
    const nextVidIdx = Math.max(0, Math.min(videoIds.length - 1, vidIdx + direction));
    if (nextVidIdx !== vidIdx) {
        currentIdx = videoGroups[videoIds[nextVidIdx]][0];
        loadFrame();
    }
}
```

- [ ] **Step 2: 이전 프레임 공 위치 가이드 표시**

`redraw()` 함수에서 이전 프레임에 라벨링된 공 위치를 반투명 원으로 표시한다.

```javascript
// redraw() 시작 부분, 현재 프레임 그린 뒤:
if (currentIdx > 0) {
    const prevName = images[currentIdx - 1];
    const prevBall = annotations[prevName];
    if (prevBall && prevBall.visibility > 0 && prevBall.x >= 0) {
        const px = prevBall.x * canvas.width;
        const py = prevBall.y * canvas.height;
        ctx.strokeStyle = 'rgba(100, 100, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = 'rgba(100, 100, 255, 0.3)';
        ctx.font = '10px monospace';
        ctx.fillText('prev', px + 14, py - 2);
    }
}
```

- [ ] **Step 3: instructions 영역에 새 단축키 설명 추가**

```html
<strong>G/H</strong> = prev/next video group<br>
<em>파란 점선 = 이전 프레임 공 위치</em>
```

- [ ] **Step 4: 테스트 실행**

Run: `cd ml/src && python label_ball.py --frames ../data/sntc/frames --port 8081`
Expected: 브라우저에서 G/H로 비디오 그룹 이동, 이전 프레임 공 위치가 파란 점선으로 표시

- [ ] **Step 5: Commit**

```bash
git add ml/src/label_ball.py
git commit -m "feat: add video group navigation and prev-frame guide to ball labeling tool"
```

---

## Task 3: 데이터 수집 (촬영 + 프레임 추출)

**Files:**
- 해당 없음 (데이터 작업)

이 Task는 코드 작성이 아닌 데이터 수집 작업이다. 최소 10개 이상의 영상에서 5,000+ 프레임을 확보해야 한다.

- [ ] **Step 1: 데이터 소스 목록 정리**

필요한 데이터 다양성:
| 환경 | 최소 영상 수 | 비고 |
|------|------------|------|
| 실외 하드코트 주간 | 3 | 가장 일반적 |
| 실내 하드코트 | 2 | 조명 다양성 |
| 실외 클레이 | 1 | 코트 색상 다양성 |
| 야간 조명 | 2 | 공 보임 조건 어려움 |
| 방송 경기 | 2+ | 고화질, 보충용 |

촬영 조건:
- 삼각대 고정, 베이스라인 뒤 중앙 1~2m
- 30fps 이상
- 최소 2분 이상 랠리/서브 포함

- [ ] **Step 2: 기존 SNTC 데이터 활용 가능성 확인**

기존 데이터: `ml/data/sntc/frames/` (2,032 프레임, ~166MB)
기존 어노테이션: `ml/data/sntc/annotations.json` (10,388 entries) + `selected_annotations.json` (3,600 entries)

Run: `cd ml && python -c "import json; d=json.load(open('data/sntc/selected_annotations.json')); print(f'Selected: {len(d)}, visible: {sum(1 for x in d if x.get(\"visibility\",0)>0)}')" `

기존 데이터 형식이 `label_ball.py` 출력 형식(`{image, x, y, visibility}`)과 호환되는지 확인.

- [ ] **Step 3: 신규 영상 촬영 + 프레임 추출**

```bash
# 촬영한 영상들을 모아서 프레임 추출
python extract_ball_frames.py --video-dir ../data/ball/raw_videos --output ../data/ball/frames --fps 30
```

- [ ] **Step 4: 전체 데이터 통계 확인**

목표: 총 5,000+ 프레임 (기존 SNTC 3,600 + 신규 1,400+)
- 영상별 프레임 수 확인
- 환경별 비율 확인

---

## Task 4: 공 위치 라벨링

**Files:**
- 해당 없음 (데이터 작업, `label_ball.py` 사용)

- [ ] **Step 1: 기존 SNTC 데이터로 라벨링 시작**

```bash
cd ml/src
python label_ball.py --frames ../data/sntc/frames --output ../data/sntc/ball_annotations.json --port 8081
```

기존 `selected_annotations.json`이 있으므로 이를 `ball_annotations.json`으로 복사하여 시작점으로 사용:
```bash
cp ../data/sntc/selected_annotations.json ../data/sntc/ball_annotations.json
```

- [ ] **Step 2: 신규 데이터 라벨링**

```bash
python label_ball.py --frames ../data/ball/frames --output ../data/ball/ball_annotations.json --port 8081
```

- [ ] **Step 3: 프레임 통합 + 라벨링 데이터 통합**

신규 프레임과 SNTC 프레임을 하나의 디렉토리로 통합한다 (학습 스크립트가 단일 `--frames` 디렉토리를 요구하므로).

```bash
# 통합 프레임 디렉토리 생성 (심볼릭 링크 또는 복사)
mkdir -p ../data/ball_all/frames
cp ../data/sntc/frames/*.jpg ../data/ball_all/frames/
cp ../data/ball/frames/*.jpg ../data/ball_all/frames/
```

```python
# 어노테이션 병합 스크립트 (일회성)
import json
all_ann = []
for f in ["../data/sntc/ball_annotations.json", "../data/ball/ball_annotations.json"]:
    with open(f) as fp:
        all_ann.extend(json.load(fp))
with open("../data/ball_all/ball_combined.json", "w") as fp:
    json.dump(all_ann, fp, indent=2)
print(f"Total: {len(all_ann)}, Visible: {sum(1 for a in all_ann if a['visibility']>0)}")
```

학습 시: `python train_tracknet.py --data ../data/ball_all/ball_combined.json --frames ../data/ball_all/frames`

---

## Task 5: 라벨링 데이터 검수 도구

**Files:**
- Create: `ml/src/review_ball_data.py`

**⚠️ 검수 완료 전까지 모델 학습 절대 금지** (피드백 메모리 참조)

- [ ] **Step 1: 검수 도구 작성**

```python
"""
Ball annotation review tool.

랜덤 N프레임을 시각화하여 라벨링 품질을 확인한다.
오류율 < 5%이어야 학습 진행 가능.

Usage:
    python review_ball_data.py --data ../data/ball_combined.json \
                               --frames ../data/sntc/frames \
                               --samples 100 --port 8082
"""

import argparse
import json
import os
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import mimetypes

REVIEW_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>Ball Annotation Review</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; }
.header { padding: 10px 16px; background: #16213e; border-bottom: 2px solid #00b894;
           display: flex; justify-content: space-between; align-items: center; }
.header h1 { font-size: 16px; color: #00b894; }
.stats { font-size: 13px; color: #aaa; }
.main { display: flex; height: calc(100vh - 42px); }
.canvas-area { flex: 1; display: flex; justify-content: center; align-items: center;
               background: #0a0a1a; position: relative; }
#canvas { max-width: 100%; max-height: 100%; }
.sidebar { width: 240px; background: #16213e; padding: 12px;
           display: flex; flex-direction: column; gap: 8px; }
.btn { padding: 8px 12px; border: none; border-radius: 6px; cursor: pointer;
       font-size: 13px; font-weight: 600; width: 100%; }
.btn-ok { background: #00b894; color: white; }
.btn-bad { background: #d63031; color: white; }
.btn-skip { background: #0f3460; color: #ddd; }
.btn:hover { opacity: 0.85; }
.info { padding: 8px; background: #0a0a1a; border-radius: 6px;
        font-size: 12px; font-family: monospace; white-space: pre-line; }
.result { padding: 8px; background: rgba(0,184,148,0.1); border-radius: 6px;
          font-size: 13px; text-align: center; }
</style>
</head>
<body>
<div class="header">
    <h1>Ball Annotation Review</h1>
    <div class="stats" id="stats">Loading...</div>
</div>
<div class="main">
    <div class="canvas-area"><canvas id="canvas"></canvas></div>
    <div class="sidebar">
        <div class="info" id="info">Loading...</div>
        <button class="btn btn-ok" onclick="mark('ok')">✓ Correct (Enter)</button>
        <button class="btn btn-bad" onclick="mark('bad')">✗ Wrong (X)</button>
        <button class="btn btn-skip" onclick="mark('skip')">Skip (S)</button>
        <div class="result" id="result"></div>
        <div style="font-size:11px; color:#666; margin-top:auto;">
            Enter=correct, X=wrong, S=skip<br>
            Arrow keys = navigate
        </div>
    </div>
</div>
<script>
let samples = [];
let idx = 0;
let reviews = {};
let canvas, ctx, img = new Image();

async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    const resp = await fetch('/api/samples');
    samples = await resp.json();
    document.addEventListener('keydown', e => {
        if (e.key === 'Enter') mark('ok');
        else if (e.key === 'x' || e.key === 'X') mark('bad');
        else if (e.key === 's' || e.key === 'S') mark('skip');
        else if (e.key === 'ArrowRight') { idx = Math.min(samples.length-1, idx+1); show(); }
        else if (e.key === 'ArrowLeft') { idx = Math.max(0, idx-1); show(); }
    });
    show();
}

function show() {
    const s = samples[idx];
    img.onload = () => {
        canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        if (s.visibility > 0 && s.x >= 0) {
            const bx = s.x * canvas.width, by = s.y * canvas.height;
            ctx.strokeStyle = '#00ff00'; ctx.lineWidth = 2;
            ctx.beginPath(); ctx.arc(bx, by, 12, 0, Math.PI*2); ctx.stroke();
            ctx.strokeStyle = '#e94560'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(bx-20,by); ctx.lineTo(bx+20,by);
            ctx.moveTo(bx,by-20); ctx.lineTo(bx,by+20); ctx.stroke();
        }
    };
    img.src = '/frames/' + s.image;
    const vis = ['NOT VISIBLE', 'VISIBLE', 'OCCLUDED'][s.visibility];
    document.getElementById('info').textContent =
        `File: ${s.image}\\nPos: (${(s.x*100).toFixed(1)}%, ${(s.y*100).toFixed(1)}%)\\nVis: ${vis}`;
    updateStats();
}

function mark(verdict) {
    reviews[idx] = verdict;
    if (idx < samples.length - 1) { idx++; show(); }
    updateStats();
    fetch('/api/review', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({idx, verdict, image: samples[idx-1]?.image})
    });
}

function updateStats() {
    const ok = Object.values(reviews).filter(v=>v==='ok').length;
    const bad = Object.values(reviews).filter(v=>v==='bad').length;
    const total = ok + bad;
    const rate = total > 0 ? (bad/total*100).toFixed(1) : '—';
    document.getElementById('stats').textContent =
        `${idx+1}/${samples.length} | OK:${ok} BAD:${bad} | Error rate: ${rate}%`;
    const pass = total >= 20 && parseFloat(rate) < 5;
    document.getElementById('result').innerHTML = total < 20
        ? `Review at least 20 samples (${total}/20)`
        : pass ? '✅ PASS — Error rate < 5%' : '❌ FAIL — Error rate ≥ 5%, fix annotations';
}

init();
</script>
</body>
</html>"""


class ReviewHandler(SimpleHTTPRequestHandler):
    samples = []
    frames_dirs = []
    reviews = []

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", ""):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(REVIEW_HTML.encode("utf-8"))
        elif parsed.path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(ReviewHandler.samples).encode("utf-8"))
        elif parsed.path.startswith("/frames/"):
            filename = parsed.path[8:]
            for d in ReviewHandler.frames_dirs:
                fp = Path(d) / filename
                if fp.exists():
                    mime, _ = mimetypes.guess_type(str(fp))
                    self.send_response(200)
                    self.send_header("Content-Type", mime or "image/jpeg")
                    self.end_headers()
                    with open(fp, "rb") as f:
                        self.wfile.write(f.read())
                    return
            self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == "/api/review":
            length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(length))
            ReviewHandler.reviews.append(data)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Review ball annotations")
    parser.add_argument("--data", required=True, help="Ball annotations JSON")
    parser.add_argument("--frames", required=True, nargs="+", help="Frame directories")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()

    with open(args.data) as f:
        all_ann = json.load(f)

    # Random sample
    n = min(args.samples, len(all_ann))
    ReviewHandler.samples = random.sample(all_ann, n)
    ReviewHandler.frames_dirs = args.frames

    visible = sum(1 for a in all_ann if a.get("visibility", 0) > 0)
    print(f"\\n=== Ball Annotation Review ===")
    print(f"Total annotations: {len(all_ann)}")
    print(f"Visible: {visible}, Not visible: {len(all_ann) - visible}")
    print(f"Reviewing {n} random samples")
    print(f"\\n  Open http://localhost:{args.port}")
    print(f"  Pass criteria: error rate < 5%")
    print(f"  Press Ctrl+C to stop\\n")

    server = HTTPServer(("localhost", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ok = sum(1 for r in ReviewHandler.reviews if r.get("verdict") == "ok")
        bad = sum(1 for r in ReviewHandler.reviews if r.get("verdict") == "bad")
        total = ok + bad
        rate = bad / total * 100 if total > 0 else 0
        print(f"\\nReview results: OK={ok}, BAD={bad}, Error rate={rate:.1f}%")
        if total >= 20 and rate < 5:
            print("✅ PASS — Proceed with training")
        else:
            print("❌ FAIL — Fix annotations before training")

        # Save review report
        report = {
            "date": __import__("datetime").datetime.now().isoformat(),
            "total_annotations": len(ReviewHandler.samples),
            "reviewed": total,
            "ok": ok,
            "bad": bad,
            "error_rate_pct": round(rate, 2),
            "pass": total >= 20 and rate < 5,
            "bad_samples": [r for r in ReviewHandler.reviews if r.get("verdict") == "bad"]
        }
        report_path = args.data.replace(".json", "_review_report.json")
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2, ensure_ascii=False)
        print(f"Report saved: {report_path}")
        server.server_close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 테스트 실행**

Run: `cd ml/src && python review_ball_data.py --data ../data/sntc/selected_annotations.json --frames ../data/sntc/frames --samples 50 --port 8082`
Expected: 브라우저에서 랜덤 50개 샘플을 OK/BAD로 판정, 오류율 표시

- [ ] **Step 3: Commit**

```bash
git add ml/src/review_ball_data.py
git commit -m "feat: add ball annotation review tool for data quality validation"
```

---

## Task 6: 어그멘테이션 파이프라인

**Files:**
- Create: `ml/src/augmentations_ball.py`

- [ ] **Step 1: 공 추적 전용 어그멘테이션 작성**

공 추적에서는 3프레임 연속 입력이므로, 프레임 간 일관된 변환이 필요하다 (같은 색상 변환 적용, 기하 변환은 미적용).

```python
"""
Ball tracking augmentations.

3프레임 연속 입력에 일관된 변환을 적용한다.
기하 변환(회전, 크롭)은 프레임 간 공 위치 관계를 깨뜨리므로 제한적으로 적용.
"""

import numpy as np
import cv2


class BallAugmentor:
    """3프레임 triplet에 일관된 어그멘테이션을 적용."""

    def __init__(self, mode="train"):
        self.mode = mode

    def __call__(self, frames: np.ndarray, heatmap: np.ndarray):
        """
        Args:
            frames: [9, H, W] float32 (3 RGB frames concatenated)
            heatmap: [H, W] float32 target heatmap
        Returns:
            augmented (frames, heatmap)
        """
        if self.mode != "train":
            return frames, heatmap

        # 1. Color jitter (same transform for all 3 frames)
        if np.random.random() < 0.5:
            brightness = np.random.uniform(0.7, 1.3)
            frames = np.clip(frames * brightness, 0, 1)

        if np.random.random() < 0.3:
            # Apply same color shift per-channel across all 3 frames
            channel_shift = np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
            for i in range(3):
                for c in range(3):
                    frames[i * 3 + c] += channel_shift[c]
            frames = np.clip(frames, 0, 1)

        # 2. Gaussian noise (independent per frame, simulates sensor noise)
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, frames.shape).astype(np.float32)
            frames = np.clip(frames + noise, 0, 1)

        # 3. Horizontal flip (flip both frames AND heatmap)
        if np.random.random() < 0.5:
            frames = np.flip(frames, axis=2).copy()  # flip W axis
            heatmap = np.flip(heatmap, axis=1).copy()  # flip W axis

        # 4. Slight vertical flip (rare, simulates upside-down camera)
        # Skip: 삼각대 고정이므로 상하 반전은 비현실적

        return frames, heatmap
```

- [ ] **Step 2: Commit**

```bash
git add ml/src/augmentations_ball.py
git commit -m "feat: add ball tracking augmentation pipeline for TrackNet"
```

---

## Task 7: 학습 스크립트 개선

**Files:**
- Modify: `ml/src/train_tracknet.py`

- [ ] **Step 1: 비디오별 train/val split 적용**

현재 코드는 프레임 단위로 랜덤 split하는데, 같은 비디오의 프레임이 train/val에 섞이면 데이터 누수가 발생한다. 비디오 단위 split으로 변경한다.

`main()` 함수의 split 로직 교체:

```python
    # Video-level split (prevent data leakage)
    video_frames = defaultdict(list)
    for ann in all_annotations:
        name = Path(ann["image"]).stem
        match = re.match(r"(.+?)_frame_\d+$", name)
        vid_id = match.group(1) if match else name
        video_frames[vid_id].append(ann)

    video_ids = sorted(video_frames.keys())
    np.random.seed(42)
    np.random.shuffle(video_ids)
    split_idx = max(1, int(len(video_ids) * 0.8))

    train_ann = []
    for vid in video_ids[:split_idx]:
        train_ann.extend(video_frames[vid])
    val_ann = []
    for vid in video_ids[split_idx:]:
        val_ann.extend(video_frames[vid])

    print(f"Videos: {len(video_ids)} total, {split_idx} train, {len(video_ids) - split_idx} val")
    print(f"Frames: {len(train_ann)} train, {len(val_ann)} val")
```

- [ ] **Step 2: early stopping 추가**

학습 루프에 early stopping 추가:

```python
    patience = 15
    no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        # ... 기존 학습 코드 ...

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            # save best model...
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break
```

- [ ] **Step 3: 어그멘테이션 모듈 연결**

`TrackNetDataset._augment()` 메서드를 `BallAugmentor` 클래스 사용으로 교체:

```python
from augmentations_ball import BallAugmentor

# __init__에서:
self.augmentor = BallAugmentor("train") if augment else BallAugmentor("val")

# _augment 교체:
def _augment(self, img, heatmap):
    img, hm = self.augmentor(img, heatmap.numpy())
    return img, torch.from_numpy(hm).float()
```

- [ ] **Step 4: 테스트 (dry run)**

```bash
cd ml/src
python -c "
from train_tracknet import TrackNetDataset
import json
with open('../data/sntc/selected_annotations.json') as f:
    ann = json.load(f)
ds = TrackNetDataset(ann[:100], '../data/sntc/frames', augment=True)
x, y, v = ds[0]
print(f'Input: {x.shape}, Target: {y.shape}, Visibility: {v}')
print(f'Dataset size: {len(ds)}')
"
```
Expected: `Input: torch.Size([9, 128, 320]), Target: torch.Size([1, 128, 320]), Visibility: tensor(1)`

- [ ] **Step 5: Commit**

```bash
git add ml/src/train_tracknet.py
git commit -m "feat: improve TrackNet training with video-level split and early stopping"
```

---

## Task 8: ONNX 변환 스크립트

**Files:**
- Create: `ml/src/export_tracknet_onnx.py`

- [ ] **Step 1: TrackNet ONNX 변환 스크립트 작성**

```python
"""
TrackNet → ONNX conversion.

Usage:
    python export_tracknet_onnx.py --checkpoint models/tracknet_best.pth \
                                   --output models/ball_tracking.onnx
    python export_tracknet_onnx.py --dummy  # Test with untrained model
"""

import argparse
import os
import numpy as np
import torch

from model_tracknet import TrackNet


def export_to_onnx(model, output_path, input_h=128, input_w=320):
    """Export TrackNet to ONNX format."""
    model.eval()
    dummy_input = torch.randn(1, 9, input_h, input_w)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["heatmap"],
        dynamic_axes=None,  # Fixed input size for mobile
    )
    print(f"Exported to: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def validate_onnx(onnx_path, input_h=128, input_w=320):
    """Validate ONNX model with ORT."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 9, input_h, input_w).astype(np.float32)
    result = session.run(None, {"input": dummy})

    heatmap = result[0]
    print(f"ONNX output shape: {heatmap.shape}")
    print(f"ONNX output range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    # Compare with PyTorch
    return heatmap


def compare_pytorch_onnx(model, onnx_path, input_h=128, input_w=320):
    """Verify PyTorch and ONNX outputs match."""
    import onnxruntime as ort

    model.eval()
    dummy = np.random.randn(1, 9, input_h, input_w).astype(np.float32)

    # PyTorch
    with torch.no_grad():
        pt_out = model(torch.from_numpy(dummy)).numpy()

    # ONNX
    session = ort.InferenceSession(onnx_path)
    ort_out = session.run(None, {"input": dummy})[0]

    diff = np.abs(pt_out - ort_out)
    print(f"Max diff: {diff.max():.8f}")
    print(f"Mean diff: {diff.mean():.8f}")

    if diff.max() < 1e-5:
        print("✅ PyTorch ↔ ONNX: MATCH")
    else:
        print("⚠️ PyTorch ↔ ONNX: slight difference (likely numerical)")


def main():
    parser = argparse.ArgumentParser(description="Export TrackNet to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint .pth")
    parser.add_argument("--output", type=str, default="models/ball_tracking.onnx")
    parser.add_argument("--dummy", action="store_true", help="Test with untrained model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = TrackNet(input_channels=9, base_filters=32)

    if args.dummy:
        print("Using DUMMY (untrained) model for pipeline test")
    elif args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("ERROR: --checkpoint or --dummy required")
        return

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params * 4 / 1024 / 1024:.1f} MB FP32)")

    export_to_onnx(model, args.output)
    validate_onnx(args.output)
    compare_pytorch_onnx(model, args.output)

    print(f"\n✅ Export complete: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 더미 모델로 파이프라인 테스트**

```bash
cd ml/src
python export_tracknet_onnx.py --dummy --output ../models/ball_tracking_dummy.onnx
```
Expected: ONNX 파일 생성 (~3.2MB), PyTorch↔ONNX diff < 1e-5

- [ ] **Step 3: Commit**

```bash
git add ml/src/export_tracknet_onnx.py
git commit -m "feat: add TrackNet ONNX export script"
```

---

## Task 9: Colab 학습 노트북

**Files:**
- Create: `ml/notebooks/TrackNet_Training.ipynb`

- [ ] **Step 1: 노트북 작성**

노트북 구조:
1. **셀 1: 환경 설정** — pip install, Google Drive 마운트
2. **셀 2: 데이터 로드** — Drive에서 어노테이션 + 프레임 로드
3. **셀 3: 데이터 통계** — visible/not visible 비율, 비디오별 분포
4. **셀 4: 학습 실행** — `train_tracknet.py` 호출 (epochs=100, batch=16, lr=1e-3)
5. **셀 5: 학습 곡선 시각화** — loss/accuracy 그래프
6. **셀 6: 검증 결과** — 테스트 프레임에서 히트맵 시각화
7. **셀 7: ONNX 변환** — `export_tracknet_onnx.py` 호출
8. **셀 8: 모델 다운로드** — Drive에 저장 또는 로컬 다운로드

```python
# 셀 1: 환경 설정
!pip install torch torchvision onnx onnxruntime pillow

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/SHOT/ml/src')

# 셀 4: 학습
!python /content/drive/MyDrive/SHOT/ml/src/train_tracknet.py \
    --data /content/drive/MyDrive/SHOT/ml/data/ball_all/ball_combined.json \
    --frames /content/drive/MyDrive/SHOT/ml/data/ball_all/frames \
    --epochs 100 --batch-size 16 --lr 1e-3 \
    --output-dir /content/drive/MyDrive/SHOT/ml/models

# 셀 7: ONNX 변환
!python /content/drive/MyDrive/SHOT/ml/src/export_tracknet_onnx.py \
    --checkpoint /content/drive/MyDrive/SHOT/ml/models/tracknet_best.pth \
    --output /content/drive/MyDrive/SHOT/ml/models/ball_tracking.onnx
```

- [ ] **Step 2: Commit**

```bash
git add ml/notebooks/TrackNet_Training.ipynb
git commit -m "feat: add TrackNet training notebook for Google Colab"
```

---

## Task 10: 학습 실행 + 검증

**Files:**
- 해당 없음 (실행 작업)

**⚠️ 선행 조건: Task 5 검수 통과 (오류율 < 5%) 필수**

- [ ] **Step 1: Colab에서 학습 실행**

```bash
python train_tracknet.py --data ball_combined.json --frames frames/ \
    --epochs 100 --batch-size 16 --lr 1e-3 --output-dir models/
```

목표 지표:
| 지표 | 목표 |
|------|------|
| Val detection accuracy | > 85% |
| Val position error | < 5px (320×128 해상도) |
| Val loss | 수렴 |

- [ ] **Step 2: 테스트 영상에서 시각적 검증**

학습 완료 후 테스트 영상 5개에서 공 검출 결과를 시각화하여 확인:
- 빠른 서브에서 공 추적 여부
- 네트 근처에서 공 검출 여부
- 선수 뒤에 가려질 때 미검출 (visibility=0) 올바른 처리

- [ ] **Step 3: ONNX 변환 + 검증**

```bash
python export_tracknet_onnx.py --checkpoint models/tracknet_best.pth \
    --output models/ball_tracking.onnx
```

- [ ] **Step 4: Commit (학습 결과물)**

```bash
git add ml/models/ball_tracking.onnx
git commit -m "feat: add trained TrackNet model (ball_tracking.onnx)"
```

---

## 전체 체크리스트

| # | Task | 산출물 | 소요 예상 |
|---|------|--------|----------|
| 1 | 프레임 추출 도구 | `extract_ball_frames.py` | 30분 |
| 2 | 라벨링 도구 개선 | `label_ball.py` 수정 | 1시간 |
| 3 | 데이터 수집 | 프레임 5,000+ | 3~5일 (촬영 포함) |
| 4 | 공 라벨링 | `ball_annotations.json` | 3~5일 |
| 5 | 검수 도구 | `review_ball_data.py` | 30분 |
| 6 | 어그멘테이션 | `augmentations_ball.py` | 30분 |
| 7 | 학습 스크립트 개선 | `train_tracknet.py` 수정 | 1시간 |
| 8 | ONNX 변환 | `export_tracknet_onnx.py` | 30분 |
| 9 | Colab 노트북 | `TrackNet_Training.ipynb` | 1시간 |
| 10 | 학습 + 검증 | `ball_tracking.onnx` | 1~2일 |

**총 예상: ~2주 (촬영 + 라벨링이 가장 시간 소요)**

**Phase 1a 완료 마일스톤:** 테스트 영상에서 공 검출률 > 85%, `ball_tracking.onnx` 생성 완료
