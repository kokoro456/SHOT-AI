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
import datetime
import json
import os
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
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
        <button class="btn btn-ok" onclick="mark('ok')">Correct (Enter)</button>
        <button class="btn btn-bad" onclick="mark('bad')">Wrong (X)</button>
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
    const rate = total > 0 ? (bad/total*100).toFixed(1) : '0';
    document.getElementById('stats').textContent =
        `${idx+1}/${samples.length} | OK:${ok} BAD:${bad} | Error rate: ${rate}%`;
    const pass = total >= 20 && parseFloat(rate) < 5;
    document.getElementById('result').innerHTML = total < 20
        ? `Review at least 20 samples (${total}/20)`
        : pass ? 'PASS - Error rate < 5%' : 'FAIL - Error rate >= 5%, fix annotations';
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
    print(f"\n=== Ball Annotation Review ===")
    print(f"Total annotations: {len(all_ann)}")
    print(f"Visible: {visible}, Not visible: {len(all_ann) - visible}")
    print(f"Reviewing {n} random samples")
    print(f"\n  Open http://localhost:{args.port}")
    print(f"  Pass criteria: error rate < 5%")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        ok = sum(1 for r in ReviewHandler.reviews if r.get("verdict") == "ok")
        bad = sum(1 for r in ReviewHandler.reviews if r.get("verdict") == "bad")
        total = ok + bad
        rate = bad / total * 100 if total > 0 else 0
        print(f"\nReview results: OK={ok}, BAD={bad}, Error rate={rate:.1f}%")
        if total >= 20 and rate < 5:
            print("PASS - Proceed with training")
        else:
            print("FAIL - Fix annotations before training")

        # Save review report
        report = {
            "date": datetime.datetime.now().isoformat(),
            "total_annotations": len(all_ann),
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
