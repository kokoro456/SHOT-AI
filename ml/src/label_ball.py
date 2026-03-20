"""
Ball Position Labeling Tool for TrackNet

Browser-based tool for labeling tennis ball positions in video frames.
Supports frame-by-frame navigation with video playback controls.

Workflow:
1. Load extracted frames (grouped by video)
2. Click ball position in each frame (or mark as "not visible")
3. Navigate with arrow keys for fast labeling
4. Saves annotations for TrackNet training

Labels: (x, y, visibility)
- visibility: 0=not visible, 1=visible, 2=occluded (partially hidden)

Usage:
    python label_ball.py --frames data/sntc/frames --port 8081
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
import mimetypes

BALL_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SHOT Ball Labeling Tool</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; overflow: hidden; }

.header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 16px; background: #16213e; border-bottom: 2px solid #e94560;
}
.header h1 { font-size: 15px; color: #e94560; }
.progress { font-size: 13px; color: #aaa; }

.main { display: flex; height: calc(100vh - 38px); }

.canvas-container {
    flex: 1; display: flex; justify-content: center; align-items: center;
    background: #0a0a1a; position: relative; overflow: hidden;
}
#canvas { cursor: crosshair; max-width: 100%; max-height: 100%; }

.sidebar {
    width: 260px; background: #16213e; padding: 10px;
    display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}

.status-msg {
    padding: 6px 8px; border-radius: 4px; font-size: 12px; text-align: center;
}
.status-msg.info { background: rgba(9,132,227,0.2); color: #74b9ff; }
.status-msg.success { background: rgba(0,184,148,0.2); color: #55efc4; }
.status-msg.warn { background: rgba(253,203,110,0.2); color: #ffeaa7; }

.ball-info {
    padding: 8px; background: #0a0a1a; border-radius: 4px; font-size: 12px;
    font-family: monospace;
}

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
.btn {
    padding: 6px 10px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 12px; font-weight: 600; flex: 1;
}
.btn-success { background: #00b894; color: white; }
.btn-secondary { background: #0f3460; color: #ddd; }
.btn-danger { background: #d63031; color: white; }
.btn-warning { background: #fdcb6e; color: #2d3436; }
.btn:hover { opacity: 0.85; }

.nav-row { display: flex; gap: 4px; }
.nav-btn { flex: 1; padding: 8px; font-size: 14px; }

.instructions {
    font-size: 11px; color: #888; line-height: 1.4;
    padding: 8px; background: #0a0a1a; border-radius: 4px;
}

.filename { font-size: 10px; color: #636e72; text-align: center; }

.crosshair {
    position: absolute; pointer-events: none;
}
</style>
</head>
<body>

<div class="header">
    <h1>SHOT Ball Labeling</h1>
    <div class="progress" id="progress">Loading...</div>
</div>

<div class="main">
    <div class="canvas-container" id="canvasContainer">
        <canvas id="canvas"></canvas>
    </div>

    <div class="sidebar">
        <div class="status-msg info" id="statusMsg">Click ball position</div>
        <div class="filename" id="filename"></div>

        <div class="ball-info" id="ballInfo">
            Ball: not labeled
        </div>

        <div class="btn-row">
            <button class="btn btn-success" onclick="saveAndNext()">Save & Next (Enter)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-warning" onclick="markNotVisible()">Not Visible (N)</button>
            <button class="btn btn-secondary" onclick="markOccluded()">Occluded (O)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-danger" onclick="clearBall()">Clear (C)</button>
            <button class="btn btn-secondary" onclick="skipFrame()">Skip (S)</button>
        </div>

        <div class="nav-row">
            <button class="btn btn-secondary nav-btn" onclick="prevFrame()">< Prev (A)</button>
            <button class="btn btn-secondary nav-btn" onclick="nextFrame()">Next > (D)</button>
        </div>

        <div class="nav-row">
            <button class="btn btn-secondary nav-btn" onclick="jumpFrames(-10)"><<< -10</button>
            <button class="btn btn-secondary nav-btn" onclick="jumpFrames(10)">+10 >>></button>
        </div>

        <div class="instructions">
            <strong>Click</strong> = mark ball position<br>
            <strong>N</strong> = ball not visible<br>
            <strong>O</strong> = ball occluded (behind net/player)<br>
            <strong>Enter</strong> = save & next frame<br>
            <strong>A/D</strong> = prev/next | <strong>C</strong> = clear<br>
            <strong>S</strong> = skip without saving<br>
            <strong>G/H</strong> = prev/next video group<br>
            <strong>Scroll</strong> = zoom in/out<br><br>
            <em>Blue dashed circle = prev frame ball<br>
            Same video = camera is fixed</em>
        </div>
    </div>
</div>

<script>
let images = [];
let currentIdx = 0;
let annotations = {};
let ballPos = null; // {x, y, visibility}
let img = new Image();
let canvas, ctx;
let scale = 1;
let videoGroups = {};
let videoIds = [];

async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    const resp = await fetch('/api/data');
    const data = await resp.json();
    images = data.images;
    annotations = data.annotations || {};

    // Build video group index
    images.forEach((name, idx) => {
        const match = name.match(/^(.+?)_frame_/);
        const vid = match ? match[1] : 'unknown';
        if (!videoGroups[vid]) videoGroups[vid] = [];
        videoGroups[vid].push(idx);
    });
    videoIds = Object.keys(videoGroups);

    // Find first unlabeled
    for (let i = 0; i < images.length; i++) {
        if (!annotations[images[i]]) { currentIdx = i; break; }
    }

    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('wheel', onWheel);
    document.addEventListener('keydown', onKeyDown);

    loadFrame();
}

function loadFrame() {
    const filename = images[currentIdx];
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        // Auto-fit: scale canvas to fill container
        const container = document.getElementById('canvasContainer');
        const cw = container.clientWidth;
        const ch = container.clientHeight;
        const fitScale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight) * 0.95;
        scale = Math.max(fitScale, 1.5); // at least 1.5x zoom
        canvas.style.transform = `scale(${scale})`;
        canvas.style.transformOrigin = 'center center';
        redraw();
    };
    img.src = '/frames/' + filename;

    if (annotations[filename]) {
        ballPos = JSON.parse(JSON.stringify(annotations[filename]));
    } else {
        ballPos = null;
    }
    updateUI();
}

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    // Show previous frame ball position as guide
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

    if (ballPos && ballPos.visibility > 0) {
        const x = ballPos.x * canvas.width;
        const y = ballPos.y * canvas.height;

        // Crosshair
        ctx.strokeStyle = ballPos.visibility === 2 ? '#fdcb6e' : '#e94560';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 20, y); ctx.lineTo(x + 20, y);
        ctx.moveTo(x, y - 20); ctx.lineTo(x, y + 20);
        ctx.stroke();

        // Circle
        ctx.strokeStyle = ballPos.visibility === 2 ? '#fdcb6e' : '#00ff00';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.stroke();

        // Label
        ctx.fillStyle = '#00ff00';
        ctx.font = 'bold 12px monospace';
        ctx.fillText(`(${(ballPos.x*100).toFixed(1)}%, ${(ballPos.y*100).toFixed(1)}%)`,
                     x + 14, y - 6);
    }
}

function onCanvasClick(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX / canvas.width;
    const y = (e.clientY - rect.top) * scaleY / canvas.height;

    ballPos = { x: x, y: y, visibility: 1 };
    redraw();
    updateUI();
    // Auto-save and advance to next frame
    saveAndNext();
}

function markNotVisible() {
    ballPos = { x: -1, y: -1, visibility: 0 };
    updateUI();
    setStatus('Not Visible - saving...', 'warn');
    saveAndNext();
}

function markOccluded() {
    if (ballPos && ballPos.x >= 0) {
        ballPos.visibility = 2;
        redraw();
        updateUI();
        setStatus('Marked: occluded at current position', 'warn');
    } else {
        setStatus('Click ball position first, then press O', 'info');
    }
}

function clearBall() {
    ballPos = null;
    redraw();
    updateUI();
}

async function saveAndNext() {
    if (!ballPos) {
        setStatus('Click ball or press N for not visible', 'info');
        return;
    }
    const filename = images[currentIdx];
    annotations[filename] = JSON.parse(JSON.stringify(ballPos));

    await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, ball: ballPos })
    });

    setStatus('Saved!', 'success');
    setTimeout(() => nextFrame(), 150);
}

function skipFrame() { nextFrame(); }
function nextFrame() { if (currentIdx < images.length - 1) { currentIdx++; loadFrame(); } }
function prevFrame() { if (currentIdx > 0) { currentIdx--; loadFrame(); } }
function jumpFrames(n) {
    currentIdx = Math.max(0, Math.min(images.length - 1, currentIdx + n));
    loadFrame();
}

function onKeyDown(e) {
    switch(e.key) {
        case 'Enter': saveAndNext(); break;
        case 'n': case 'N': markNotVisible(); break;
        case 'o': case 'O': markOccluded(); break;
        case 'c': case 'C': clearBall(); break;
        case 's': case 'S': skipFrame(); break;
        case 'a': case 'A': case 'ArrowLeft': prevFrame(); break;
        case 'd': case 'D': case 'ArrowRight': nextFrame(); break;
        case 'g': case 'G': jumpToVideoGroup(-1); break;
        case 'h': case 'H': jumpToVideoGroup(1); break;
    }
}

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

function onWheel(e) {
    e.preventDefault();
    scale *= e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.5, Math.min(5, scale));
    canvas.style.transform = `scale(${scale})`;
}

function updateUI() {
    const filename = images[currentIdx];
    const labeled = Object.keys(annotations).length;

    document.getElementById('progress').textContent =
        `${labeled}/${images.length} | Frame ${currentIdx + 1}/${images.length}`;
    document.getElementById('filename').textContent = filename;

    let info = 'Ball: ';
    if (!ballPos) info += 'not labeled';
    else if (ballPos.visibility === 0) info += 'NOT VISIBLE';
    else if (ballPos.visibility === 2) info += `OCCLUDED (${(ballPos.x*100).toFixed(1)}%, ${(ballPos.y*100).toFixed(1)}%)`;
    else info += `(${(ballPos.x*100).toFixed(1)}%, ${(ballPos.y*100).toFixed(1)}%)`;

    document.getElementById('ballInfo').textContent = info;

    if (annotations[filename]) {
        setStatus('Already labeled. Edit or Enter to keep.', 'success');
    } else {
        setStatus('Click ball position or N for not visible', 'info');
    }
}

function setStatus(msg, type) {
    const el = document.getElementById('statusMsg');
    el.textContent = msg;
    el.className = 'status-msg ' + type;
}

init();
</script>
</body>
</html>"""


class BallLabelHandler(SimpleHTTPRequestHandler):
    frames_dir = ""
    output_file = ""
    annotations = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", ""):
            self.serve_html()
        elif parsed.path == "/api/data":
            self.serve_data()
        elif parsed.path.startswith("/frames/"):
            self.serve_frame(parsed.path[8:])
        else:
            self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path == "/api/save":
            length = int(self.headers["Content-Length"])
            data = json.loads(self.rfile.read(length))
            BallLabelHandler.annotations[data["filename"]] = data["ball"]
            self.save_annotations()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_error(404)

    def serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(BALL_HTML.encode("utf-8"))

    def serve_data(self):
        frames_dir = Path(BallLabelHandler.frames_dir)
        extensions = {".jpg", ".jpeg", ".png"}
        images = sorted([f.name for f in frames_dir.iterdir() if f.suffix.lower() in extensions])
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "images": images,
            "annotations": BallLabelHandler.annotations
        }).encode("utf-8"))

    def serve_frame(self, filename):
        filepath = Path(BallLabelHandler.frames_dir) / filename
        if not filepath.exists():
            self.send_error(404)
            return
        mime, _ = mimetypes.guess_type(str(filepath))
        self.send_response(200)
        self.send_header("Content-Type", mime or "image/jpeg")
        self.end_headers()
        with open(filepath, "rb") as f:
            self.wfile.write(f.read())

    def save_annotations(self):
        output = []
        for filename, ball in BallLabelHandler.annotations.items():
            output.append({
                "image": filename,
                "x": round(ball.get("x", -1), 6),
                "y": round(ball.get("y", -1), 6),
                "visibility": ball.get("visibility", 0)
            })
        with open(BallLabelHandler.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Ball position labeling tool")
    parser.add_argument("--frames", required=True, help="Frames directory")
    parser.add_argument("--output", default=None, help="Output file")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found")
        sys.exit(1)

    output_file = args.output or str(frames_dir.parent / "ball_annotations.json")
    BallLabelHandler.frames_dir = str(frames_dir)
    BallLabelHandler.output_file = output_file

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for entry in json.load(f):
                BallLabelHandler.annotations[entry["image"]] = {
                    "x": entry["x"], "y": entry["y"],
                    "visibility": entry["visibility"]
                }
        print(f"Loaded {len(BallLabelHandler.annotations)} annotations")

    extensions = {".jpg", ".jpeg", ".png"}
    count = len([f for f in frames_dir.iterdir() if f.suffix.lower() in extensions])

    print(f"\n=== SHOT Ball Labeling Tool ===")
    print(f"Frames: {count} in {frames_dir}")
    print(f"Output: {output_file}")
    print(f"\n  Open http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", args.port), BallLabelHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(BallLabelHandler.annotations)} annotations")
        server.server_close()


if __name__ == "__main__":
    main()
