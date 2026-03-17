"""
Keypoint Labeling Tool - Local Web Server

Simple browser-based tool for manually labeling 8 tennis court keypoints.

Workflow:
1. Displays each image from the frames directory
2. User clicks 8 keypoints in order: 9,10,11,12,13,14,15,16
3. Supports undo, skip, and visible/not-visible toggle
4. Saves annotations in SHOT format (compatible with dataset.py)
5. Progress is auto-saved - can resume anytime

Usage:
    python labeling_tool.py --frames data/youtube/review/frames --port 8080

Then open http://localhost:8080 in your browser.
"""

import argparse
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import mimetypes

# Keypoint definitions
KEYPOINT_ORDER = [9, 10, 11, 12, 13, 14, 15, 16]
KEYPOINT_NAMES = {
    9: "Pt9 서비스라인 좌 (singles left)",
    10: "Pt10 서비스라인 중앙 (center mark)",
    11: "Pt11 서비스라인 우 (singles right)",
    12: "Pt12 베이스라인 복식좌 (doubles left)",
    13: "Pt13 베이스라인 단식좌 (singles left)",
    14: "Pt14 베이스라인 중앙 (center mark)",
    15: "Pt15 베이스라인 단식우 (singles right)",
    16: "Pt16 베이스라인 복식우 (doubles right)",
}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SHOT Keypoint Labeling Tool</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; overflow: hidden; }

.header {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 16px; background: #16213e; border-bottom: 2px solid #0f3460;
}
.header h1 { font-size: 16px; color: #e94560; }
.progress { font-size: 14px; color: #aaa; }

.main { display: flex; height: calc(100vh - 44px); }

.canvas-container {
    flex: 1; display: flex; justify-content: center; align-items: center;
    background: #0a0a1a; position: relative; overflow: hidden;
}
#canvas { cursor: crosshair; max-width: 100%; max-height: 100%; }

.sidebar {
    width: 280px; background: #16213e; padding: 12px;
    display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}

.kp-list { display: flex; flex-direction: column; gap: 4px; }
.kp-item {
    display: flex; align-items: center; gap: 8px; padding: 6px 8px;
    border-radius: 4px; font-size: 12px; cursor: pointer;
    border: 1px solid transparent;
}
.kp-item.active { border-color: #e94560; background: rgba(233,69,96,0.15); }
.kp-item.done { border-color: #00b894; background: rgba(0,184,148,0.1); }
.kp-item.skipped { border-color: #636e72; background: rgba(99,110,114,0.1); opacity: 0.6; }
.kp-dot {
    width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0;
}
.kp-name { flex: 1; }
.kp-coord { font-size: 10px; color: #aaa; font-family: monospace; }

.btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
.btn {
    padding: 6px 12px; border: none; border-radius: 4px;
    cursor: pointer; font-size: 12px; font-weight: 600;
}
.btn-primary { background: #e94560; color: white; }
.btn-secondary { background: #0f3460; color: #ddd; }
.btn-success { background: #00b894; color: white; }
.btn-danger { background: #d63031; color: white; }
.btn-warning { background: #fdcb6e; color: #2d3436; }
.btn:hover { opacity: 0.85; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }

.instructions {
    font-size: 11px; color: #888; line-height: 1.5;
    padding: 8px; background: #0a0a1a; border-radius: 4px;
}
.status-msg {
    padding: 6px 8px; border-radius: 4px; font-size: 12px;
    text-align: center; min-height: 28px;
}
.status-msg.info { background: rgba(9,132,227,0.2); color: #74b9ff; }
.status-msg.success { background: rgba(0,184,148,0.2); color: #55efc4; }
.status-msg.warning { background: rgba(253,203,110,0.2); color: #ffeaa7; }

.filename { font-size: 11px; color: #636e72; text-align: center; margin-top: 4px; }

.zoom-info {
    position: absolute; bottom: 8px; left: 8px;
    font-size: 11px; color: #636e72; background: rgba(0,0,0,0.6);
    padding: 2px 6px; border-radius: 3px;
}
</style>
</head>
<body>

<div class="header">
    <h1>SHOT Keypoint Labeling Tool</h1>
    <div class="progress" id="progress">Loading...</div>
</div>

<div class="main">
    <div class="canvas-container" id="canvasContainer">
        <canvas id="canvas"></canvas>
        <div class="zoom-info" id="zoomInfo">100%</div>
    </div>

    <div class="sidebar">
        <div class="status-msg info" id="statusMsg">Click to place keypoint</div>
        <div class="filename" id="filename"></div>

        <div class="kp-list" id="kpList"></div>

        <div class="btn-row">
            <button class="btn btn-secondary" onclick="undoLast()" id="btnUndo">Undo (Z)</button>
            <button class="btn btn-warning" onclick="toggleNotVisible()" id="btnNotVis">Not Visible (V)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-success" onclick="saveAndNext()" id="btnSave" disabled>Save & Next (Enter)</button>
            <button class="btn btn-secondary" onclick="skipImage()" id="btnSkip">Skip (S)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-secondary" onclick="prevImage()">← Prev (A)</button>
            <button class="btn btn-secondary" onclick="nextImage()">Next → (D)</button>
        </div>

        <div class="instructions">
            <strong>Keypoint order:</strong><br>
            9→10→11 (service line L→C→R)<br>
            12→13→14→15→16 (baseline DL→SL→C→SR→DR)<br><br>
            <strong>Shortcuts:</strong><br>
            Click = place point | Z = undo | V = not visible<br>
            Enter = save & next | S = skip | A/D = prev/next<br>
            Scroll = zoom | Drag = pan (when zoomed)
        </div>
    </div>
</div>

<script>
const KEYPOINTS = __KEYPOINTS_JSON__;
const KP_COLORS = [
    '#e94560', '#ff6b6b', '#ee5a24',  // service line (9,10,11)
    '#0984e3', '#00b894', '#fdcb6e', '#6c5ce7', '#a29bfe'  // baseline (12-16)
];

let images = [];
let currentIdx = 0;
let annotations = {};
let currentKpIdx = 0;
let currentPoints = {};
let img = new Image();
let canvas, ctx;
let scale = 1;
let offsetX = 0, offsetY = 0;
let isDragging = false, dragStartX, dragStartY;

async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    // Load image list
    const resp = await fetch('/api/images');
    const data = await resp.json();
    images = data.images;
    annotations = data.annotations || {};

    // Find first unlabeled image
    currentIdx = 0;
    for (let i = 0; i < images.length; i++) {
        if (!annotations[images[i]]) { currentIdx = i; break; }
    }

    canvas.addEventListener('click', onCanvasClick);
    canvas.addEventListener('wheel', onWheel);
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    document.addEventListener('keydown', onKeyDown);

    loadImage();
}

function loadImage() {
    if (images.length === 0) return;

    const filename = images[currentIdx];
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        scale = 1; offsetX = 0; offsetY = 0;
        redraw();
    };
    img.src = '/frames/' + filename;

    // Load existing annotation or start fresh
    if (annotations[filename]) {
        currentPoints = JSON.parse(JSON.stringify(annotations[filename]));
        currentKpIdx = 8; // All done
    } else {
        currentPoints = {};
        currentKpIdx = 0;
    }

    updateUI();
}

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    // Draw existing points
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    for (let i = 0; i < kpIds.length; i++) {
        const kpId = kpIds[i];
        const pt = currentPoints[kpId];
        if (!pt) continue;

        const color = KP_COLORS[i];
        const x = pt.x * canvas.width;
        const y = pt.y * canvas.height;

        if (pt.visible === false) {
            // Draw X for not-visible
            ctx.strokeStyle = '#636e72';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
            ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6);
            ctx.stroke();
        } else {
            // Draw circle
            ctx.fillStyle = color;
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            // Draw label
            ctx.fillStyle = color;
            ctx.font = 'bold 14px monospace';
            ctx.fillText(kpId.toString(), x + 10, y - 4);
        }
    }

    // Draw lines between placed points
    const serviceLine = [9, 10, 11];
    const baseLine = [12, 13, 14, 15, 16];
    const sideLines = [[9, 13], [11, 15]];

    drawConnectedLine(serviceLine, '#e94560');
    drawConnectedLine(baseLine, '#0984e3');
    sideLines.forEach(pair => drawConnectedLine(pair, '#00b894'));

    document.getElementById('zoomInfo').textContent = Math.round(scale * 100) + '%';
}

function drawConnectedLine(ids, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    let started = false;
    for (const id of ids) {
        const pt = currentPoints[id];
        if (pt && pt.visible !== false) {
            const x = pt.x * canvas.width;
            const y = pt.y * canvas.height;
            if (!started) { ctx.moveTo(x, y); started = true; }
            else { ctx.lineTo(x, y); }
        }
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;
}

function onCanvasClick(e) {
    if (currentKpIdx >= 8) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX / canvas.width;
    const y = (e.clientY - rect.top) * scaleY / canvas.height;

    const kpId = Object.keys(KEYPOINTS).map(Number)[currentKpIdx];
    currentPoints[kpId] = { x: x, y: y, visible: true };
    currentKpIdx++;

    redraw();
    updateUI();
}

function undoLast() {
    if (currentKpIdx <= 0) return;
    currentKpIdx--;
    const kpId = Object.keys(KEYPOINTS).map(Number)[currentKpIdx];
    delete currentPoints[kpId];
    redraw();
    updateUI();
}

function toggleNotVisible() {
    if (currentKpIdx >= 8) return;
    const kpId = Object.keys(KEYPOINTS).map(Number)[currentKpIdx];
    currentPoints[kpId] = { x: 0, y: 0, visible: false };
    currentKpIdx++;
    redraw();
    updateUI();
}

async function saveAndNext() {
    const filename = images[currentIdx];
    annotations[filename] = JSON.parse(JSON.stringify(currentPoints));

    // Send to server
    await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, keypoints: currentPoints })
    });

    setStatus('Saved! ✓', 'success');
    setTimeout(() => nextImage(), 300);
}

function skipImage() {
    nextImage();
}

function nextImage() {
    if (currentIdx < images.length - 1) {
        currentIdx++;
        loadImage();
    }
}

function prevImage() {
    if (currentIdx > 0) {
        currentIdx--;
        loadImage();
    }
}

function onKeyDown(e) {
    if (e.key === 'z' || e.key === 'Z') undoLast();
    else if (e.key === 'v' || e.key === 'V') toggleNotVisible();
    else if (e.key === 'Enter') { if (currentKpIdx >= 8) saveAndNext(); }
    else if (e.key === 's' || e.key === 'S') skipImage();
    else if (e.key === 'a' || e.key === 'A') prevImage();
    else if (e.key === 'd' || e.key === 'D') nextImage();
}

function onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.5, Math.min(5, scale * delta));
    const container = document.getElementById('canvasContainer');
    canvas.style.transform = `scale(${scale}) translate(${offsetX}px, ${offsetY}px)`;
    document.getElementById('zoomInfo').textContent = Math.round(scale * 100) + '%';
}

function onMouseDown(e) {
    if (scale > 1 && e.button === 0 && e.shiftKey) {
        isDragging = true;
        dragStartX = e.clientX - offsetX;
        dragStartY = e.clientY - offsetY;
    }
}
function onMouseMove(e) {
    if (isDragging) {
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
        canvas.style.transform = `scale(${scale}) translate(${offsetX}px, ${offsetY}px)`;
    }
}
function onMouseUp() { isDragging = false; }

function updateUI() {
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    const filename = images[currentIdx];

    // Progress
    const labeled = Object.keys(annotations).length;
    document.getElementById('progress').textContent =
        `${labeled}/${images.length} labeled | Image ${currentIdx + 1}/${images.length}`;
    document.getElementById('filename').textContent = filename;

    // Keypoint list
    let html = '';
    for (let i = 0; i < kpIds.length; i++) {
        const kpId = kpIds[i];
        const pt = currentPoints[kpId];
        const isActive = i === currentKpIdx;
        const isDone = pt !== undefined;
        const isNotVis = pt && pt.visible === false;

        let cls = '';
        if (isActive) cls = 'active';
        else if (isNotVis) cls = 'skipped';
        else if (isDone) cls = 'done';

        const coord = isDone && pt.visible !== false
            ? `(${(pt.x * 100).toFixed(1)}%, ${(pt.y * 100).toFixed(1)}%)`
            : isNotVis ? '(not visible)' : '';

        html += `<div class="kp-item ${cls}" onclick="jumpToKp(${i})">
            <div class="kp-dot" style="background:${KP_COLORS[i]}"></div>
            <span class="kp-name">${kpId}: ${KEYPOINTS[kpId].split('(')[0]}</span>
            <span class="kp-coord">${coord}</span>
        </div>`;
    }
    document.getElementById('kpList').innerHTML = html;

    // Status message
    if (currentKpIdx < 8) {
        const nextKp = kpIds[currentKpIdx];
        setStatus(`Click to place Pt${nextKp}: ${KEYPOINTS[nextKp]}`, 'info');
    } else {
        setStatus('All 8 keypoints placed. Press Enter to save.', 'warning');
    }

    // Save button
    document.getElementById('btnSave').disabled = currentKpIdx < 8;
}

function jumpToKp(idx) {
    // Allow re-placing a keypoint
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    // Remove this and all subsequent points
    for (let i = idx; i < kpIds.length; i++) {
        delete currentPoints[kpIds[i]];
    }
    currentKpIdx = idx;
    redraw();
    updateUI();
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


class LabelingHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the labeling tool."""

    frames_dir = ""
    annotations_file = ""
    annotations = {}

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "":
            self.serve_html()
        elif parsed.path == "/api/images":
            self.serve_image_list()
        elif parsed.path.startswith("/frames/"):
            self.serve_frame(parsed.path[8:])  # strip /frames/
        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/save":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body)

            filename = data["filename"]
            keypoints = data["keypoints"]
            LabelingHandler.annotations[filename] = keypoints

            # Save to file
            self.save_annotations()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_error(404)

    def serve_html(self):
        kp_json = json.dumps({str(k): v for k, v in KEYPOINT_NAMES.items()}, ensure_ascii=False)
        html = HTML_TEMPLATE.replace("__KEYPOINTS_JSON__", kp_json)

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def serve_image_list(self):
        frames_dir = Path(LabelingHandler.frames_dir)
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = sorted([
            f.name for f in frames_dir.iterdir()
            if f.suffix.lower() in extensions
        ])

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "images": images,
            "annotations": {
                k: {int(kk): vv for kk, vv in v.items()}
                for k, v in LabelingHandler.annotations.items()
            }
        }, ensure_ascii=False).encode("utf-8"))

    def serve_frame(self, filename):
        filepath = Path(LabelingHandler.frames_dir) / filename
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
        """Save annotations in SHOT training format."""
        output = []
        for filename, kps in LabelingHandler.annotations.items():
            keypoints = {}
            for kp_id_str, pt in kps.items():
                kp_id = str(kp_id_str)
                keypoints[kp_id] = {
                    "x": round(pt["x"], 6),
                    "y": round(pt["y"], 6),
                    "visible": pt.get("visible", True),
                }
            output.append({
                "image": filename,
                "keypoints": keypoints,
            })

        with open(LabelingHandler.annotations_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def log_message(self, format, *args):
        # Suppress request logs for cleaner output
        pass


def main():
    parser = argparse.ArgumentParser(description="Keypoint labeling tool")
    parser.add_argument("--frames", type=str, default="data/youtube/review/frames",
                        help="Directory containing frame images")
    parser.add_argument("--output", type=str, default="data/youtube/labeled_annotations.json",
                        help="Output annotation file")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        sys.exit(1)

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_count = len([f for f in frames_dir.iterdir() if f.suffix.lower() in extensions])

    # Load existing annotations
    LabelingHandler.frames_dir = str(frames_dir)
    LabelingHandler.annotations_file = args.output

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for entry in existing:
            kps = {}
            for kp_id, pt in entry["keypoints"].items():
                kps[int(kp_id)] = pt
            LabelingHandler.annotations[entry["image"]] = kps
        print(f"Loaded {len(LabelingHandler.annotations)} existing annotations")

    print(f"\n=== SHOT Keypoint Labeling Tool ===")
    print(f"Frames: {image_count} images in {frames_dir}")
    print(f"Output: {args.output}")
    print(f"\n  Open http://localhost:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", args.port), LabelingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(LabelingHandler.annotations)} annotations to {args.output}")
        server.server_close()


if __name__ == "__main__":
    main()
