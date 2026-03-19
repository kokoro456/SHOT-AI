"""
Semi-Automatic Keypoint Labeling Tool v2

Improvements over v1:
1. Model prediction as initial keypoint positions (auto-placed)
2. Same-video label propagation: label first frame, auto-copy to subsequent frames
3. Accept mode: if auto-placed looks good, just press Enter to confirm

Workflow:
- Frame groups: frames from same video are grouped together
- First frame in group: model predicts → user corrects → save
- Subsequent frames: copy labels from first frame → user confirms or adjusts
- Color coding: green=model prediction, yellow=copied from sibling, blue=manually placed

Usage:
    python labeling_tool_v2.py --frames data/sntc/frames --model models/court_keypoint.tflite --port 8080
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

import numpy as np
from PIL import Image

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


def load_tflite_model(model_path):
    """Load TFLite model for prediction."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"WARNING: Could not load TFLite model: {e}")
        print("  Model predictions will not be available.")
        return None


def predict_keypoints(interpreter, image_path):
    """Run model inference on an image, return normalized keypoints."""
    if interpreter is None:
        return None

    try:
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()

        img = Image.open(image_path).convert("RGB").resize((256, 256))
        img_arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_arr = (img_arr - mean) / std
        img_arr = np.expand_dims(img_arr, 0).astype(np.float32)

        interpreter.set_tensor(inp[0]["index"], img_arr)
        interpreter.invoke()
        output = interpreter.get_tensor(out[0]["index"]).flatten()

        keypoints = {}
        for i, kp_id in enumerate(KEYPOINT_ORDER):
            x = float(output[i * 3])
            y = float(output[i * 3 + 1])
            conf = float(output[i * 3 + 2])
            keypoints[kp_id] = {
                "x": round(x, 6),
                "y": round(y, 6),
                "visible": conf > 0.3,
                "confidence": round(conf, 4),
                "source": "model"
            }
        return keypoints
    except Exception as e:
        print(f"  Prediction error for {image_path}: {e}")
        return None


def group_by_video(image_files):
    """Group frame filenames by source video.

    Assumes naming convention: videoId_frameNNN.jpg or similar.
    Falls back to treating each image as its own group.
    """
    groups = defaultdict(list)
    for f in image_files:
        # Try to extract video ID from filename
        # Common patterns: VIDEO_ID_frame_001.jpg, VIDEO_ID_001.jpg
        name = Path(f).stem
        # Split on last _NNN or _frame_NNN pattern
        match = re.match(r"(.+?)(?:_frame)?_(\d+)$", name)
        if match:
            video_id = match.group(1)
        else:
            video_id = name  # each image is its own group
        groups[video_id].append(f)

    # Sort frames within each group
    for vid in groups:
        groups[vid].sort()

    return dict(groups)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>SHOT Labeling Tool v2 (Semi-Auto)</title>
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
    width: 300px; background: #16213e; padding: 12px;
    display: flex; flex-direction: column; gap: 8px; overflow-y: auto;
}

.source-badge {
    display: inline-block; padding: 1px 5px; border-radius: 3px;
    font-size: 9px; font-weight: bold; margin-left: 4px;
}
.source-model { background: #00b894; color: white; }
.source-copied { background: #fdcb6e; color: #2d3436; }
.source-manual { background: #0984e3; color: white; }

.group-info {
    padding: 6px 8px; border-radius: 4px; font-size: 11px;
    background: rgba(108,92,231,0.2); color: #a29bfe;
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
.kp-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
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
.status-msg.auto { background: rgba(108,92,231,0.2); color: #a29bfe; }

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
    <h1>SHOT Labeling v2 (Semi-Auto)</h1>
    <div class="progress" id="progress">Loading...</div>
</div>

<div class="main">
    <div class="canvas-container" id="canvasContainer">
        <canvas id="canvas"></canvas>
        <div class="zoom-info" id="zoomInfo">100%</div>
    </div>

    <div class="sidebar">
        <div class="status-msg info" id="statusMsg">Loading...</div>
        <div class="filename" id="filename"></div>
        <div class="group-info" id="groupInfo"></div>

        <div class="kp-list" id="kpList"></div>

        <div class="btn-row">
            <button class="btn btn-secondary" onclick="undoLast()" id="btnUndo">Undo (Z)</button>
            <button class="btn btn-warning" onclick="toggleNotVisible()" id="btnNotVis">Not Visible (V)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-success" onclick="saveAndNext()" id="btnSave">✓ Accept & Next (Enter)</button>
            <button class="btn btn-secondary" onclick="skipImage()" id="btnSkip">Skip (S)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-secondary" onclick="prevImage()">← Prev (A)</button>
            <button class="btn btn-secondary" onclick="nextImage()">Next → (D)</button>
        </div>

        <div class="btn-row">
            <button class="btn btn-danger" onclick="clearAll()">Clear All (C)</button>
        </div>

        <div class="instructions">
            <strong>Semi-Auto Mode:</strong><br>
            • Green dots = model prediction<br>
            • Yellow dots = copied from sibling frame<br>
            • Click a dot to drag/adjust it<br>
            • If it looks correct, just press Enter<br><br>
            <strong>Shortcuts:</strong><br>
            Enter = accept & next | S = skip<br>
            Z = undo | V = not visible | C = clear all<br>
            A/D = prev/next | Scroll = zoom
        </div>
    </div>
</div>

<script>
const KEYPOINTS = __KEYPOINTS_JSON__;
const KP_COLORS = [
    '#e94560', '#ff6b6b', '#ee5a24',
    '#0984e3', '#00b894', '#fdcb6e', '#6c5ce7', '#a29bfe'
];

let images = [];
let currentIdx = 0;
let annotations = {};
let predictions = {};
let videoGroups = {};
let currentPoints = {};
let currentKpIdx = 8; // Start at 8 = all placed (for auto mode)
let autoMode = false; // Whether current points are auto-placed
let img = new Image();
let canvas, ctx;
let scale = 1, offsetX = 0, offsetY = 0;
let isDragging = false, dragStartX, dragStartY;
let draggingKp = null; // Which keypoint is being dragged

async function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');

    const resp = await fetch('/api/images');
    const data = await resp.json();
    images = data.images;
    annotations = data.annotations || {};
    predictions = data.predictions || {};
    videoGroups = data.video_groups || {};

    // Find first unlabeled
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

function getVideoGroup(filename) {
    for (const [vid, frames] of Object.entries(videoGroups)) {
        if (frames.includes(filename)) return { videoId: vid, frames: frames };
    }
    return null;
}

function getSiblingLabel(filename) {
    const group = getVideoGroup(filename);
    if (!group) return null;
    // Find a labeled sibling in the same video group
    for (const sibling of group.frames) {
        if (sibling !== filename && annotations[sibling]) {
            return { label: JSON.parse(JSON.stringify(annotations[sibling])), source: sibling };
        }
    }
    return null;
}

function loadImage() {
    if (images.length === 0) return;
    const filename = images[currentIdx];

    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        scale = 1; offsetX = 0; offsetY = 0;
        canvas.style.transform = '';
        redraw();
    };
    img.src = '/frames/' + filename;

    if (annotations[filename]) {
        // Already labeled
        currentPoints = JSON.parse(JSON.stringify(annotations[filename]));
        currentKpIdx = 8;
        autoMode = false;
    } else {
        // Try sibling label first, then model prediction
        const sibling = getSiblingLabel(filename);
        if (sibling) {
            currentPoints = sibling.label;
            // Mark as copied
            for (const kpId of Object.keys(currentPoints)) {
                currentPoints[kpId].source = 'copied';
            }
            currentKpIdx = 8;
            autoMode = true;
        } else if (predictions[filename]) {
            currentPoints = JSON.parse(JSON.stringify(predictions[filename]));
            currentKpIdx = 8;
            autoMode = true;
        } else {
            currentPoints = {};
            currentKpIdx = 0;
            autoMode = false;
        }
    }

    updateUI();
}

function redraw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    const kpIds = Object.keys(KEYPOINTS).map(Number);

    // Draw lines first (behind dots)
    const serviceLine = [9, 10, 11];
    const baseLine = [12, 13, 14, 15, 16];
    const sideLines = [[9, 13], [11, 15], [9, 12], [11, 16]];
    const centerLine = [[10, 14]];

    drawConnectedLine(serviceLine, '#e94560');
    drawConnectedLine(baseLine, '#0984e3');
    sideLines.forEach(pair => drawConnectedLine(pair, '#00b894'));
    centerLine.forEach(pair => drawConnectedLine(pair, '#fdcb6e'));

    // Draw keypoints
    for (let i = 0; i < kpIds.length; i++) {
        const kpId = kpIds[i];
        const pt = currentPoints[kpId];
        if (!pt) continue;

        const x = pt.x * canvas.width;
        const y = pt.y * canvas.height;

        if (pt.visible === false) {
            ctx.strokeStyle = '#636e72';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
            ctx.moveTo(x + 6, y - 6); ctx.lineTo(x - 6, y + 6);
            ctx.stroke();
        } else {
            // Color based on source
            let dotColor = KP_COLORS[i];
            let ringColor = 'white';
            if (pt.source === 'model') ringColor = '#00b894';
            else if (pt.source === 'copied') ringColor = '#fdcb6e';

            ctx.fillStyle = dotColor;
            ctx.strokeStyle = ringColor;
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(x, y, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = 'white';
            ctx.font = 'bold 13px monospace';
            ctx.fillText(kpId.toString(), x + 11, y - 5);
        }
    }

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

function findNearestKp(mx, my, threshold) {
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    let bestDist = threshold;
    let bestId = null;
    for (const kpId of kpIds) {
        const pt = currentPoints[kpId];
        if (!pt || pt.visible === false) continue;
        const dx = mx - pt.x * canvas.width;
        const dy = my - pt.y * canvas.height;
        const dist = Math.sqrt(dx*dx + dy*dy);
        if (dist < bestDist) {
            bestDist = dist;
            bestId = kpId;
        }
    }
    return bestId;
}

function onCanvasClick(e) {
    if (draggingKp !== null) return; // Was dragging, not clicking

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    // If all keypoints placed, check if clicking near existing one to adjust
    if (currentKpIdx >= 8) {
        const nearId = findNearestKp(mx, my, 20);
        if (nearId !== null) {
            // Re-place this keypoint
            currentPoints[nearId] = {
                x: mx / canvas.width,
                y: my / canvas.height,
                visible: true,
                source: 'manual'
            };
            autoMode = false;
            redraw();
            updateUI();
            return;
        }
    }

    // Place next keypoint in sequence
    if (currentKpIdx < 8) {
        const kpId = Object.keys(KEYPOINTS).map(Number)[currentKpIdx];
        currentPoints[kpId] = {
            x: mx / canvas.width,
            y: my / canvas.height,
            visible: true,
            source: 'manual'
        };
        currentKpIdx++;
        autoMode = false;
        redraw();
        updateUI();
    }
}

function onMouseDown(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    const nearId = findNearestKp(mx, my, 15);
    if (nearId !== null) {
        draggingKp = nearId;
        e.preventDefault();
    } else if (scale > 1 && e.shiftKey) {
        isDragging = true;
        dragStartX = e.clientX - offsetX;
        dragStartY = e.clientY - offsetY;
    }
}

function onMouseMove(e) {
    if (draggingKp !== null) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const mx = (e.clientX - rect.left) * scaleX;
        const my = (e.clientY - rect.top) * scaleY;
        currentPoints[draggingKp].x = mx / canvas.width;
        currentPoints[draggingKp].y = my / canvas.height;
        currentPoints[draggingKp].source = 'manual';
        autoMode = false;
        redraw();
    } else if (isDragging) {
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
        canvas.style.transform = `scale(${scale}) translate(${offsetX}px, ${offsetY}px)`;
    }
}

function onMouseUp() {
    draggingKp = null;
    isDragging = false;
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
    currentPoints[kpId] = { x: 0, y: 0, visible: false, source: 'manual' };
    currentKpIdx++;
    redraw();
    updateUI();
}

function clearAll() {
    currentPoints = {};
    currentKpIdx = 0;
    autoMode = false;
    redraw();
    updateUI();
}

async function saveAndNext() {
    const filename = images[currentIdx];
    // Clean source field before saving
    const cleanPoints = {};
    for (const [kpId, pt] of Object.entries(currentPoints)) {
        cleanPoints[kpId] = { x: pt.x, y: pt.y, visible: pt.visible };
    }
    annotations[filename] = cleanPoints;

    await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, keypoints: cleanPoints })
    });

    setStatus('Saved! ✓', 'success');
    setTimeout(() => nextImage(), 200);
}

function skipImage() { nextImage(); }
function nextImage() { if (currentIdx < images.length - 1) { currentIdx++; loadImage(); } }
function prevImage() { if (currentIdx > 0) { currentIdx--; loadImage(); } }

function onKeyDown(e) {
    if (e.key === 'z' || e.key === 'Z') undoLast();
    else if (e.key === 'v' || e.key === 'V') toggleNotVisible();
    else if (e.key === 'Enter') saveAndNext();
    else if (e.key === 's' || e.key === 'S') skipImage();
    else if (e.key === 'a' || e.key === 'A') prevImage();
    else if (e.key === 'd' || e.key === 'D') nextImage();
    else if (e.key === 'c' || e.key === 'C') clearAll();
}

function onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.max(0.5, Math.min(5, scale * delta));
    canvas.style.transform = `scale(${scale}) translate(${offsetX}px, ${offsetY}px)`;
    document.getElementById('zoomInfo').textContent = Math.round(scale * 100) + '%';
}

function updateUI() {
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    const filename = images[currentIdx];
    const labeled = Object.keys(annotations).length;

    document.getElementById('progress').textContent =
        `${labeled}/${images.length} labeled | Image ${currentIdx + 1}/${images.length}`;
    document.getElementById('filename').textContent = filename;

    // Group info
    const group = getVideoGroup(filename);
    if (group) {
        const pos = group.frames.indexOf(filename) + 1;
        document.getElementById('groupInfo').textContent =
            `Video group: ${group.videoId} (${pos}/${group.frames.length})`;
    } else {
        document.getElementById('groupInfo').textContent = '';
    }

    // Keypoint list
    let html = '';
    for (let i = 0; i < kpIds.length; i++) {
        const kpId = kpIds[i];
        const pt = currentPoints[kpId];
        const isActive = i === currentKpIdx && currentKpIdx < 8;
        const isDone = pt !== undefined;
        const isNotVis = pt && pt.visible === false;

        let cls = '';
        if (isActive) cls = 'active';
        else if (isNotVis) cls = 'skipped';
        else if (isDone) cls = 'done';

        let coord = '';
        let badge = '';
        if (isDone && pt.visible !== false) {
            coord = `(${(pt.x*100).toFixed(1)}%, ${(pt.y*100).toFixed(1)}%)`;
            if (pt.source === 'model') badge = '<span class="source-badge source-model">AI</span>';
            else if (pt.source === 'copied') badge = '<span class="source-badge source-copied">복사</span>';
            else if (pt.source === 'manual') badge = '<span class="source-badge source-manual">수동</span>';
        } else if (isNotVis) {
            coord = '(not visible)';
        }

        html += `<div class="kp-item ${cls}" onclick="jumpToKp(${i})">
            <div class="kp-dot" style="background:${KP_COLORS[i]}"></div>
            <span class="kp-name">${kpId}: ${KEYPOINTS[kpId].split('(')[0]}${badge}</span>
            <span class="kp-coord">${coord}</span>
        </div>`;
    }
    document.getElementById('kpList').innerHTML = html;

    // Status
    if (annotations[filename]) {
        setStatus('Already labeled. Edit or press Enter to keep.', 'success');
    } else if (autoMode) {
        const src = currentPoints[kpIds[0]]?.source;
        if (src === 'copied') {
            setStatus('Labels copied from sibling frame. Verify & Enter to accept.', 'auto');
        } else {
            setStatus('Model prediction shown. Adjust if needed, Enter to accept.', 'auto');
        }
    } else if (currentKpIdx < 8) {
        const nextKp = kpIds[currentKpIdx];
        setStatus(`Click to place Pt${nextKp}: ${KEYPOINTS[nextKp]}`, 'info');
    } else {
        setStatus('All 8 keypoints placed. Press Enter to save.', 'warning');
    }
}

function jumpToKp(idx) {
    const kpIds = Object.keys(KEYPOINTS).map(Number);
    for (let i = idx; i < kpIds.length; i++) {
        delete currentPoints[kpIds[i]];
    }
    currentKpIdx = idx;
    autoMode = false;
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
    frames_dir = ""
    annotations_file = ""
    annotations = {}
    predictions = {}
    video_groups = {}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "":
            self.serve_html()
        elif parsed.path == "/api/images":
            self.serve_image_list()
        elif parsed.path.startswith("/frames/"):
            self.serve_frame(parsed.path[8:])
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
            "annotations": LabelingHandler.annotations,
            "predictions": LabelingHandler.predictions,
            "video_groups": LabelingHandler.video_groups,
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
            output.append({"image": filename, "keypoints": keypoints})

        with open(LabelingHandler.annotations_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Semi-automatic keypoint labeling tool v2")
    parser.add_argument("--frames", type=str, required=True,
                        help="Directory containing frame images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output annotation file (default: {frames_dir}/annotations.json)")
    parser.add_argument("--model", type=str, default=None,
                        help="TFLite model for auto-prediction")
    parser.add_argument("--port", type=int, default=8080,
                        help="Server port")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    if not frames_dir.exists():
        print(f"ERROR: Frames directory not found: {frames_dir}")
        sys.exit(1)

    output_file = args.output or str(frames_dir / "annotations.json")

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f.name for f in frames_dir.iterdir() if f.suffix.lower() in extensions])
    print(f"Found {len(image_files)} images in {frames_dir}")

    # Group by video
    video_groups = group_by_video(image_files)
    print(f"Grouped into {len(video_groups)} video groups")
    LabelingHandler.video_groups = video_groups

    # Load existing annotations
    LabelingHandler.frames_dir = str(frames_dir)
    LabelingHandler.annotations_file = output_file

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for entry in existing:
            kps = {}
            for kp_id, pt in entry["keypoints"].items():
                kps[kp_id] = pt
            LabelingHandler.annotations[entry["image"]] = kps
        print(f"Loaded {len(LabelingHandler.annotations)} existing annotations")

    # Generate predictions with model
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            print(f"\nGenerating model predictions...")
            interpreter = load_tflite_model(model_path)
            if interpreter:
                predicted = 0
                for i, img_file in enumerate(image_files):
                    if img_file in LabelingHandler.annotations:
                        continue  # Skip already labeled
                    pred = predict_keypoints(interpreter, frames_dir / img_file)
                    if pred:
                        LabelingHandler.predictions[img_file] = pred
                        predicted += 1
                    if (i + 1) % 50 == 0:
                        print(f"  Predicted {i+1}/{len(image_files)}...")
                print(f"  Generated {predicted} predictions")
        else:
            print(f"WARNING: Model not found at {model_path}")

    unlabeled = len(image_files) - len(LabelingHandler.annotations)
    print(f"\n=== SHOT Labeling Tool v2 (Semi-Auto) ===")
    print(f"Frames: {len(image_files)} images ({len(LabelingHandler.annotations)} labeled, {unlabeled} remaining)")
    print(f"Video groups: {len(video_groups)}")
    print(f"Model predictions: {len(LabelingHandler.predictions)}")
    print(f"Output: {output_file}")
    print(f"\n  Open http://localhost:{args.port} in your browser")
    print(f"  Press Ctrl+C to stop\n")

    server = HTTPServer(("localhost", args.port), LabelingHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(LabelingHandler.annotations)} annotations to {output_file}")
        server.server_close()


if __name__ == "__main__":
    main()
