"""
Video Frame Extractor for Tennis Court Training Data

Downloads YouTube videos and extracts representative frames.
Each video produces only a few frames (not thousands) since fixed-camera
videos have nearly identical frames throughout.

Requires: yt-dlp, opencv-python (pip install yt-dlp opencv-python)

Usage:
    # Extract 1 frame per video (recommended for fixed-camera videos)
    python extract_frames.py --input data/youtube/video_list.json --output data/youtube/frames

    # Extract frames at specific intervals
    python extract_frames.py --input data/youtube/video_list.json --output data/youtube/frames --interval 300

    # Extract from a single video URL
    python extract_frames.py --url "https://youtube.com/watch?v=..." --output data/youtube/frames
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)


def download_video(url: str, output_dir: str, video_id: str) -> Optional[str]:
    """
    Download a YouTube video using yt-dlp.

    Downloads at 720p max to save space while maintaining quality.
    Returns the path to the downloaded video file.
    """
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        url,
        "-o", output_template,
        "-f", "best[height<=720]/best",  # Max 720p
        "--no-playlist",
        "--quiet",
        "--no-warnings",
    ]

    try:
        subprocess.run(cmd, check=True, timeout=300)

        # Find the downloaded file
        for ext in ["mp4", "webm", "mkv"]:
            path = os.path.join(output_dir, f"{video_id}.{ext}")
            if os.path.exists(path):
                return path

        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    Download failed: {e}")
        return None


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    video_id: str,
    interval_sec: int = 0,
    max_frames: int = 3,
) -> List[str]:
    """
    Extract representative frames from a video.

    Strategy:
    - If interval_sec == 0: Extract frames at 25%, 50%, 75% of video duration
    - If interval_sec > 0: Extract every N seconds

    Returns list of saved frame file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if duration_sec < 10:
        cap.release()
        return []

    # Determine which frames to extract
    if interval_sec > 0:
        # Extract at fixed intervals
        timestamps = list(range(interval_sec, int(duration_sec), interval_sec))
        timestamps = timestamps[:max_frames]
    else:
        # Extract at 25%, 50%, 75% of video
        timestamps = [
            duration_sec * 0.25,
            duration_sec * 0.50,
            duration_sec * 0.75,
        ]

    saved_paths = []
    for i, ts in enumerate(timestamps):
        frame_num = int(ts * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            continue

        # Check frame quality (skip very dark or very bright frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = gray.mean()
        if mean_brightness < 20 or mean_brightness > 245:
            continue

        # Save frame
        frame_filename = f"{video_id}_frame{i:02d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(frame_path)

    cap.release()
    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Extract frames from YouTube tennis videos")
    parser.add_argument("--input", type=str, default=None,
                        help="Input video list JSON (from youtube_collect.py)")
    parser.add_argument("--url", type=str, default=None,
                        help="Single YouTube video URL")
    parser.add_argument("--output", type=str, default="data/youtube/frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--interval", type=int, default=0,
                        help="Frame extraction interval in seconds (0 = auto: 25/50/75%%)")
    parser.add_argument("--max-frames", type=int, default=3,
                        help="Max frames per video")
    parser.add_argument("--keep-videos", action="store_true",
                        help="Keep downloaded video files (default: delete after extraction)")
    args = parser.parse_args()

    if not args.input and not args.url:
        print("ERROR: Provide --input (video list JSON) or --url (single video URL)")
        sys.exit(1)

    # Setup directories
    frames_dir = Path(args.output)
    frames_dir.mkdir(parents=True, exist_ok=True)
    temp_video_dir = frames_dir.parent / "temp_videos"
    temp_video_dir.mkdir(parents=True, exist_ok=True)

    # Build video list
    videos = []
    if args.url:
        # Single URL mode
        video_id = args.url.split("v=")[-1].split("&")[0]
        videos.append({"id": video_id, "url": args.url, "title": "manual"})
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        videos = data.get("videos", [])

    print(f"\n=== Frame Extraction ===")
    print(f"Videos to process: {len(videos)}")
    print(f"Output directory: {frames_dir}")
    print(f"Interval: {'auto (25/50/75%)' if args.interval == 0 else f'{args.interval}s'}")
    print(f"Max frames per video: {args.max_frames}")
    print()

    all_frames = []
    extraction_log = []

    for i, video in enumerate(videos):
        vid = video["id"]
        url = video["url"]
        title = video.get("title", "")[:50]
        try:
            print(f"[{i+1}/{len(videos)}] {title}...")
        except UnicodeEncodeError:
            print(f"[{i+1}/{len(videos)}] {vid}...")

        # Download
        video_path = download_video(url, str(temp_video_dir), vid)
        if not video_path:
            print(f"    SKIP: Download failed")
            extraction_log.append({"id": vid, "status": "download_failed"})
            continue

        # Extract frames
        frames = extract_frames_from_video(
            video_path, str(frames_dir), vid,
            interval_sec=args.interval,
            max_frames=args.max_frames,
        )

        if frames:
            print(f"    Extracted {len(frames)} frames")
            all_frames.extend(frames)
            extraction_log.append({
                "id": vid,
                "url": url,
                "title": video.get("title", ""),
                "channel": video.get("channel", ""),
                "status": "success",
                "frames": [os.path.basename(f) for f in frames],
            })
        else:
            print(f"    SKIP: No valid frames extracted")
            extraction_log.append({"id": vid, "status": "no_frames"})

        # Clean up video file
        if not args.keep_videos and os.path.exists(video_path):
            os.remove(video_path)

    # Clean up temp directory
    if not args.keep_videos and temp_video_dir.exists():
        try:
            temp_video_dir.rmdir()
        except OSError:
            pass

    # Save extraction log
    log_path = frames_dir.parent / "extraction_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_frames": len(all_frames),
            "total_videos_processed": len(videos),
            "successful": len([e for e in extraction_log if e.get("status") == "success"]),
            "entries": extraction_log,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n=== Summary ===")
    print(f"Total frames extracted: {len(all_frames)}")
    print(f"Frames saved to: {frames_dir}")
    print(f"Extraction log: {log_path}")
    print(f"\nNext step: Run predict_and_preview.py to generate keypoint previews")


if __name__ == "__main__":
    main()
