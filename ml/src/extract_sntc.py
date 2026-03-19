"""
SNTC Channel Frame Extractor

Downloads videos from the SNTC YouTube channel and extracts 3 frames per video.
Frames are taken at 25%, 50%, 75% of video duration to get diverse court angles.

Usage:
    python extract_sntc.py --output data/sntc/frames --max-videos 678
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_video_list(channel_url, max_videos=None):
    """Get list of video IDs from channel."""
    cmd = [
        "yt-dlp", "--flat-playlist",
        "--print", "%(id)s|%(title)s|%(duration)s",
        channel_url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

    videos = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) >= 3:
            vid_id = parts[0].strip()
            title = parts[1].strip()
            try:
                duration = int(parts[2].strip()) if parts[2].strip() != "NA" else 0
            except ValueError:
                duration = 0
            # Skip very short videos (< 2 min) and very long (> 60 min)
            if duration >= 120 and duration <= 3600:
                videos.append({"id": vid_id, "title": title, "duration": duration})

    if max_videos:
        videos = videos[:max_videos]

    return videos


def extract_frames_from_video(video_id, output_dir, num_frames=3):
    """Download video and extract frames at specified positions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing = list(output_dir.glob(f"{video_id}_frame_*.jpg"))
    if len(existing) >= num_frames:
        return len(existing), "skipped"

    try:
        # Get video duration
        cmd = ["yt-dlp", "--print", "%(duration)s", f"https://www.youtube.com/watch?v={video_id}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = int(result.stdout.strip())

        if duration < 120:
            return 0, "too_short"

        # Calculate frame positions (25%, 50%, 75%)
        positions = [duration * p for p in [0.25, 0.50, 0.75]]

        extracted = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download video at low quality for speed
            video_path = os.path.join(tmpdir, "video.mp4")
            dl_cmd = [
                "yt-dlp",
                "-f", "worst[ext=mp4]/worst",  # lowest quality for speed
                "--no-playlist",
                "-o", video_path,
                f"https://www.youtube.com/watch?v={video_id}"
            ]
            subprocess.run(dl_cmd, capture_output=True, timeout=120)

            if not os.path.exists(video_path):
                return 0, "download_failed"

            # Extract frames with ffmpeg
            for i, pos in enumerate(positions):
                frame_path = output_dir / f"{video_id}_frame_{i:03d}.jpg"
                if frame_path.exists():
                    extracted += 1
                    continue

                ff_cmd = [
                    "ffmpeg", "-ss", str(int(pos)),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",  # high quality JPEG
                    "-y",
                    str(frame_path)
                ]
                subprocess.run(ff_cmd, capture_output=True, timeout=30)

                if frame_path.exists():
                    extracted += 1

        return extracted, "ok"

    except subprocess.TimeoutExpired:
        return 0, "timeout"
    except Exception as e:
        return 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from SNTC YouTube channel")
    parser.add_argument("--output", type=str, default="data/sntc/frames",
                        help="Output directory for frames")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max number of videos to process")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel download workers")
    parser.add_argument("--channel", type=str,
                        default="https://www.youtube.com/@sntcsaturdaynighttennisclub/videos",
                        help="YouTube channel URL")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get video list
    print("Fetching video list from SNTC channel...")
    videos = get_video_list(args.channel, args.max_videos)
    print(f"Found {len(videos)} eligible videos (2-60 min duration)")

    # Save video list
    list_file = output_dir.parent / "sntc_video_list.json"
    with open(list_file, "w", encoding="utf-8") as f:
        json.dump(videos, f, ensure_ascii=False, indent=2)
    print(f"Video list saved to {list_file}")

    # Step 2: Extract frames
    print(f"\nExtracting 3 frames per video (target: {len(videos) * 3} frames)...")

    total_extracted = 0
    total_skipped = 0
    total_failed = 0

    for i, video in enumerate(videos):
        vid_id = video["id"]
        count, status = extract_frames_from_video(vid_id, output_dir)

        if status == "skipped":
            total_skipped += count
        elif status == "ok":
            total_extracted += count
        else:
            total_failed += 1

        if (i + 1) % 10 == 0 or i == len(videos) - 1:
            existing = len(list(output_dir.glob("*.jpg")))
            print(f"  [{i+1}/{len(videos)}] extracted={total_extracted} skipped={total_skipped} "
                  f"failed={total_failed} total_frames={existing}")

    final_count = len(list(output_dir.glob("*.jpg")))
    print(f"\n=== Done ===")
    print(f"Total frames: {final_count}")
    print(f"Output dir: {output_dir}")
    print(f"\nNext step: Run labeling tool")
    print(f"  python labeling_tool_v2.py --frames {output_dir} --model ../court-detection/src/main/assets/court_keypoint.tflite")


if __name__ == "__main__":
    main()
