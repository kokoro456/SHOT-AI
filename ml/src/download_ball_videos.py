"""
Phase 2b 볼 추적 학습용 핸드폰 영상 다운로드 + 프레임 추출.

1단계: yt-dlp로 영상 다운로드 (720p)
2단계: 5fps로 프레임 추출 (30fps 전부 추출하면 너무 많음, 라벨링 부담 감소)
3단계: label_ball.py로 라벨링

Usage:
    # 전체 다운로드 + 추출
    python download_ball_videos.py --output data/phone_ball

    # 다운로드만
    python download_ball_videos.py --output data/phone_ball --download-only

    # 이미 다운로드된 영상에서 프레임 추출만
    python download_ball_videos.py --output data/phone_ball --extract-only

    # 라벨링 시작
    python label_ball.py --frames data/phone_ball/frames --port 8081
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 핸드폰 촬영 테니스 영상 22개
VIDEOS = [
    ("phone_ball_01", "https://www.youtube.com/watch?v=vfuvgJ5KJdY"),
    ("phone_ball_02", "https://www.youtube.com/watch?v=WMxDpBTTjrw"),
    ("phone_ball_03", "https://www.youtube.com/watch?v=SzgRA1TaXWw"),
    ("phone_ball_04", "https://www.youtube.com/watch?v=-WP_RxJtHhs"),
    ("phone_ball_05", "https://www.youtube.com/watch?v=h2ha0G9q4Go"),
    ("phone_ball_06", "https://www.youtube.com/watch?v=SHPOBcExoaM"),
    ("phone_ball_07", "https://www.youtube.com/watch?v=YennrQPfGl4"),
    ("phone_ball_08", "https://www.youtube.com/watch?v=mu18wybnF4U"),
    ("phone_ball_09", "https://www.youtube.com/watch?v=FAdb0xNND0c"),
    ("phone_ball_10", "https://www.youtube.com/watch?v=YdzMsOxP7TM"),
    ("phone_ball_11", "https://www.youtube.com/watch?v=usQTGHb0yQM"),
    ("phone_ball_12", "https://www.youtube.com/watch?v=__FN7Crtv4E"),
    ("phone_ball_13", "https://www.youtube.com/watch?v=sgXcznUg5YI"),
    ("phone_ball_14", "https://www.youtube.com/watch?v=H2N_hd0o6M4"),
    ("phone_ball_15", "https://www.youtube.com/watch?v=fjbsFLui3ow"),
    ("phone_ball_16", "https://www.youtube.com/watch?v=9Aoh9OpCG_w"),
    ("phone_ball_17", "https://www.youtube.com/watch?v=3YRT5lI0bFM"),
    ("phone_ball_18", "https://www.youtube.com/watch?v=DCkAdVcetVY"),
    ("phone_ball_19", "https://www.youtube.com/watch?v=UlE1FfFk12s"),
    ("phone_ball_20", "https://www.youtube.com/watch?v=9jAE_pc950g"),
    ("phone_ball_21", "https://www.youtube.com/watch?v=lv_dE14LrFg"),
    ("phone_ball_22", "https://www.youtube.com/watch?v=1En-avsyvc8"),
]


def get_ytdlp_cmd():
    """yt-dlp 실행 명령어를 찾는다."""
    # 직접 실행 시도
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return ["yt-dlp"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # python -m yt_dlp 시도
    try:
        subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                       capture_output=True, check=True)
        return [sys.executable, "-m", "yt_dlp"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return None


def check_ytdlp():
    """yt-dlp 설치 확인."""
    cmd = get_ytdlp_cmd()
    if cmd:
        return True
    print("yt-dlp가 설치되어 있지 않습니다.")
    print("설치: pip install yt-dlp")
    return False


def download_videos(output_dir: str, max_duration: int = 300):
    """영상 다운로드 (720p, 최대 5분)."""
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    ytdlp_cmd = get_ytdlp_cmd()
    downloaded = 0
    skipped = 0

    for vid_id, url in VIDEOS:
        output_path = os.path.join(videos_dir, f"{vid_id}.mp4")

        if os.path.exists(output_path):
            print(f"  [SKIP] {vid_id} -already exists (skip)")
            skipped += 1
            continue

        print(f"  [DOWN] {vid_id} <- {url}")
        try:
            cmd = ytdlp_cmd + [
                "-f", "best[height<=720][ext=mp4]/best[height<=720]/best",
                "--match-filter", f"duration<{max_duration}",
                "-o", output_path,
                "--no-playlist",
                "--js-runtimes", "node",
                url,
            ]
            subprocess.run(cmd, check=True, timeout=180)
            downloaded += 1
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT: {vid_id}")
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: {vid_id} -- {e}")

    print(f"\nDownload done: {downloaded} new, {skipped} skipped")
    return downloaded


def extract_frames(output_dir: str, fps: int = 5):
    """Extract frames from downloaded videos."""
    videos_dir = os.path.join(output_dir, "videos")
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    try:
        import cv2
    except ImportError:
        print("OpenCV required: pip install opencv-python")
        return 0

    video_files = sorted(Path(videos_dir).glob("*.mp4"))
    print(f"Extracting frames from {len(video_files)} videos (target: {fps}fps)")

    total_frames = 0
    for video_path in video_files:
        vid_id = video_path.stem

        existing = list(Path(frames_dir).glob(f"{vid_id}_frame_*.jpg"))
        if len(existing) > 10:
            print(f"  [SKIP] {vid_id} - {len(existing)} frames already exist")
            total_frames += len(existing)
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] {vid_id} - cannot open")
            continue

        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_src_frames / src_fps if src_fps > 0 else 0
        frame_interval = max(1, round(src_fps / fps))

        count = 0
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                filename = f"{vid_id}_frame_{count:04d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, filename), frame)
                count += 1
            frame_idx += 1

        cap.release()
        total_frames += count
        print(f"  [OK] {vid_id}: {count} frames ({duration:.0f}s, {src_fps:.0f}fps -> {fps}fps)")

    print(f"\nTotal frames: {total_frames} -> {frames_dir}")
    return total_frames


def main():
    parser = argparse.ArgumentParser(description="Download & extract ball training frames")
    parser.add_argument("--output", default="data/phone_ball", help="Output directory")
    parser.add_argument("--fps", type=int, default=5, help="Frame extraction rate (default: 5)")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-duration", type=int, default=300,
                        help="Max video duration in seconds (default: 300)")
    args = parser.parse_args()

    print("=" * 50)
    print("SHOT Phase 2b: Phone ball tracking data prep")
    print(f"Videos: {len(VIDEOS)}")
    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}")
    print("=" * 50)

    if not args.extract_only:
        if not check_ytdlp():
            return
        print("\n--- Step 1: Download videos ---")
        download_videos(args.output, args.max_duration)

    if not args.download_only:
        print("\n--- Step 2: Extract frames ---")
        total = extract_frames(args.output, args.fps)

        if total > 0:
            print(f"\n--- Next step ---")
            print(f"Start labeling:")
            print(f"  python label_ball.py --frames {args.output}/frames --port 8081")


if __name__ == "__main__":
    main()
