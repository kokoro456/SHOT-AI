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
