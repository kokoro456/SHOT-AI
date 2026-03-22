"""
720p(f298) 영상에서 총 3000프레임을 균등 추출.

Usage:
    python extract_3000_frames.py
"""

import os
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    sys.exit(1)

VIDEOS_DIR = Path(__file__).resolve().parent.parent / "data" / "phone_ball" / "videos"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "phone_ball" / "frames_720p"
TOTAL_TARGET = 3000


def get_720p_videos():
    """f298(720p) 완전한 영상 파일만 반환."""
    videos = sorted(VIDEOS_DIR.glob("*.f298.mp4"))
    # .part 파일(다운로드 중) 제외
    videos = [v for v in videos if not Path(str(v) + ".part").exists()
              and not str(v).endswith(".part")]
    return videos


def get_video_info(path):
    """영상의 총 프레임 수와 fps 반환."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total, fps


def extract_frames(video_path, output_dir, num_frames):
    """영상에서 num_frames만큼 균등 간격으로 추출."""
    total_frames, fps = get_video_info(video_path)
    if total_frames == 0:
        print(f"  [ERROR] Cannot open {video_path.name}")
        return 0

    # vid_id: phone_ball_01.f298 → phone_ball_01
    vid_id = video_path.stem
    if ".f" in vid_id:
        vid_id = vid_id.split(".f")[0]

    # 균등 간격 계산
    interval = total_frames / num_frames
    target_indices = [int(i * interval) for i in range(num_frames)]

    cap = cv2.VideoCapture(str(video_path))
    extracted = 0

    for seq, frame_idx in enumerate(target_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = f"{vid_id}_frame_{seq:04d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame)
        extracted += 1

    cap.release()
    duration = total_frames / fps if fps > 0 else 0
    print(f"  [OK] {vid_id}: {extracted}/{num_frames} frames "
          f"(total {total_frames} frames, {duration:.0f}s, {fps:.0f}fps, "
          f"interval={interval:.1f})")
    return extracted


def main():
    videos = get_720p_videos()
    if not videos:
        print("720p(f298) 영상이 없습니다.")
        return

    print(f"720p 영상 {len(videos)}개 발견:")
    for v in videos:
        total, fps = get_video_info(v)
        dur = total / fps if fps > 0 else 0
        print(f"  {v.name} - {total} frames, {dur:.0f}s, {fps:.0f}fps")

    frames_per_video = TOTAL_TARGET // len(videos)
    remainder = TOTAL_TARGET % len(videos)

    print(f"\n목표: {TOTAL_TARGET}프레임, 영상당 {frames_per_video}프레임"
          f" (나머지 {remainder}프레임은 앞쪽 영상에 +1)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_extracted = 0
    for i, video in enumerate(videos):
        n = frames_per_video + (1 if i < remainder else 0)
        total_extracted += extract_frames(video, OUTPUT_DIR, n)

    print(f"\n완료: {total_extracted}프레임 → {OUTPUT_DIR}")
    print(f"\n라벨링 시작:")
    print(f"  python label_ball.py --frames {OUTPUT_DIR} --port 8081")


if __name__ == "__main__":
    main()
