"""
YouTube Tennis Video Collector

Searches YouTube for amateur tennis court videos filmed from behind the court
(similar to SHOT app's actual usage environment) and collects video URLs.

Requires: yt-dlp (pip install yt-dlp)

Usage:
    python youtube_collect.py --output data/youtube/video_list.json --max-results 200
    python youtube_collect.py --url-file data/youtube/manual_urls.txt --output data/youtube/video_list.json
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


# Search queries designed to find phone-filmed amateur tennis court videos
DEFAULT_SEARCH_QUERIES = [
    # Korean amateur tennis
    "동호인 테니스 풀영상",
    "테니스 게임 영상 복식",
    "테니스 단식 영상",
    "생활체육 테니스",
    "테니스 연습 경기",
    "테니스 클럽 경기 영상",
    "테니스 동호회 경기",
    # English amateur tennis
    "amateur tennis match full video",
    "tennis practice match behind court",
    "recreational tennis doubles match",
    "tennis club match footage",
    "tennis court behind baseline camera",
]


def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True, text=True, check=True
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def search_youtube(query: str, max_results: int = 30) -> List[Dict]:
    """
    Search YouTube for videos matching query using yt-dlp.

    Returns list of video metadata dicts.
    """
    print(f"  Searching: '{query}' (max {max_results})...")

    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )

        videos = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                videos.append({
                    "id": data.get("id", ""),
                    "url": f"https://www.youtube.com/watch?v={data.get('id', '')}",
                    "title": data.get("title", ""),
                    "channel": data.get("channel", data.get("uploader", "")),
                    "duration": data.get("duration", 0),
                    "view_count": data.get("view_count", 0),
                    "query": query,
                })
            except json.JSONDecodeError:
                continue

        print(f"    Found {len(videos)} videos")
        return videos

    except subprocess.TimeoutExpired:
        print(f"    Timeout for query: {query}")
        return []
    except Exception as e:
        print(f"    Error: {e}")
        return []


def get_video_info(url: str) -> Optional[Dict]:
    """Get metadata for a single video URL."""
    cmd = [
        "yt-dlp",
        url,
        "--dump-json",
        "--no-download",
        "--quiet",
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        return {
            "id": data.get("id", ""),
            "url": url,
            "title": data.get("title", ""),
            "channel": data.get("channel", data.get("uploader", "")),
            "duration": data.get("duration", 0),
            "view_count": data.get("view_count", 0),
            "query": "manual",
        }
    except Exception:
        return None


def load_manual_urls(url_file: str) -> List[str]:
    """Load manually curated video URLs from a text file (one URL per line)."""
    urls = []
    with open(url_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def filter_videos(videos: List[Dict], min_duration: int = 60, max_duration: int = 7200) -> List[Dict]:
    """
    Filter videos by basic criteria.

    Args:
        min_duration: Minimum video length in seconds (default: 1 min)
        max_duration: Maximum video length in seconds (default: 2 hours)
    """
    filtered = []
    for v in videos:
        duration = v.get("duration", 0) or 0

        # Skip very short or very long videos
        if duration < min_duration or duration > max_duration:
            continue

        filtered.append(v)

    return filtered


def deduplicate(videos: List[Dict]) -> List[Dict]:
    """Remove duplicate videos by ID."""
    seen = set()
    unique = []
    for v in videos:
        vid = v["id"]
        if vid not in seen:
            seen.add(vid)
            unique.append(v)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Collect YouTube tennis video URLs")
    parser.add_argument("--output", type=str, default="data/youtube/video_list.json",
                        help="Output JSON file path")
    parser.add_argument("--max-results", type=int, default=20,
                        help="Max results per search query")
    parser.add_argument("--url-file", type=str, default=None,
                        help="Text file with manually curated URLs (one per line)")
    parser.add_argument("--queries", nargs="+", default=None,
                        help="Custom search queries (overrides defaults)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip YouTube search, only process manual URLs")
    args = parser.parse_args()

    if not check_ytdlp():
        print("ERROR: yt-dlp is not installed.")
        print("Install with: pip install yt-dlp")
        sys.exit(1)

    all_videos = []

    # 1. Search YouTube
    if not args.skip_search:
        queries = args.queries or DEFAULT_SEARCH_QUERIES
        print(f"\n=== YouTube Search ({len(queries)} queries) ===\n")

        for query in queries:
            results = search_youtube(query, args.max_results)
            all_videos.extend(results)

    # 2. Add manual URLs
    if args.url_file and os.path.exists(args.url_file):
        print(f"\n=== Manual URLs from {args.url_file} ===\n")
        urls = load_manual_urls(args.url_file)
        print(f"  Loading {len(urls)} manual URLs...")

        for url in urls:
            info = get_video_info(url)
            if info:
                all_videos.append(info)
                print(f"    + {info['title'][:50]}... ({info['duration']}s)")
            else:
                print(f"    SKIP: {url}")

    # 3. Deduplicate
    all_videos = deduplicate(all_videos)
    print(f"\nTotal unique videos: {len(all_videos)}")

    # 4. Filter
    filtered = filter_videos(all_videos)
    print(f"After filtering (duration 1min~2hr): {len(filtered)}")

    # 5. Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "collected_at": datetime.now().isoformat(),
        "total_videos": len(filtered),
        "videos": filtered,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_path}")
    print(f"\nNext step: Run extract_frames.py to download and extract frames")


if __name__ == "__main__":
    main()
