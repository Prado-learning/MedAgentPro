import argparse
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_SOURCE_URL = (
    "https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/sams/sam_vit_b_01ec64.pth"
)
DEFAULT_OUTPUT_PATH = os.path.join(
    "tools",
    "MSA",
    "checkpoint",
    "sam",
    "sam_vit_b_01ec64.pth",
)


def normalize_huggingface_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc != "huggingface.co":
        return url

    parts = [part for part in parsed.path.split("/") if part]
    if "blob" not in parts:
        return url

    blob_index = parts.index("blob")
    parts[blob_index] = "resolve"
    new_path = "/" + "/".join(parts)
    return urllib.parse.urlunparse(parsed._replace(path=new_path, query="download=true"))


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MedAgentPro/1.0 checkpoint-downloader",
        },
    )

    with urllib.request.urlopen(request) as response, open(output_path, "wb") as out_file:
        total_size = response.headers.get("Content-Length")
        total_size = int(total_size) if total_size and total_size.isdigit() else None

        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = downloaded / total_size * 100
                print(
                    f"\rDownloading: {percent:6.2f}% "
                    f"({format_size(downloaded)}/{format_size(total_size)})",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"\rDownloading: {format_size(downloaded)}",
                    end="",
                    flush=True,
                )

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the SAM ViT-B checkpoint used by MedAgentPro.")
    parser.add_argument(
        "--url",
        default=DEFAULT_SOURCE_URL,
        help="Source URL. Hugging Face blob links will be converted to resolve links automatically.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help="Destination path for the checkpoint file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_url = normalize_huggingface_url(args.url)
    output_path = os.path.abspath(args.output)

    print(f"Source URL: {source_url}")
    print(f"Output path: {output_path}")

    if os.path.exists(output_path) and not args.force:
        print("Checkpoint already exists. Use --force to re-download.")
        return 0

    try:
        download_file(source_url, output_path)
    except urllib.error.HTTPError as exc:
        print(f"HTTP error while downloading checkpoint: {exc.code} {exc.reason}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Network error while downloading checkpoint: {exc.reason}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Filesystem error while saving checkpoint: {exc}", file=sys.stderr)
        return 1

    final_size = os.path.getsize(output_path)
    print(f"Download complete: {output_path} ({format_size(final_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
