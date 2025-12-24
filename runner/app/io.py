import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable

import boto3

@dataclass
class S3Prefix:
    bucket: str
    key: str


def parse_s3_url(uri: str) -> S3Prefix:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {uri}")

    rest = uri[len("s3://") :]
    if "/" not in rest:
        raise ValueError(f"S3 URL must include object key: {uri}")

    bucket, key = rest.split("/", 1)
    key = key.lstrip("/")
    return S3Prefix(bucket=bucket, key=key)


def download_s3_file(
    s3_uri: str,
    dst_path: Path,
    region_name: Optional[str] = None,
) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    obj = parse_s3_url(s3_uri)
    session = boto3.session.Session(region_name=region_name)
    s3 = session.client("s3")

    s3.download_file(Bucket=obj.bucket, Key=obj.key, Filename=str(dst_path))
    return dst_path



def iter_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def _guess_content_type(path: Path) -> Optional[str]:
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype


def upload_dir(
    *,
    local_dir: Path,
    s3_url: str,
    s3_subdir: str = "",
    region_name: Optional[str] = None,
) -> tuple[int, int]:
    local_dir = local_dir.resolve()
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"local_dir must exist and be a directory: {local_dir}")

    target = parse_s3_url(s3_url)
    prefix_parts = [p for p in [target.prefix, s3_subdir.strip("/")] if p]
    final_prefix = "/".join(prefix_parts)

    session = boto3.session.Session(region_name=region_name)
    s3 = session.client("s3")

    uploaded = 0
    skipped = 0

    for file_path in iter_files(local_dir):
        rel = file_path.relative_to(local_dir).as_posix()
        key = f"{final_prefix}/{rel}" if final_prefix else rel

        extra_args = {}
        ctype = _guess_content_type(file_path)
        if ctype:
            extra_args["ContentType"] = ctype

        s3.upload_file(
            Filename=str(file_path),
            Bucket=target.bucket,
            Key=key,
            ExtraArgs=extra_args or None,
        )
        uploaded += 1

    return uploaded, skipped


def parse_s3_prefix(s3_prefix: str) -> S3Prefix:
    rest = s3_prefix[len("s3://") :].strip("/")
    if not rest:
        raise ValueError(f"Invalid S3 prefix: {s3_prefix}")

    parts = rest.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    prefix = prefix.strip("/")
    return S3Prefix(bucket=bucket, key=prefix)

def join_s3_key(prefix: str, key: str) -> str:
    key = key.lstrip("/")
    if not prefix:
        return key
    return f"{prefix}/{key}"

def download_by_key(
    s3_prefix: str,
    key_no_prefix: str,
    dst_path: Path,
    region_name: Optional[str] = None,
) -> Path:
    loc = parse_s3_prefix(s3_prefix)
    full_key = join_s3_key(loc.key, key_no_prefix)

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    session = boto3.session.Session(region_name=region_name)
    s3 = session.client("s3")
    s3.download_file(Bucket=loc.bucket, Key=full_key, Filename=str(dst_path))
    return dst_path