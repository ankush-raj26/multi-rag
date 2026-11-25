"""Where uploaded documents (the "media") live: an S3 bucket, with the local folder as a cache.

    S3_BUCKET set       -> S3BlobStore (MinIO locally, AWS S3 / R2 / any S3 API in production)
    S3_BUCKET empty     -> LocalBlobStore (the plain data/documents folder, as before; tests use this)

The rest of the app keeps reading `settings.documents_dir`. `sync_documents()` makes that folder a
mirror of the bucket: files only in the bucket are downloaded, files only in the folder (the bundled
samples, a file copied in by hand) are uploaded. It runs at app start-up, before indexing, and after
every upload, so a teammate who clones the repo gets the same documents once they point at the
same bucket.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from app.config import Settings

logger = logging.getLogger(__name__)


class BlobStore(Protocol):
    def put(self, name: str, data: bytes) -> None: ...
    def get(self, name: str) -> bytes | None: ...
    def list(self) -> dict[str, int]: ...  # name -> size in bytes
    def delete(self, name: str) -> None: ...


class LocalBlobStore:
    """No bucket configured: the documents folder itself. put/get are then just file writes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def put(self, name: str, data: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / Path(name).name).write_bytes(data)

    def get(self, name: str) -> bytes | None:
        path = self.directory / Path(name).name
        return path.read_bytes() if path.is_file() else None

    def list(self) -> dict[str, int]:
        if not self.directory.is_dir():
            return {}
        return {p.name: p.stat().st_size for p in self.directory.iterdir() if p.is_file()}

    def delete(self, name: str) -> None:
        (self.directory / Path(name).name).unlink(missing_ok=True)


class MemoryBlobStore:
    """For tests: behaves like a bucket, lives in a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    def get(self, name: str) -> bytes | None:
        return self.objects.get(name)

    def list(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.objects.items()}

    def delete(self, name: str) -> None:
        self.objects.pop(name, None)


class S3BlobStore:
    def __init__(self, bucket: str, prefix: str = "documents/", **client_kwargs: Any) -> None:
        import boto3

        self.bucket = bucket
        self.prefix = prefix.strip("/") + "/" if prefix.strip("/") else ""
        self._s3 = boto3.client("s3", **client_kwargs)

    @classmethod
    def from_settings(cls, settings: Settings) -> S3BlobStore:
        kwargs: dict[str, Any] = {"region_name": settings.s3_region or "us-east-1"}
        if settings.s3_endpoint_url:
            kwargs["endpoint_url"] = settings.s3_endpoint_url
        if settings.s3_access_key_id:
            kwargs["aws_access_key_id"] = settings.s3_access_key_id
            kwargs["aws_secret_access_key"] = settings.s3_secret_access_key
        return cls(settings.s3_bucket, settings.s3_prefix, **kwargs)

    def _key(self, name: str) -> str:
        return self.prefix + Path(name).name

    def ensure_bucket(self) -> None:
        """Create the bucket if it does not exist (MinIO starts empty)."""
        from botocore.exceptions import ClientError

        try:
            self._s3.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
                raise
            self._s3.create_bucket(Bucket=self.bucket)
            logger.info("Created bucket %s", self.bucket)

    def ping(self) -> str:
        """'ok' or a short reason, for the health check."""
        try:
            self.ensure_bucket()
            return "ok"
        except Exception as exc:  # noqa: BLE001  (any failure is a health-check failure)
            return f"{type(exc).__name__}"

    def put(self, name: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._key(name), Body=data)

    def get(self, name: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            body = self._s3.get_object(Bucket=self.bucket, Key=self._key(name))["Body"]
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return None
            raise
        data: bytes = body.read()
        return data

    def list(self) -> dict[str, int]:
        out: dict[str, int] = {}
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"][len(self.prefix) :]
                if name and "/" not in name:
                    out[name] = int(obj["Size"])
        return out

    def delete(self, name: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._key(name))


def get_blob_store(settings: Settings) -> BlobStore:
    return S3BlobStore.from_settings(settings) if settings.s3_bucket else LocalBlobStore(settings.documents_dir)


def sync_documents(settings: Settings, blob: BlobStore | None = None) -> tuple[list[str], list[str]]:
    """Make `documents_dir` mirror the bucket. Returns (downloaded, uploaded) file names.

    With the local store this is a no-op (the folder *is* the store).
    """
    blob = blob or get_blob_store(settings)
    if isinstance(blob, LocalBlobStore):
        return [], []
    folder = settings.documents_dir
    folder.mkdir(parents=True, exist_ok=True)
    allowed = tuple(settings.allowed_upload_suffixes)
    remote = blob.list()
    local = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed}

    downloaded: list[str] = []
    for name, size in remote.items():
        if not name.lower().endswith(allowed):
            continue
        if local.get(name) != size:
            data = blob.get(name)
            if data is not None:
                (folder / name).write_bytes(data)
                downloaded.append(name)

    uploaded: list[str] = []
    for name in local:
        if name not in remote:
            blob.put(name, (folder / name).read_bytes())
            uploaded.append(name)

    if downloaded or uploaded:
        logger.info("Documents synced with bucket: %d downloaded, %d uploaded", len(downloaded), len(uploaded))
    return downloaded, uploaded
