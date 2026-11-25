from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings


def test_upload_saves_markdown(client: TestClient, settings: Settings) -> None:
    response = client.post("/api/documents", files={"file": ("notes.md", b"# Notes\nhello", "text/markdown")})
    assert response.status_code == 201
    assert (settings.documents_dir / "notes.md").read_text() == "# Notes\nhello"
    assert client.get("/api/documents").json() == ["notes.md"]


def test_upload_rejects_other_types(client: TestClient) -> None:
    response = client.post("/api/documents", files={"file": ("evil.exe", b"MZ", "application/octet-stream")})
    assert response.status_code == 400


def test_upload_strips_path_from_filename(client: TestClient, settings: Settings) -> None:
    client.post("/api/documents", files={"file": ("../../escape.md", b"x", "text/markdown")})
    assert (settings.documents_dir / "escape.md").exists()


def test_sync_documents_mirrors_bucket(settings: Settings) -> None:
    from dataclasses import replace

    from app.storage.blob import MemoryBlobStore, sync_documents

    s3 = replace(settings, s3_bucket="unit")
    bucket = MemoryBlobStore()
    bucket.put("remote.md", b"from the bucket")
    s3.documents_dir.mkdir(parents=True, exist_ok=True)
    (s3.documents_dir / "local.md").write_bytes(b"from the folder")

    downloaded, uploaded = sync_documents(s3, bucket)
    assert downloaded == ["remote.md"] and uploaded == ["local.md"]
    assert (s3.documents_dir / "remote.md").read_bytes() == b"from the bucket"
    assert bucket.get("local.md") == b"from the folder"
    assert sync_documents(s3, bucket) == ([], []), "a second sync has nothing to do"
