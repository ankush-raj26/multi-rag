from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings, load_settings
from app.main import create_app

SAMPLE_DOCS = PROJECT_ROOT / "data" / "documents"


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointing every folder at a fresh temp dir."""
    return replace(
        load_settings(),
        data_dir=tmp_path,
        sessions_dir=tmp_path / "sessions",
        documents_dir=tmp_path / "documents",
        vector_dir=tmp_path / "vectorstore",
    )


@pytest.fixture()
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
