from pathlib import Path

import pytest


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "claude_sanitized.jsonl"


@pytest.fixture
def slice2_fixture_path() -> Path:
    return Path(__file__).parent / "fixtures" / "slice2_sanitized.jsonl"


@pytest.fixture
def localize_planted_fault_path() -> Path:
    return Path(__file__).parent / "fixtures" / "localize_planted_fault.jsonl"
