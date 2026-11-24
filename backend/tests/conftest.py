import logging
import warnings
from dataclasses import replace

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning)

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    tmp = tmp_path_factory.mktemp("var")
    return replace(Settings(), llm_provider="offline", embedder="hash", var_dir=tmp, audit_log_path=tmp / "audit.jsonl")


@pytest.fixture(scope="session")
def app(settings):
    logging.getLogger("salesai").disabled = True
    return create_app(settings)


@pytest.fixture(scope="session")
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def agent(app):
    return app.state.agent


@pytest.fixture(scope="session")
def users(agent):
    return agent.store.users


def hdr(user_id: str) -> dict[str, str]:
    return {"X-User-Id": user_id}
