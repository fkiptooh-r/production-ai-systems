import pytest

from shared.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Run each test in an empty folder with no relevant env vars and a fresh cache."""
    monkeypatch.chdir(tmp_path)
    for name in ("OLLAMA_BASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults_use_ipv4_loopback_not_localhost():
    settings = Settings()
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert "@127.0.0.1:5433/" in settings.database_url


def test_environment_variable_overrides_default(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.example.com:5432/prod")
    assert Settings().database_url == "postgresql://u:p@db.example.com:5432/prod"


def test_project_env_file_overrides_repo_root_env_file(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (tmp_path / ".env").write_text(
        "OLLAMA_BASE_URL=http://root:1\nDATABASE_URL=postgresql://root\n"
    )
    (project / ".env").write_text("OLLAMA_BASE_URL=http://project:2\n")

    monkeypatch.chdir(project)
    settings = Settings()

    assert settings.ollama_base_url == "http://project:2"  # project .env wins
    assert settings.database_url == "postgresql://root"  # falls back to root .env


def test_real_environment_beats_env_files(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("OLLAMA_BASE_URL=http://from-file:1\n")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:2")
    assert Settings().ollama_base_url == "http://from-env:2"


def test_settings_are_cached():
    assert get_settings() is get_settings()
