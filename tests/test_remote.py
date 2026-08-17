import pytest

from grove.remote import MlxSshWorker


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/adapter",
        "/Users/grove-worker/grove/adapters/",
        "/Users/grove-worker/grove/adapters/good/../../escape",
        "/Users/grove-worker/grove/adapters/candidate;touch-pwned",
    ],
)
def test_adapter_download_rejects_unsafe_remote_paths(tmp_path, path) -> None:
    with pytest.raises(ValueError):
        MlxSshWorker().download_adapter(path, tmp_path / "adapter")
