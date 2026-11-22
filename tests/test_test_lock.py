"""The pre-registration lock: same config -> allowed rerun; modified config -> refused."""
import pytest

from src.evaluation.run_test import TestLockError, check_lock


def test_lock_allows_identical_rerun_and_refuses_changes(tmp_path):
    lock = tmp_path / "lock.json"
    final = {"selected_model": "gbm", "gbm_params": {"learning_rate": 0.05}}
    assert check_lock(final, lock)["rerun"] is False
    assert check_lock(dict(final), lock)["rerun"] is True
    with pytest.raises(TestLockError):
        check_lock({**final, "gbm_params": {"learning_rate": 0.1}}, lock)
