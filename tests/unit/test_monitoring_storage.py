import json

import pytest

from driftsentinel.monitoring.storage import MonitoringSnapshotStore


@pytest.fixture
def store(tmp_path):
    database_path = tmp_path / "monitoring" / "test.db"
    instance = MonitoringSnapshotStore(database_path)
    instance.initialize()
    return instance


def sample_summary():
    return {
        "prediction_count": 100,
        "raw_probability": {
            "mean": 0.12,
            "median": 0.04,
        },
        "action_counts": {
            "approve": 70,
            "review": 20,
            "block": 10,
        },
    }


def test_initialize_creates_database(store):
    assert store.database_path.exists()


def test_save_snapshot_returns_unique_id(store):
    first_id = store.save_snapshot(sample_summary())
    second_id = store.save_snapshot(sample_summary())

    assert first_id != second_id


def test_save_and_retrieve_snapshot(store):
    summary = sample_summary()
    drift = {
        "features_evaluated": 431,
        "features_above_threshold": 12,
    }
    metadata = {"source": "validation_batch"}

    snapshot_id = store.save_snapshot(
        prediction_summary=summary,
        drift_summary=drift,
        metadata=metadata,
    )

    result = store.get_snapshot(snapshot_id)

    assert result is not None
    assert result["snapshot_id"] == snapshot_id
    assert result["prediction_count"] == 100
    assert result["prediction_summary"] == summary
    assert result["drift_summary"] == drift
    assert result["metadata"] == metadata
    assert result["created_at_utc"].endswith("+00:00")


def test_snapshot_persists_across_store_instances(tmp_path):
    database_path = tmp_path / "persistent.db"

    first_store = MonitoringSnapshotStore(database_path)
    first_store.initialize()

    snapshot_id = first_store.save_snapshot(sample_summary())

    second_store = MonitoringSnapshotStore(database_path)
    second_store.initialize()

    result = second_store.get_snapshot(snapshot_id)

    assert result is not None
    assert result["prediction_summary"] == sample_summary()


def test_missing_snapshot_returns_none(store):
    assert store.get_snapshot("does-not-exist") is None


def test_list_snapshots_respects_limit(store):
    ids = [
        store.save_snapshot(sample_summary())
        for _ in range(3)
    ]

    results = store.list_snapshots(limit=2)

    assert len(results) == 2
    assert all(item["snapshot_id"] in ids for item in results)


def test_list_snapshots_returns_empty_list_when_empty(store):
    assert store.list_snapshots() == []


def test_optional_drift_and_metadata_defaults(store):
    snapshot_id = store.save_snapshot(sample_summary())

    result = store.get_snapshot(snapshot_id)

    assert result["drift_summary"] is None
    assert result["metadata"] == {}


@pytest.mark.parametrize(
    "summary",
    [
        None,
        [],
        "invalid",
    ],
)
def test_rejects_non_dictionary_prediction_summary(store, summary):
    with pytest.raises(TypeError):
        store.save_snapshot(summary)


def test_rejects_missing_prediction_count(store):
    with pytest.raises(ValueError, match="prediction_count"):
        store.save_snapshot({"action_counts": {"approve": 10}})


@pytest.mark.parametrize("count", [-1, 1.5, True, "10", None])
def test_rejects_invalid_prediction_count(store, count):
    with pytest.raises(ValueError, match="prediction_count"):
        store.save_snapshot({"prediction_count": count})


def test_rejects_non_dictionary_drift_summary(store):
    with pytest.raises(TypeError):
        store.save_snapshot(sample_summary(), drift_summary=[])


def test_rejects_non_dictionary_metadata(store):
    with pytest.raises(TypeError):
        store.save_snapshot(sample_summary(), metadata=[])


def test_rejects_non_json_metadata(store):
    with pytest.raises(ValueError, match="metadata"):
        store.save_snapshot(
            sample_summary(),
            metadata={"unsupported": object()},
        )


def test_rejects_non_finite_json_values(store):
    with pytest.raises(ValueError, match="prediction_summary"):
        store.save_snapshot(
            {
                "prediction_count": 1,
                "metric": float("nan"),
            }
        )


def test_rejects_naive_window_timestamp(store):
    with pytest.raises(ValueError, match="timezone"):
        store.save_snapshot(
            sample_summary(),
            window_start_utc="2026-11-17T10:00:00",
        )


def test_rejects_invalid_window_order(store):
    with pytest.raises(ValueError, match="cannot be later"):
        store.save_snapshot(
            sample_summary(),
            window_start_utc="2026-11-17T12:00:00+00:00",
            window_end_utc="2026-11-17T10:00:00+00:00",
        )


def test_accepts_timezone_aware_window_timestamps(store):
    snapshot_id = store.save_snapshot(
        sample_summary(),
        window_start_utc="2026-11-17T10:00:00+00:00",
        window_end_utc="2026-11-17T11:00:00+00:00",
    )

    result = store.get_snapshot(snapshot_id)

    assert result["window_start_utc"] == "2026-11-17T10:00:00+00:00"
    assert result["window_end_utc"] == "2026-11-17T11:00:00+00:00"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "5"])
def test_rejects_invalid_list_limit(store, limit):
    with pytest.raises(ValueError, match="limit"):
        store.list_snapshots(limit=limit)


def test_snapshot_summary_is_valid_json(store):
    snapshot_id = store.save_snapshot(sample_summary())
    result = store.get_snapshot(snapshot_id)

    serialized = json.dumps(result["prediction_summary"])
    assert json.loads(serialized) == sample_summary()
