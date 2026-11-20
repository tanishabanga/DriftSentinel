import sqlite3

import pytest

from driftsentinel.monitoring.prediction_storage import (
    PredictionLedgerStore,
    PredictionStorageError,
)


@pytest.fixture
def store(tmp_path):
    database_path = tmp_path / "monitoring" / "test.db"
    instance = PredictionLedgerStore(database_path)
    instance.initialize()
    return instance


def sample_predictions():
    return [
        {
            "TransactionID": 1001,
            "raw_probability": 0.75,
            "calibrated_probability": 0.70,
            "action": "block",
            "expected_cost": 0.0,
        },
        {
            "TransactionID": "TX-1002",
            "raw_probability": 0.20,
            "calibrated_probability": 0.15,
            "action": "review",
            "expected_cost": 1.5,
        },
    ]


def test_initialize_creates_database(store):
    assert store.database_path.exists()


def test_save_batch_returns_unique_id(store):
    first_id = store.save_batch(sample_predictions())
    second_id = store.save_batch(sample_predictions())

    assert first_id != second_id


def test_save_and_retrieve_batch(store):
    batch_id = store.save_batch(sample_predictions())

    result = store.get_batch(batch_id)

    assert result is not None
    assert result["batch_id"] == batch_id
    assert result["prediction_count"] == 2
    assert result["predictions"][0]["TransactionID"] == 1001
    assert result["predictions"][1]["TransactionID"] == "TX-1002"
    assert result["predictions"][0]["raw_probability"] == 0.75
    assert result["predictions"][0]["calibrated_probability"] == 0.70
    assert result["predictions"][0]["action"] == "block"
    assert result["predictions"][0]["expected_cost"] == 0.0
    assert result["predicted_at_utc"].endswith("+00:00")


def test_batch_persists_across_store_instances(tmp_path):
    database_path = tmp_path / "persistent.db"

    first_store = PredictionLedgerStore(database_path)
    first_store.initialize()
    batch_id = first_store.save_batch(sample_predictions())

    second_store = PredictionLedgerStore(database_path)
    second_store.initialize()

    result = second_store.get_batch(batch_id)

    assert result is not None
    assert result["prediction_count"] == 2


def test_missing_batch_returns_none(store):
    assert store.get_batch("does-not-exist") is None


def test_list_predictions_returns_latest_records(store):
    first_batch = store.save_batch(sample_predictions())
    second_batch = store.save_batch(sample_predictions())

    results = store.list_predictions(limit=2)

    assert len(results) == 2
    assert all(item["batch_id"] == second_batch for item in results)
    assert all(item["batch_id"] != first_batch for item in results)


def test_list_predictions_returns_empty_list_when_empty(store):
    assert store.list_predictions() == []


def test_preserves_missing_transaction_id(store):
    prediction = sample_predictions()[0]
    prediction["TransactionID"] = None

    batch_id = store.save_batch([prediction])

    result = store.get_batch(batch_id)

    assert result["predictions"][0]["TransactionID"] is None


def test_rejects_empty_batch(store):
    with pytest.raises(ValueError, match="non-empty"):
        store.save_batch([])


@pytest.mark.parametrize("predictions", [None, {}, "invalid"])
def test_rejects_invalid_batch_container(store, predictions):
    with pytest.raises(ValueError, match="non-empty"):
        store.save_batch(predictions)


def test_rejects_non_dictionary_prediction(store):
    with pytest.raises(TypeError, match="dictionary"):
        store.save_batch([{"TransactionID": 1}, "invalid"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("raw_probability", -0.1),
        ("raw_probability", 1.1),
        ("raw_probability", float("nan")),
        ("raw_probability", float("inf")),
        ("calibrated_probability", -0.1),
        ("calibrated_probability", 1.1),
        ("calibrated_probability", float("nan")),
        ("expected_cost", float("nan")),
        ("expected_cost", float("inf")),
        ("expected_cost", True),
    ],
)
def test_rejects_invalid_numeric_values(store, field, value):
    prediction = sample_predictions()[0]
    prediction[field] = value

    with pytest.raises(ValueError):
        store.save_batch([prediction])


@pytest.mark.parametrize("action", ["unknown", "APPROVE", None, 1])
def test_rejects_invalid_action(store, action):
    prediction = sample_predictions()[0]
    prediction["action"] = action

    with pytest.raises(ValueError, match="action"):
        store.save_batch([prediction])


@pytest.mark.parametrize(
    "transaction_id",
    [True, float("nan"), float("inf"), [], {}],
)
def test_rejects_invalid_transaction_id(store, transaction_id):
    prediction = sample_predictions()[0]
    prediction["TransactionID"] = transaction_id

    with pytest.raises(ValueError, match="TransactionID"):
        store.save_batch([prediction])


def test_invalid_record_does_not_partially_write_batch(store):
    predictions = sample_predictions()
    predictions.append(
        {
            "TransactionID": 1003,
            "raw_probability": 2.0,
            "calibrated_probability": 0.5,
            "action": "block",
            "expected_cost": 0.0,
        }
    )

    with pytest.raises(ValueError):
        store.save_batch(predictions)

    assert store.list_predictions() == []


def test_rejects_invalid_list_limit(store):
    for limit in (0, -1, True, 1.5, "5"):
        with pytest.raises(ValueError, match="limit"):
            store.list_predictions(limit=limit)


def test_ledger_does_not_store_raw_features(store):
    prediction = sample_predictions()[0]
    prediction["card_number"] = "sensitive-raw-feature"

    batch_id = store.save_batch([prediction])
    result = store.get_batch(batch_id)

    assert "card_number" not in result["predictions"][0]

    with sqlite3.connect(store.database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(prediction_records)"
            )
        }

    assert "card_number" not in columns
    assert "raw_features" not in columns


def test_storage_error_is_wrapped(tmp_path):
    database_path = tmp_path / "not-a-database"
    database_path.mkdir()

    instance = PredictionLedgerStore(database_path)

    with pytest.raises(PredictionStorageError):
        instance.initialize()

def test_list_batches_returns_batch_summaries(store):
    first_batch = store.save_batch(sample_predictions())
    second_batch = store.save_batch(sample_predictions())

    batches = store.list_batches()

    assert len(batches) == 2
    assert batches[0]["batch_id"] == second_batch
    assert batches[1]["batch_id"] == first_batch

    latest = batches[0]
    assert latest["prediction_count"] == 2
    assert latest["approve_count"] == 0
    assert latest["review_count"] == 1
    assert latest["block_count"] == 1


def test_list_batches_calculates_full_batch_probability_bands(store):
    predictions = [dict(item) for item in sample_predictions() for _ in range(3)]
    probabilities = [0.0, 0.3999, 0.4, 0.7999, 0.8, 1.0]

    for prediction, probability in zip(predictions, probabilities):
        prediction["calibrated_probability"] = probability

    store.save_batch(predictions)

    batch = store.list_batches()[0]

    assert batch["prediction_count"] == 6
    assert batch["lower_probability_count"] == 2
    assert batch["elevated_probability_count"] == 2
    assert batch["highest_probability_count"] == 2


def test_list_batches_returns_empty_list_when_empty(store):
    assert store.list_batches() == []


def test_get_batch_page_returns_requested_page(store):
    predictions = sample_predictions() * 2
    batch_id = store.save_batch(predictions)

    result = store.get_batch_page(batch_id, limit=2, offset=1)

    assert result is not None
    assert result["batch_id"] == batch_id
    assert result["prediction_count"] == 4
    assert result["total"] == 4
    assert result["limit"] == 2
    assert result["offset"] == 1
    assert len(result["predictions"]) == 2


def test_get_batch_page_filters_by_action(store):
    predictions = sample_predictions()
    predictions[1]["action"] = "approve"
    batch_id = store.save_batch(predictions)

    result = store.get_batch_page(batch_id, action="approve")

    assert result is not None
    assert result["prediction_count"] == 2
    assert result["total"] == 1
    assert len(result["predictions"]) == 1
    assert result["predictions"][0]["action"] == "approve"


def test_get_batch_page_searches_transaction_id(store):
    batch_id = store.save_batch(sample_predictions())

    result = store.get_batch_page(batch_id, search="TX-1002")

    assert result is not None
    assert result["total"] == 1
    assert result["predictions"][0]["TransactionID"] == "TX-1002"


def test_get_batch_page_returns_none_for_missing_batch(store):
    assert store.get_batch_page("missing-batch") is None


def test_get_batch_page_returns_empty_page_when_offset_exceeds_total(store):
    batch_id = store.save_batch(sample_predictions())

    result = store.get_batch_page(batch_id, limit=25, offset=100)

    assert result is not None
    assert result["prediction_count"] == 2
    assert result["total"] == 2
    assert result["predictions"] == []


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5, "5"])
def test_get_batch_page_rejects_invalid_limit(store, limit):
    batch_id = store.save_batch(sample_predictions())

    with pytest.raises(ValueError, match="limit"):
        store.get_batch_page(batch_id, limit=limit)


@pytest.mark.parametrize("offset", [-1, True, 1.5, "5"])
def test_get_batch_page_rejects_invalid_offset(store, offset):
    batch_id = store.save_batch(sample_predictions())

    with pytest.raises(ValueError, match="offset"):
        store.get_batch_page(batch_id, offset=offset)


def test_get_batch_page_rejects_invalid_action(store):
    batch_id = store.save_batch(sample_predictions())

    with pytest.raises(ValueError, match="action"):
        store.get_batch_page(batch_id, action="unknown")


def test_list_batches_rejects_invalid_limit(store):
    for limit in (0, -1, 101, True, 1.5, "5"):
        with pytest.raises(ValueError, match="limit"):
            store.list_batches(limit=limit)


def test_database_insert_failure_rolls_back_entire_batch(store):
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER fail_second_prediction
            BEFORE INSERT ON prediction_records
            WHEN NEW.transaction_id_json = '"TX-1002"'
            BEGIN
                SELECT RAISE(ABORT, 'forced insert failure');
            END;
            """
        )

    with pytest.raises(PredictionStorageError):
        store.save_batch(sample_predictions())

    assert store.list_predictions() == []
