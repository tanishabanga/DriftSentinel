import { useEffect, useState } from "react";
import {
  Search,
  X,
  ExternalLink,
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";

import PageHeader from "../components/common/PageHeader";
import {
  getPredictionBatches,
  getPredictionBatchPage,
} from "../lib/api";

const PAGE_SIZE = 25;

function formatProbability(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatCost(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return value.toFixed(4);
}

function formatTimestamp(value) {
  if (!value) return "Unavailable";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return date.toLocaleString();
}

function normalizeAction(action) {
  return typeof action === "string" ? action.toLowerCase() : "unknown";
}

function getActionLabel(action) {
  const normalized = normalizeAction(action);

  if (normalized === "approve") return "Approve";
  if (normalized === "review") return "Review";
  if (normalized === "block") return "Block";

  return "Unknown";
}

function Transactions() {
  const [searchParams, setSearchParams] = useSearchParams();
  const search = searchParams.get("search") || "";

  const [batches, setBatches] = useState([]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const [predictions, setPredictions] = useState([]);
  const [totalRecords, setTotalRecords] = useState(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  const [actionFilter, setActionFilter] = useState("All");
  const [page, setPage] = useState(0);
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  const selectedBatch = batches.find(
    (batch) => batch.batch_id === selectedBatchId,
  );

  useEffect(() => {
    let active = true;

    async function loadBatches() {
      setError("");

      try {
        const response = await getPredictionBatches(100);
        if (!active) return;

        const nextBatches = Array.isArray(response.batches)
          ? response.batches
          : [];

        setBatches(nextBatches);

        setSelectedBatchId((currentId) => {
          if (
            currentId &&
            nextBatches.some((batch) => batch.batch_id === currentId)
          ) {
            return currentId;
          }

          return nextBatches[0]?.batch_id || "";
        });
      } catch (err) {
        if (!active) return;

        setError(err.message || "Unable to load prediction batches.");
        setBatches([]);
        setSelectedBatchId("");
      }
    }

    loadBatches();

    return () => {
      active = false;
    };
  }, [refreshKey]);

  useEffect(() => {
    let active = true;

    async function loadBatchPage() {
      if (!selectedBatchId) {
        setPredictions([]);
        setTotalRecords(0);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");

      try {
        const response = await getPredictionBatchPage(selectedBatchId, {
          limit: PAGE_SIZE,
          offset: page * PAGE_SIZE,
          search,
          action: actionFilter === "All" ? "" : actionFilter,
        });

        if (!active) return;

        const batch = response.batch || {};

        setPredictions(
          Array.isArray(batch.predictions) ? batch.predictions : [],
        );
        setTotalRecords(
          typeof batch.total === "number" ? batch.total : 0,
        );
      } catch (err) {
        if (!active) return;

        setError(err.message || "Unable to load prediction records.");
        setPredictions([]);
        setTotalRecords(0);
      } finally {
        if (active) setLoading(false);
      }
    }

    loadBatchPage();

    return () => {
      active = false;
    };
  }, [
    selectedBatchId,
    page,
    search,
    actionFilter,
    refreshKey,
  ]);

  function updateSearch(value) {
    const nextParams = new URLSearchParams(searchParams);

    if (value.trim()) {
      nextParams.set("search", value);
    } else {
      nextParams.delete("search");
    }

    setSearchParams(nextParams, { replace: true });
    setPage(0);
  }

  function updateActionFilter(value) {
    setActionFilter(value);
    setPage(0);
  }

  function changeBatch(batchId) {
    setSelectedBatchId(batchId);
    setPage(0);
    setSelectedTransaction(null);
  }

  function clearFilters() {
    setSearchParams({}, { replace: true });
    setActionFilter("All");
    setPage(0);
  }

  function closeDetails() {
    setSelectedTransaction(null);
  }

  const hasActiveFilters =
    search.trim() !== "" || actionFilter !== "All";

  const totalPages = Math.ceil(totalRecords / PAGE_SIZE);
  const firstRecord = totalRecords === 0 ? 0 : page * PAGE_SIZE + 1;
  const lastRecord = Math.min((page + 1) * PAGE_SIZE, totalRecords);

  const approvedCount = selectedBatch?.approve_count ?? 0;
  const reviewCount = selectedBatch?.review_count ?? 0;
  const blockedCount = selectedBatch?.block_count ?? 0;

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / TRANSACTIONS"
        title="Prediction history"
        description="Search and inspect transactions evaluated by the DriftSentinel risk engine."
      />

      <section className="metrics-grid transaction-summary">
        <div className="panel metric-card">
          <span className="summary-label">Batch predictions</span>
          <strong className="metric-value">
            {loading && !selectedBatch ? "—" : selectedBatch?.prediction_count ?? 0}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">Approved</span>
          <strong className="metric-value">
            {selectedBatch ? approvedCount : 0}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">Review required</span>
          <strong className="metric-value">
            {selectedBatch ? reviewCount : 0}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">Blocked</span>
          <strong className="metric-value">
            {selectedBatch ? blockedCount : 0}
          </strong>
        </div>
      </section>

      <section className="panel transactions-panel">
        <div className="panel-heading">
          <div>
            <h3>Prediction records</h3>
            <p>
              {loading
                ? "Loading prediction records..."
                : `Showing ${firstRecord}–${lastRecord} of ${totalRecords} matching records`}
            </p>
          </div>

          <div className="transaction-header-actions">
            <button
              type="button"
              className="clear-filters-button"
              onClick={() => setRefreshKey((value) => value + 1)}
              disabled={loading}
            >
              <RefreshCw size={14} />
              Refresh
            </button>

            {hasActiveFilters && (
              <button
                type="button"
                className="clear-filters-button"
                onClick={clearFilters}
              >
                <X size={14} />
                Clear filters
              </button>
            )}
          </div>
        </div>

        <div className="filter-toolbar">
          <select
            value={selectedBatchId}
            onChange={(event) => changeBatch(event.target.value)}
            aria-label="Select prediction batch"
            disabled={batches.length === 0}
          >
            {batches.length === 0 && (
              <option value="">No prediction batches</option>
            )}

            {batches.map((batch) => (
              <option key={batch.batch_id} value={batch.batch_id}>
                {formatTimestamp(batch.predicted_at_utc)} —{" "}
                {batch.prediction_count} records
              </option>
            ))}
          </select>

          <div className="search-field">
            <Search size={17} />

            <input
              type="text"
              value={search}
              onChange={(event) => updateSearch(event.target.value)}
              placeholder="Search transaction ID or prediction ID..."
              aria-label="Search prediction records"
            />

            {search && (
              <button
                type="button"
                className="clear-search-button"
                onClick={() => updateSearch("")}
                aria-label="Clear search"
              >
                <X size={15} />
              </button>
            )}
          </div>

          <select
            value={actionFilter}
            onChange={(event) => updateActionFilter(event.target.value)}
            aria-label="Filter by recommended action"
          >
            <option value="All">All decisions</option>
            <option value="approve">Approve</option>
            <option value="review">Review</option>
            <option value="block">Block</option>
          </select>
        </div>

        {error && (
          <div className="empty-transactions" role="alert">
            <ShieldAlert size={22} />
            <strong>Unable to load prediction history</strong>
            <span>{error}</span>

            <button
              type="button"
              className="clear-filters-button"
              onClick={() => setRefreshKey((value) => value + 1)}
            >
              <RefreshCw size={14} />
              Retry
            </button>
          </div>
        )}

        {!error && (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>TRANSACTION ID</th>
                  <th>CALIBRATED PROBABILITY</th>
                  <th>RAW PROBABILITY</th>
                  <th>EXPECTED COST</th>
                  <th>DECISION</th>
                  <th>TIME</th>
                </tr>
              </thead>

              <tbody>
                {!loading &&
                  predictions.map((prediction) => {
                    const action = normalizeAction(prediction.action);

                    return (
                      <tr key={prediction.prediction_id}>
                        <td>
                          <button
                            type="button"
                            className="transaction-detail-trigger"
                            onClick={() => setSelectedTransaction(prediction)}
                          >
                            {prediction.TransactionID ?? "Unavailable"}
                            <ExternalLink size={12} />
                          </button>
                        </td>

                        <td>
                          {formatProbability(
                            prediction.calibrated_probability,
                          )}
                        </td>

                        <td>
                          {formatProbability(prediction.raw_probability)}
                        </td>

                        <td>{formatCost(prediction.expected_cost)}</td>

                        <td>
                          <span
                            className={`action-label ${
                              ["approve", "review", "block"].includes(action)
                                ? action
                                : ""
                            }`}
                          >
                            {getActionLabel(action)}
                          </span>
                        </td>

                        <td className="time-cell">
                          {formatTimestamp(prediction.predicted_at_utc)}
                        </td>
                      </tr>
                    );
                  })}

                {loading && (
                  <tr>
                    <td colSpan="6" className="empty-state">
                      Loading prediction records...
                    </td>
                  </tr>
                )}

                {!loading && !error && predictions.length === 0 && (
                  <tr>
                    <td colSpan="6" className="empty-state">
                      <div className="empty-transactions">
                        <Search size={22} />
                        <strong>
                          {batches.length === 0
                            ? "No prediction records yet"
                            : "No matching predictions found"}
                        </strong>
                        <span>
                          {batches.length === 0
                            ? "Predictions generated through the API will appear here."
                            : "Try changing your search or decision filter."}
                        </span>

                        {hasActiveFilters && (
                          <button
                            type="button"
                            onClick={clearFilters}
                            className="clear-filters-button"
                          >
                            Clear all filters
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="transaction-pagination">
          <span>
            {totalRecords > 0
              ? `Page ${page + 1} of ${totalPages}`
              : "No records"}
          </span>

          <div>
            <button
              type="button"
              className="clear-filters-button"
              onClick={() => setPage((current) => Math.max(0, current - 1))}
              disabled={loading || page === 0}
            >
              <ChevronLeft size={14} />
              Previous
            </button>

            <button
              type="button"
              className="clear-filters-button"
              onClick={() =>
                setPage((current) =>
                  Math.min(Math.max(0, totalPages - 1), current + 1),
                )
              }
              disabled={loading || page + 1 >= totalPages}
            >
              Next
              <ChevronRight size={14} />
            </button>
          </div>
        </div>

        <p className="demo-disclaimer">
          Records are retrieved from the selected persisted prediction batch.
          Search and decision filters are applied by the API. Transaction
          feature data and original transaction amounts are not stored in
          the prediction ledger.
        </p>
      </section>

      {selectedTransaction && (
        <section className="panel transaction-details-panel">
          <div className="panel-heading">
            <div>
              <span className="details-eyebrow">
                PERSISTED PREDICTION DETAILS
              </span>
              <h3>
                Transaction{" "}
                {selectedTransaction.TransactionID ?? "Unavailable"}
              </h3>
              <p>
                Prediction ID: {selectedTransaction.prediction_id}
              </p>
            </div>

            <button
              type="button"
              className="icon-button"
              onClick={closeDetails}
              aria-label="Close prediction details"
            >
              <X size={18} />
            </button>
          </div>

          <div className="transaction-details-grid">
            <div className="detail-item">
              <span>Transaction ID</span>
              <strong>
                {selectedTransaction.TransactionID ?? "Unavailable"}
              </strong>
            </div>

            <div className="detail-item">
              <span>Raw probability</span>
              <strong>
                {formatProbability(selectedTransaction.raw_probability)}
              </strong>
            </div>

            <div className="detail-item">
              <span>Calibrated probability</span>
              <strong>
                {formatProbability(
                  selectedTransaction.calibrated_probability,
                )}
              </strong>
            </div>

            <div className="detail-item">
              <span>Expected cost</span>
              <strong>{formatCost(selectedTransaction.expected_cost)}</strong>
            </div>

            <div className="detail-item">
              <span>Prediction timestamp</span>
              <strong>
                {formatTimestamp(selectedTransaction.predicted_at_utc)}
              </strong>
            </div>

            <div className="detail-item">
              <span>Prediction batch ID</span>
              <strong>{selectedTransaction.batch_id}</strong>
            </div>
          </div>

          <div className="decision-summary">
            <div>
              {normalizeAction(selectedTransaction.action) === "approve" ? (
                <ShieldCheck size={21} />
              ) : (
                <ShieldAlert size={21} />
              )}

              <div>
                <span>Recommended action</span>
                <strong
                  className={`action-label ${normalizeAction(
                    selectedTransaction.action,
                  )}`}
                >
                  {getActionLabel(selectedTransaction.action)}
                </strong>
              </div>
            </div>
          </div>

          <div className="batch-details-section">
            <h4>Batch information</h4>
            {selectedBatch && selectedBatch.batch_id === selectedTransaction.batch_id ? (
              <p>
                This batch contains{" "}
                <strong>{selectedBatch.prediction_count}</strong>{" "}
                prediction record(s).
              </p>
            ) : (
              <p>Batch summary is unavailable for this selection.</p>
            )}
          </div>

          <p className="demo-disclaimer">
            Details are retrieved from the persistent prediction ledger.
            Original transaction features are intentionally not retained.
          </p>
        </section>
      )}
    </div>
  );
}

export default Transactions;