import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  Filter,
  X,
  RefreshCw,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";
import RiskDistribution from "../components/common/RiskDistribution";
import {
  getPredictionBatches,
  getPredictionBatchTopRisks,
} from "../lib/api";

const ACTIONS = ["approve", "review", "block"];

const ACTION_COLORS = {
  approve: "#269D83",
  review: "#E8A642",
  block: "#F47781",
};

function formatProbability(value) {
  const probability = Number(value);
  return Number.isFinite(probability)
    ? `${(probability * 100).toFixed(2)}%`
    : "?";
}

function formatCost(value) {
  const cost = Number(value);
  return Number.isFinite(cost) ? cost.toFixed(4) : "?";
}

function formatDate(value) {
  if (!value) return "?";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "?";

  return date.toLocaleString();
}

function RiskAnalysis() {
  const [batches, setBatches] = useState([]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const [topPredictions, setTopPredictions] = useState([]);
  const [actionFilter, setActionFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");

  async function loadBatches(isRefresh = false) {
    if (isRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    setError("");

    try {
      const response = await getPredictionBatches(100);

      if (!Array.isArray(response.batches)) {
        throw new Error("The prediction batch response is invalid.");
      }

      setBatches(response.batches);

      setSelectedBatchId((currentId) => {
        if (currentId && response.batches.some(
          (batch) => batch.batch_id === currentId
        )) {
          return currentId;
        }

        return response.batches[0]?.batch_id || "";
      });
    } catch (err) {
      setError(err.message || "Could not load prediction batches.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    loadBatches();
  }, []);

  useEffect(() => {
    let active = true;

    async function loadTopRisks() {
      if (!selectedBatchId) {
        setTopPredictions([]);
        return;
      }

      setError("");

      try {
        const response = await getPredictionBatchTopRisks(
          selectedBatchId,
          {
            limit: 20,
            action: actionFilter === "all" ? "" : actionFilter,
          }
        );

        if (!Array.isArray(response.batch?.predictions)) {
          throw new Error("The top-risk response is invalid.");
        }

        if (active) {
          setTopPredictions(response.batch.predictions);
        }
      } catch (err) {
        if (active) {
          setTopPredictions([]);
          setError(err.message || "Could not load top-risk predictions.");
        }
      }
    }

    loadTopRisks();

    return () => {
      active = false;
    };
  }, [selectedBatchId, actionFilter]);

  const selectedBatch = useMemo(
    () => batches.find((batch) => batch.batch_id === selectedBatchId) || null,
    [batches, selectedBatchId]
  );

  const probabilityCounts = {
    lower: selectedBatch?.lower_probability_count || 0,
    elevated: selectedBatch?.elevated_probability_count || 0,
    highest: selectedBatch?.highest_probability_count || 0,
  };

  const actionCounts = {
    approve: selectedBatch?.approve_count || 0,
    review: selectedBatch?.review_count || 0,
    block: selectedBatch?.block_count || 0,
  };

  const totalPredictions = selectedBatch?.prediction_count || 0;

  const actionDistribution = ACTIONS.map((action) => ({
    action,
    count: actionCounts[action],
    percentage:
      totalPredictions > 0
        ? (actionCounts[action] / totalPredictions) * 100
        : 0,
  }));

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / RISK ANALYSIS"
        title="Risk analysis"
        description="Explore predicted fraud probabilities and the decisions assigned by the DriftSentinel risk engine."
      />

      {error && (
        <div className="panel form-error" role="alert">
          <strong>Unable to load risk analysis.</strong>
          <p>{error}</p>
          <button
            type="button"
            className="clear-filters-button"
            onClick={() => loadBatches(true)}
          >
            <RefreshCw size={14} />
            Retry
          </button>
        </div>
      )}

      <section className="panel risk-filter-toolbar">
        <div className="risk-filter-label">
          <Filter size={16} />
          Analyze prediction batch
        </div>

        <select
          value={selectedBatchId}
          onChange={(event) => setSelectedBatchId(event.target.value)}
          aria-label="Select prediction batch"
          disabled={loading || batches.length === 0}
        >
          {batches.length === 0 && (
            <option value="">No prediction batches available</option>
          )}
          {batches.map((batch) => (
            <option key={batch.batch_id} value={batch.batch_id}>
              {formatDate(batch.predicted_at_utc)} ?{" "}
              {Number(batch.prediction_count).toLocaleString()} predictions
            </option>
          ))}
        </select>

        <button
          type="button"
          className="clear-filters-button"
          onClick={() => loadBatches(true)}
          disabled={refreshing}
        >
          <RefreshCw size={14} />
          {refreshing ? "Refreshing..." : "Refresh"}
        </button>
      </section>

      <section className="metrics-grid">
        <div className="panel metric-card">
          <div className="metric-top">
            <span>Highest probability (80-100%)</span>
            <AlertTriangle color="#F47781" />
          </div>
          <strong className="metric-value">
            {loading ? "?" : probabilityCounts.highest}
          </strong>
          <span className="summary-label">Calibrated probability band</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Elevated probability (40-80%)</span>
            <ShieldAlert color="#E8A642" />
          </div>
          <strong className="metric-value">
            {loading ? "?" : probabilityCounts.elevated}
          </strong>
          <span className="summary-label">Calibrated probability band</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Lower probability (0-40%)</span>
            <ShieldCheck color="#269D83" />
          </div>
          <strong className="metric-value">
            {loading ? "?" : probabilityCounts.lower}
          </strong>
          <span className="summary-label">Calibrated probability band</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Manual reviews</span>
            <ShieldAlert color="#E8A642" />
          </div>
          <strong className="metric-value">
            {loading ? "?" : actionCounts.review}
          </strong>
          <span className="summary-label">Assigned by decision policy</span>
        </div>
      </section>

      <section className="charts-grid">
        <RiskDistribution counts={probabilityCounts} />

        <div className="panel risk-explanation">
          <h3>Understanding predicted risk</h3>
          <p>
            These probability bands group calibrated model scores for
            visualization. They are descriptive, not policy thresholds.
          </p>

          <div className="risk-rule">
            <span className="action-color-dot" style={{ background: "#269D83" }} />
            <span>0-40% - Lower predicted probability</span>
          </div>

          <div className="risk-rule">
            <span className="action-color-dot" style={{ background: "#E8A642" }} />
            <span>40-80% - Elevated predicted probability</span>
          </div>

          <div className="risk-rule">
            <span className="action-color-dot" style={{ background: "#F47781" }} />
            <span>80-100% - Highest predicted probability</span>
          </div>

          <p className="muted-note">
            Approve, review and block decisions are determined separately
            by the expected-cost decision policy. A probability band does
            not independently determine an action.
          </p>
        </div>
      </section>

      <section className="panel action-distribution-panel">
        <div className="panel-heading">
          <div>
            <h3>Recommended action distribution</h3>
            <p>
              Decisions across {totalPredictions.toLocaleString()} predictions
              in the selected batch
            </p>
          </div>
        </div>

        <div className="action-distribution-list">
          {actionDistribution.map(({ action, count, percentage }) => (
            <div className="action-distribution-item" key={action}>
              <div className="action-distribution-info">
                <span>
                  <span
                    className="action-color-dot"
                    style={{ backgroundColor: ACTION_COLORS[action] }}
                  />
                  {action.charAt(0).toUpperCase() + action.slice(1)}
                </span>
                <strong>
                  {count.toLocaleString()} ({percentage.toFixed(1)}%)
                </strong>
              </div>

              <div className="action-distribution-track">
                <div
                  className="action-distribution-fill"
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: ACTION_COLORS[action],
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel transactions-panel">
        <div className="panel-heading">
          <div>
            <h3>Highest predicted probabilities</h3>
            <p>
              Showing {topPredictions.length} of{" "}
              {actionFilter === "all"
                ? totalPredictions
                : actionCounts[actionFilter]}{" "}
              matching records, sorted by calibrated probability
            </p>
          </div>

          <button
            type="button"
            className="clear-filters-button"
            onClick={() => loadBatches(true)}
            disabled={refreshing}
          >
            <RefreshCw size={14} />
            {refreshing ? "Refreshing..." : "Refresh"}
          </button>
        </div>

        <div className="risk-filter-toolbar">
          <div className="risk-filter-label">
            <Filter size={16} />
            Filter by recommended action
          </div>

          <select
            value={actionFilter}
            onChange={(event) => setActionFilter(event.target.value)}
            aria-label="Filter predictions by action"
          >
            <option value="all">All decisions</option>
            <option value="approve">Approve</option>
            <option value="review">Review</option>
            <option value="block">Block</option>
          </select>

          {actionFilter !== "all" && (
            <button
              type="button"
              className="clear-filters-button"
              onClick={() => setActionFilter("all")}
            >
              <X size={14} />
              Clear filter
            </button>
          )}
        </div>

        {loading ? (
          <div className="empty-state">Loading prediction batches...</div>
        ) : !selectedBatchId ? (
          <div className="empty-state">
            No persisted prediction batches are available yet.
          </div>
        ) : (
          <div className="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>TRANSACTION ID</th>
                  <th>CALIBRATED PROBABILITY</th>
                  <th>RAW PROBABILITY</th>
                  <th>EXPECTED COST</th>
                  <th>RECOMMENDED ACTION</th>
                  <th>PREDICTED AT</th>
                </tr>
              </thead>

              <tbody>
                {topPredictions.map((prediction) => (
                  <tr key={prediction.prediction_id}>
                    <td className="transaction-id">
                      {prediction.TransactionID ?? "?"}
                    </td>
                    <td>{formatProbability(prediction.calibrated_probability)}</td>
                    <td>{formatProbability(prediction.raw_probability)}</td>
                    <td>{formatCost(prediction.expected_cost)}</td>
                    <td>
                      <span className={`action-label ${prediction.action}`}>
                        {prediction.action}
                      </span>
                    </td>
                    <td>{formatDate(prediction.predicted_at_utc)}</td>
                  </tr>
                ))}

                {topPredictions.length === 0 && (
                  <tr>
                    <td colSpan="6" className="empty-state">
                      No predictions match this decision filter.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <p className="muted-note">
        Analysis uses full-batch aggregates and the highest-risk records from
        the selected prediction batch. Probability scores are model estimates,
        not confirmed fraud outcomes. Original transaction features and
        amounts are not stored in the prediction ledger.
      </p>
    </div>
  );
}

export default RiskAnalysis;
