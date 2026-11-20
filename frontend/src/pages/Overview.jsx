import {
  Activity,
  AlertTriangle,
  RefreshCw,
  ShieldCheck,
  ShieldAlert,
  Clock3,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";
import { useMonitoringSnapshots } from "../hooks/useMonitoringSnapshots";

const ACTIONS = [
  { key: "approve", label: "Approve", color: "#269D83" },
  { key: "review", label: "Review", color: "#E8A642" },
  { key: "block", label: "Block", color: "#F04452" },
];

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(Number(value))) return "—";

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatTimestamp(value) {
  if (!value) return "Not provided";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}

function Overview() {
  const { snapshots, loading, error, refresh } =
    useMonitoringSnapshots(20);

  const latest = snapshots[0] ?? null;
  const prediction = latest?.prediction_summary;
  const drift = latest?.drift_summary;

  const actionCounts = prediction?.action_counts ?? {};
  const totalActions = Object.values(actionCounts).reduce(
    (total, count) => total + Number(count || 0),
    0,
  );

  const topDriftingFeatures = (drift?.features ?? []).slice(0, 5);

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / OVERVIEW"
        title={<>DriftSentinel <span>✦</span></>}
        description="Monitor model predictions, decision distributions and feature drift."
      >
        <button
          type="button"
          className="secondary-button"
          onClick={refresh}
          disabled={loading}
        >
          <RefreshCw size={15} />
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </PageHeader>

      {error && (
        <section className="panel" role="alert">
          <div className="panel-heading">
            <h3>Monitoring data unavailable</h3>
            <AlertTriangle color="#F04452" />
          </div>
          <p>{error}</p>
          <p className="muted-note">
            Check that the FastAPI service and monitoring database are available.
          </p>
        </section>
      )}

      {loading && !latest && (
        <section className="panel">
          <p>Loading monitoring snapshots...</p>
        </section>
      )}

      {!loading && !error && !latest && (
        <section className="panel">
          <div className="panel-heading">
            <div>
              <h3>No monitoring snapshots yet</h3>
              <p>
                The model and API are available, but no monitoring batches
                have been persisted in the current database.
              </p>
            </div>
            <Activity size={22} />
          </div>
          <p className="muted-note">
            Submit a batch through the monitoring API to populate this dashboard.
            Predictions and drift metrics will appear here after the batch is saved.
          </p>
        </section>
      )}

      {latest && (
        <>
          <section className="panel">
            <div className="panel-heading">
              <div>
                <h3>Latest monitoring snapshot</h3>
                <p>
                  Snapshot ID: <span className="transaction-id">{latest.snapshot_id}</span>
                </p>
              </div>
              <span className="chart-period">
                {formatTimestamp(latest.created_at_utc)}
              </span>
            </div>
            <p className="muted-note">
              {formatNumber(latest.prediction_count, 0)} transactions evaluated
              in this batch
            </p>
          </section>

          <section className="metrics-grid">
            <div className="panel metric-card">
              <div className="metric-top">
                <span>Predictions in batch</span>
                <Activity size={19} />
              </div>
              <strong className="metric-value">
                {formatNumber(prediction?.prediction_count, 0)}
              </strong>
              <span className="summary-label">Latest snapshot</span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Mean calibrated probability</span>
                <ShieldCheck size={19} />
              </div>
              <strong className="metric-value">
                {formatPercent(prediction?.calibrated_probability?.mean)}
              </strong>
              <span className="summary-label">Average model score</span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Mean feature PSI</span>
                <Activity size={19} />
              </div>
              <strong className="metric-value">
                {formatNumber(drift?.mean_psi)}
              </strong>
              <span className="summary-label">
                Across {formatNumber(drift?.feature_count, 0)} features
              </span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Features with PSI above 0.25</span>
                <AlertTriangle size={19} />
              </div>
              <strong className="metric-value">
                {formatNumber(drift?.features_psi_above_0_25, 0)}
              </strong>
              <span className="summary-label">High drift screening threshold</span>
            </div>
          </section>

          <section className="charts-grid">
            <div className="panel action-distribution-panel">
              <div className="panel-heading">
                <div>
                  <h3>Recommended action distribution</h3>
                  <p>Actions assigned by the decision policy</p>
                </div>
                <ShieldAlert size={20} />
              </div>

              <div className="action-distribution-list">
                {ACTIONS.map((action) => {
                  const count = Number(actionCounts[action.key] ?? 0);
                  const percentage =
                    totalActions > 0 ? (count / totalActions) * 100 : 0;

                  return (
                    <div className="action-distribution-item" key={action.key}>
                      <div className="action-distribution-info">
                        <span>
                          <i
                            className="action-color-dot"
                            style={{ backgroundColor: action.color }}
                          />
                          {action.label}
                        </span>
                        <strong>
                          {formatNumber(count, 0)} ({percentage.toFixed(1)}%)
                        </strong>
                      </div>
                      <div className="action-distribution-track">
                        <div
                          className="action-distribution-fill"
                          style={{
                            width: `${percentage}%`,
                            backgroundColor: action.color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="panel analytics-panel">
              <div className="panel-heading">
                <div>
                  <h3>Prediction summary</h3>
                  <p>Calibrated model score distribution</p>
                </div>
              </div>

              <div className="transaction-details-grid">
                <div className="detail-item">
                  <span>Mean probability</span>
                  <strong>
                    {formatPercent(prediction?.calibrated_probability?.mean)}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Median probability</span>
                  <strong>
                    {formatPercent(prediction?.calibrated_probability?.median)}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>90th percentile</span>
                  <strong>
                    {formatPercent(
                      prediction?.calibrated_probability?.quantiles?.p90,
                    )}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Mean expected cost</span>
                  <strong>
                    {formatNumber(prediction?.expected_cost?.mean)}
                  </strong>
                </div>
              </div>
            </div>
          </section>

          <section className="panel transactions-panel">
            <div className="panel-heading">
              <div>
                <h3>Top drifting features</h3>
                <p>
                  Highest population stability index values in the latest batch
                </p>
              </div>
              <Clock3 size={20} />
            </div>

            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>FEATURE</th>
                    <th>TYPE</th>
                    <th>PSI</th>
                    <th>REFERENCE MISSING</th>
                    <th>CURRENT MISSING</th>
                  </tr>
                </thead>
                <tbody>
                  {topDriftingFeatures.map((feature) => (
                    <tr key={feature.feature}>
                      <td className="transaction-id">{feature.feature}</td>
                      <td>{feature.feature_type}</td>
                      <td>{formatNumber(feature.psi, 4)}</td>
                      <td>{formatPercent(feature.reference_missing_rate)}</td>
                      <td>{formatPercent(feature.current_missing_rate)}</td>
                    </tr>
                  ))}

                  {topDriftingFeatures.length === 0 && (
                    <tr>
                      <td colSpan="5" className="empty-state">
                        No feature drift details are available.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <p className="muted-note">
            PSI is a feature-distribution screening measure. Elevated PSI does
            not by itself establish model performance degradation or concept drift.
            Predicted actions are not confirmed fraud outcomes.
          </p>
        </>
      )}

      <footer className="dashboard-footer">
        <span>DriftSentinel · Fraud Intelligence Platform</span>
        <span>Monitoring · Persisted snapshot data</span>
      </footer>
    </div>
  );
}

export default Overview;
