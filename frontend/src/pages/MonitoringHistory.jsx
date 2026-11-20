import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Clock3,
  RefreshCw,
  ShieldCheck,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import PageHeader from "../components/common/PageHeader";
import { useMonitoringSnapshots } from "../hooks/useMonitoringSnapshots";

const ACTIONS = [
  { key: "approve", label: "Approve", color: "#269D83" },
  { key: "review", label: "Review", color: "#E8A642" },
  { key: "block", label: "Block", color: "#F04452" },
];

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(Number(value))) return "—";

  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
  });
}

function formatPercent(value) {
  if (!Number.isFinite(Number(value))) return "—";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatDate(value) {
  if (!value) return "Timestamp unavailable";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}

function shortId(value) {
  return value ? `${value.slice(0, 8)}…` : "—";
}

function MonitoringHistory() {
  const { snapshots, loading, error, refresh } =
    useMonitoringSnapshots(100);

  const [selectedId, setSelectedId] = useState(null);

  const chronological = useMemo(
    () =>
      [...snapshots]
        .filter((snapshot) => snapshot.created_at_utc)
        .reverse()
        .map((snapshot, index) => ({
          ...snapshot,
          sequence: index + 1,
          label: new Date(snapshot.created_at_utc).toLocaleString(
            undefined,
            { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" },
          ),
          meanPsi: Number(snapshot.drift_summary?.mean_psi ?? 0),
          meanProbability: Number(
            snapshot.prediction_summary?.calibrated_probability?.mean ?? 0,
          ),
        })),
    [snapshots],
  );

  const selected =
    snapshots.find((snapshot) => snapshot.snapshot_id === selectedId) ??
    snapshots[0] ??
    null;

  const selectedActions = selected?.prediction_summary?.action_counts ?? {};
  const selectedDrift = selected?.drift_summary;
  const selectedPrediction = selected?.prediction_summary;

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / MONITORING"
        title={<>Monitoring <span>History</span></>}
        description="Explore persisted monitoring snapshots and compare model behaviour over time."
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
            <h3>Monitoring history unavailable</h3>
            <AlertTriangle color="#F04452" />
          </div>
          <p>{error}</p>
        </section>
      )}

      {loading && snapshots.length === 0 && (
        <section className="panel">
          <p>Loading persisted monitoring snapshots...</p>
        </section>
      )}

      {!loading && !error && snapshots.length === 0 && (
        <section className="panel history-empty">
          <Activity size={30} />
          <h3>No monitoring history yet</h3>
          <p>
            Persisted monitoring snapshots will appear here after batches
            are submitted through the monitoring API.
          </p>
        </section>
      )}

      {snapshots.length > 0 && (
        <>
          <section className="metrics-grid history-metrics">
            <div className="panel metric-card">
              <div className="metric-top">
                <span>Stored snapshots</span>
                <Clock3 size={19} />
              </div>
              <strong className="metric-value">{snapshots.length}</strong>
              <span className="summary-label">Persisted records</span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Latest batch size</span>
                <Activity size={19} />
              </div>
              <strong className="metric-value">
                {formatNumber(snapshots[0]?.prediction_count, 0)}
              </strong>
              <span className="summary-label">Transactions evaluated</span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Latest mean PSI</span>
                <AlertTriangle size={19} />
              </div>
              <strong className="metric-value">
                {formatNumber(snapshots[0]?.drift_summary?.mean_psi)}
              </strong>
              <span className="summary-label">Feature drift screening</span>
            </div>

            <div className="panel metric-card">
              <div className="metric-top">
                <span>Latest mean probability</span>
                <ShieldCheck size={19} />
              </div>
              <strong className="metric-value">
                {formatPercent(
                  snapshots[0]?.prediction_summary?.calibrated_probability?.mean,
                )}
              </strong>
              <span className="summary-label">Calibrated model score</span>
            </div>
          </section>

          <section className="history-chart-grid">
            <div className="panel analytics-panel">
              <div className="panel-heading">
                <div>
                  <h3>Mean feature PSI</h3>
                  <p>Reference-relative feature distribution shift</p>
                </div>
              </div>
              <div className="history-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chronological}>
                    <CartesianGrid stroke="#252d3a" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="label"
                      stroke="#8992a5"
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis stroke="#8992a5" tick={{ fontSize: 10 }} />
                    <Tooltip
                      contentStyle={{
                        background: "#11141c",
                        border: "1px solid #30394a",
                        borderRadius: 8,
                        color: "#f2f0ea",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="meanPsi"
                      name="Mean PSI"
                      stroke="#E8A642"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="panel analytics-panel">
              <div className="panel-heading">
                <div>
                  <h3>Mean calibrated probability</h3>
                  <p>Average prediction score per monitoring batch</p>
                </div>
              </div>
              <div className="history-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chronological}>
                    <CartesianGrid stroke="#252d3a" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="label"
                      stroke="#8992a5"
                      tick={{ fontSize: 10 }}
                    />
                    <YAxis
                      stroke="#8992a5"
                      tick={{ fontSize: 10 }}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                    />
                    <Tooltip
                      formatter={(value) => formatPercent(value)}
                      contentStyle={{
                        background: "#11141c",
                        border: "1px solid #30394a",
                        borderRadius: 8,
                        color: "#f2f0ea",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="meanProbability"
                      name="Mean calibrated probability"
                      stroke="#7589E8"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>

          {snapshots.length < 2 && (
            <p className="muted-note history-note">
              Only one persisted snapshot is currently available. Trend
              charts show the observed batch; additional snapshots are
              required for meaningful historical comparisons.
            </p>
          )}

          <section className="panel transactions-panel">
            <div className="panel-heading">
              <div>
                <h3>Snapshot history</h3>
                <p>Select a snapshot to inspect its recorded metrics</p>
              </div>
              <Clock3 size={20} />
            </div>

            <div className="table-wrapper">
              <table>
                <thead>
                  <tr>
                    <th>SNAPSHOT</th>
                    <th>CREATED</th>
                    <th>TRANSACTIONS</th>
                    <th>MEAN PSI</th>
                    <th>PSI ABOVE 0.25</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshots.map((snapshot) => (
                    <tr
                      key={snapshot.snapshot_id}
                      className={
                        selected?.snapshot_id === snapshot.snapshot_id
                          ? "history-selected-row"
                          : ""
                      }
                    >
                      <td>
                        <button
                          type="button"
                          className="history-select-button"
                          onClick={() => setSelectedId(snapshot.snapshot_id)}
                        >
                          {shortId(snapshot.snapshot_id)}
                        </button>
                      </td>
                      <td>{formatDate(snapshot.created_at_utc)}</td>
                      <td>{formatNumber(snapshot.prediction_count, 0)}</td>
                      <td>{formatNumber(snapshot.drift_summary?.mean_psi)}</td>
                      <td>
                        {formatNumber(
                          snapshot.drift_summary?.features_psi_above_0_25,
                          0,
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {selected && (
            <section className="panel history-detail-panel">
              <div className="panel-heading">
                <div>
                  <h3>Selected snapshot details</h3>
                  <p className="transaction-id">{selected.snapshot_id}</p>
                </div>
                <span className="chart-period">
                  {formatDate(selected.created_at_utc)}
                </span>
              </div>

              <div className="transaction-details-grid">
                <div className="detail-item">
                  <span>Predictions</span>
                  <strong>{formatNumber(selected.prediction_count, 0)}</strong>
                </div>
                <div className="detail-item">
                  <span>Features evaluated</span>
                  <strong>{formatNumber(selectedDrift?.feature_count, 0)}</strong>
                </div>
                <div className="detail-item">
                  <span>Mean calibrated probability</span>
                  <strong>
                    {formatPercent(selectedPrediction?.calibrated_probability?.mean)}
                  </strong>
                </div>
                <div className="detail-item">
                  <span>Mean expected cost</span>
                  <strong>
                    {formatNumber(selectedPrediction?.expected_cost?.mean)}
                  </strong>
                </div>
              </div>

              <h4 className="history-subheading">Decision distribution</h4>
              <div className="action-distribution-list">
                {ACTIONS.map((action) => {
                  const count = Number(selectedActions[action.key] ?? 0);
                  const total = Number(selected.prediction_count ?? 0);
                  const percentage = total > 0 ? (count / total) * 100 : 0;

                  return (
                    <div key={action.key}>
                      <div className="action-distribution-info">
                        <span>{action.label}</span>
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

              <h4 className="history-subheading">Snapshot metadata</h4>
              <pre className="history-metadata">
                {JSON.stringify(selected.metadata ?? {}, null, 2)}
              </pre>
            </section>
          )}

          <p className="muted-note history-note">
            Historical trends describe recorded batch summaries. PSI is a
            distribution-shift screening measure and does not independently
            establish model performance degradation or concept drift.
            Predicted actions are not confirmed fraud outcomes.
          </p>
        </>
      )}

      <footer className="dashboard-footer">
        <span>DriftSentinel · Fraud Intelligence Platform</span>
        <span>Monitoring · Persisted snapshot history</span>
      </footer>
    </div>
  );
}

export default MonitoringHistory;
