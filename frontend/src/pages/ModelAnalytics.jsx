
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  ReferenceLine,
} from "recharts";
import DecisionPolicy from "../components/dashboard/DecisionPolicy";
import PageHeader from "../components/common/PageHeader";
import modelAnalyticsData from "../data/modelAnalyticsData.json";

function ModelAnalytics() {
  const metrics = [
    ["Average precision", modelAnalyticsData.metrics.averagePrecision.toFixed(4)],
    ["ROC-AUC", modelAnalyticsData.metrics.rocAuc.toFixed(4)],
    ["Precision", modelAnalyticsData.metrics.precision.toFixed(4)],
    ["Recall", modelAnalyticsData.metrics.recall.toFixed(4)],
    ["F1 score", modelAnalyticsData.metrics.f1Score.toFixed(4)],
    ["Brier score", modelAnalyticsData.metrics.brierScore.toFixed(4)],
    ["Log loss", modelAnalyticsData.metrics.logLoss.toFixed(4)],
  ];

  return (
    
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / MODEL ANALYTICS"
        title="Model analytics"
        description="Explore validation performance, probability calibration and threshold trade-offs."
      />

      <div className="model-info-strip">
        <span><i className="status-dot" /> Model: XGBoost</span>
        <span>Dataset: IEEE-CIS Fraud Detection</span>
        <span>Evaluation: Validation data</span>
      </div>

      <section className="metrics-grid model-metrics-grid">
        {metrics.map(([label, value]) => (
          <div className="panel metric-card" key={label}>
            <span className="summary-label">{label}</span>
            <strong className="metric-value">{value}</strong>
          </div>
        ))}
      </section>

      <section className="charts-grid analytics-charts">
        <div className="panel analytics-panel">
          <div className="panel-heading">
            <div>
              <h3>Precision vs recall</h3>
              <p>Actual XGBoost validation results across thresholds</p>
            </div>
          </div>

          <div className="analytics-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelAnalyticsData.thresholdData}>
                <CartesianGrid stroke="#252B38" strokeDasharray="3 5" />
                <XAxis
                  type="number"
                  dataKey="threshold"
                  stroke="#8992A5"
                  tickLine={false}
                  domain={[0.05, 0.95]}
                  tickFormatter={(value) => value.toFixed(2)}
                  label={{ value: "Threshold", position: "insideBottom", offset: -5 }}
                />
                <YAxis
                  stroke="#8992A5"
                  domain={[0, 1]}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "#171B25",
                    border: "1px solid #303747",
                    borderRadius: 10,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="precision"
                  name="Precision"
                  stroke="#8295E2"
                  strokeWidth={2.5}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="recall"
                  name="Recall"
                  stroke="#F04452"
                  strokeWidth={2.5}
                  dot={false}
                />
                <ReferenceLine x={0.25} stroke="#E8A642" strokeDasharray="4 4" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel analytics-panel">
          <div className="panel-heading">
            <div>
              <h3>Calibration curve</h3>
              <p>Sigmoid calibration | chronological evaluation split</p>
            </div>
          </div>

          <div className="analytics-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={modelAnalyticsData.calibrationData}>
                <CartesianGrid stroke="#252B38" strokeDasharray="3 5" />
                <XAxis
                  type="number"
                  dataKey="predicted"
                  stroke="#8992A5"
                  tickLine={false}
                  domain={[0, 1]}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <YAxis
                  stroke="#8992A5"
                  tickLine={false}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    background: "#171B25",
                    border: "1px solid #303747",
                    borderRadius: 10,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="observed"
                  name="Observed"
                  stroke="#269D83"
                  strokeWidth={2.5}
                />
                <Line
                  type="linear"
                  dataKey="predicted"
                  name="Perfect calibration"
                  stroke="#8992A5"
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="panel analytics-panel model-notes">
        <h3>Model evaluation notes</h3>
        <p>
          The displayed validation metrics are from the completed XGBoost
          experiment. Average precision is 0.5854 and ROC-AUC is 0.9172.
        </p>
        <p>
          The current calibration experiment found a small Brier score
          improvement with sigmoid calibration, while log loss was slightly
          worse. Calibration choice should therefore be interpreted using
          multiple metrics.
        </p>
        <p className="muted-note">
  Threshold analysis uses all 88,581 validation transactions.
  Calibration was fitted on 44,290 earlier chronological transactions and evaluated on a separate 44,291-transaction partition. Calibration bins have unequal
  sample counts, so points with fewer observations are less stable.
  The final untouched test set has not been evaluated.
</p>
      </section>
      <DecisionPolicy />
    </div>
  );
}

export default ModelAnalytics;
