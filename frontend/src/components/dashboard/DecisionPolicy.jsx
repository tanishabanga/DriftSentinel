
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import decisionPolicyData from "../../data/decisionPolicyData.json";

const scenarios = Object.entries(decisionPolicyData).map(
  ([value, scenario]) => ({
    value,
    label: scenario.label,
  }),
);

function DecisionPolicy() {
  const [scenario, setScenario] = useState("baseline");

  const data = decisionPolicyData[scenario];

  const chartData = useMemo(
    () =>
      data.sigmoid.map((item, index) => ({
        capacity: item.capacity,
        sigmoidCost: item.realizedCost,
        uncalibratedCost: data.uncalibrated[index].realizedCost,
        expectedCost: item.expectedCost,
      })),
    [data],
  );

  const selected = data.sigmoid[data.sigmoid.length - 1];


  return (
    <section className="decision-policy-section">
      <div className="decision-heading">
        <div>
          <span className="section-eyebrow">COST-SENSITIVE EVALUATION</span>
          <h3>Decision policy analysis</h3>
          <p>
            Compare review capacity and transaction outcomes using
            calibrated and uncalibrated probabilities.
          </p>
        </div>

        <select
          className="scenario-select"
          value={scenario}
          onChange={(event) => setScenario(event.target.value)}
          aria-label="Select cost scenario"
        >
          {scenarios.map((item) => (
            <option key={item.value} value={item.value}>
              {item.label}
            </option>
          ))}
        </select>
      </div>

      <div className="decision-summary-grid">
        <div className="panel decision-summary-card">
          <span>Realized cost at 5% capacity</span>
          <strong>{selected.realizedCost.toLocaleString("en-IN")}</strong>
          <small>Sigmoid calibrated</small>
        </div>

        <div className="panel decision-summary-card">
          <span>Transactions reviewed</span>
          <strong>{selected.reviewCount.toLocaleString("en-IN")}</strong>
          <small>5% review capacity · 44,291 transactions</small>
        </div>

        <div className="panel decision-summary-card">
          <span>Fraud approved</span>
          <strong>{selected.fraudApproved}</strong>
          <small>At 5% review capacity</small>
        </div>

        <div className="panel decision-summary-card">
          <span>Legitimate transactions blocked</span>
          <strong>{selected.legitimateBlocked}</strong>
          <small>At 5% review capacity</small>
        </div>
      </div>

      <div className="panel analytics-panel decision-chart-panel">
        <div className="panel-heading">
          <div>
            <h3>Realized cost by review capacity</h3>
            <p>Sigmoid vs uncalibrated probabilities</p>
          </div>
        </div>

        <div className="analytics-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid stroke="#252B38" strokeDasharray="3 5" />
              <XAxis dataKey="capacity" stroke="#8992A5" tickLine={false} />
              <YAxis
                stroke="#8992A5"
                tickLine={false}
                tickFormatter={(value) => value.toLocaleString("en-IN")}
              />
              <Tooltip
                formatter={(value) => value.toLocaleString("en-IN")}
                contentStyle={{
                  background: "#171B25",
                  border: "1px solid #303747",
                  borderRadius: 10,
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="sigmoidCost"
                name="Sigmoid"
                stroke="#269D83"
                strokeWidth={2.5}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="uncalibratedCost"
                name="Uncalibrated"
                stroke="#8295E2"
                strokeWidth={2.5}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel analytics-panel decision-chart-panel">
        <div className="panel-heading">
          <div>
            <h3>Expected vs realized cost</h3>
            <p>Sigmoid-calibrated policy across review capacities</p>
          </div>
        </div>

        <div className="analytics-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid stroke="#252B38" strokeDasharray="3 5" />
              <XAxis dataKey="capacity" stroke="#8992A5" tickLine={false} />
              <YAxis
                stroke="#8992A5"
                tickLine={false}
                tickFormatter={(value) => value.toLocaleString("en-IN")}
              />
              <Tooltip
                formatter={(value) => value.toLocaleString("en-IN")}
                contentStyle={{
                  background: "#171B25",
                  border: "1px solid #303747",
                  borderRadius: 10,
                }}
              />
              <Legend />
              <Bar
                dataKey="expectedCost"
                name="Expected cost"
                fill="#8295E2"
                radius={[4, 4, 0, 0]}
              />
              <Bar
                dataKey="sigmoidCost"
                name="Realized cost"
                fill="#269D83"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="panel decision-interpretation">
        <h3>What the experiment indicates</h3>
        <ul>
          <li>
            Under the baseline cost assumptions, increasing review capacity
            reduced realized cost across the tested capacity-constrained
            policies.
          </li>
          <li>
            At 5% capacity, sigmoid calibration produced a realized cost of
            62,463 compared with 63,244 for uncalibrated probabilities.
          </li>
          <li>
            Review capacity is not a universal optimum. Different cost
            assumptions can produce different preferred policies.
          </li>
        </ul>

        <p className="muted-note">
          Costs are illustrative units, not actual currency. Results are
          from the chronological validation evaluation split, not the
          untouched test set. The unconstrained policy is excluded from
          these capacity comparisons.
        </p>
      </div>
    </section>
  );
}

export default DecisionPolicy;
