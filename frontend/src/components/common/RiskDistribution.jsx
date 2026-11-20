import {
  Pie,
  PieChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const BANDS = [
  { name: "0-40% Lower", key: "lower", color: "#269D83" },
  { name: "40-80% Elevated", key: "elevated", color: "#E8A642" },
  { name: "80-100% Highest", key: "highest", color: "#F47781" },
];

function RiskDistribution({ counts = {} }) {
  const data = BANDS.map((band) => ({
    name: band.name,
    value: Number(counts[band.key]) || 0,
    color: band.color,
  }));

  const total = data.reduce((sum, item) => sum + item.value, 0);

  return (
    <div className="panel risk-panel">
      <div className="panel-heading">
        <div>
          <h3>Probability distribution</h3>
          <p>Full-batch predictions grouped by calibrated probability</p>
        </div>
      </div>

      <div className="risk-chart">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={68}
              outerRadius={91}
              paddingAngle={4}
              stroke="none"
            >
              {data.map((item) => (
                <Cell key={item.name} fill={item.color} />
              ))}
            </Pie>

            <Tooltip
              contentStyle={{
                background: "#171B25",
                border: "1px solid #303747",
                borderRadius: 10,
              }}
              formatter={(value, name) => [value, name]}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="risk-center">
          <strong>{total.toLocaleString()}</strong>
          <span>Predictions</span>
        </div>
      </div>

      <div className="risk-legend">
        {data.map((item) => (
          <div className="risk-legend-item" key={item.name}>
            <span>
              <i style={{ background: item.color }} />
              {item.name}
            </span>
            <strong>
              {item.value.toLocaleString()} (
              {total > 0 ? ((item.value / total) * 100).toFixed(1) : "0.0"}%)
            </strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RiskDistribution;
