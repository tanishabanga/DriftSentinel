
import {
  Pie,
  PieChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

import { riskData } from "../../data/mockData";

function RiskDistribution() {
  return (
    <div className="panel risk-panel">
      <div className="panel-heading">
        <div>
          <h3>Risk distribution</h3>
          <p>Transactions by predicted risk level</p>
        </div>
      </div>

      <div className="risk-chart">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={riskData}
              dataKey="value"
              nameKey="name"
              innerRadius={68}
              outerRadius={91}
              paddingAngle={4}
              stroke="none"
            >
              {riskData.map((item) => (
                <Cell key={item.name} fill={item.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{
                background: "#171B25",
                border: "1px solid #303747",
                borderRadius: 10,
              }}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="risk-center">
          <strong>39.7K</strong>
          <span>Transactions</span>
        </div>
      </div>

      <div className="risk-legend">
        {riskData.map((item) => (
          <div className="risk-legend-item" key={item.name}>
            <span>
              <i style={{ background: item.color }} />
              {item.name}
            </span>
            <strong>{item.value}%</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default RiskDistribution;
