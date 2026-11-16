
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { trendData } from "../../data/mockData";

function TransactionChart() {
  return (
    <div className="panel trend-panel">
      <div className="panel-heading">
        <div>
          <h3>Transaction activity</h3>
          <p>Transaction volume and detected fraud over time</p>
        </div>
        <span className="chart-period">Weekly</span>
      </div>

      <div className="chart-legend">
        <span><i className="legend-dot transactions-dot" /> Transactions</span>
        <span><i className="legend-dot fraud-dot" /> Fraud detected</span>
      </div>

      <div className="trend-chart">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={trendData}>
            <defs>
              <linearGradient id="transactionFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7188D8" stopOpacity={0.22} />
                <stop offset="95%" stopColor="#7188D8" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#F04452" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#F04452" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid stroke="#252B38" strokeDasharray="3 5" vertical={false} />
            <XAxis dataKey="day" stroke="#8992A5" tickLine={false} axisLine={false} />
            <YAxis stroke="#8992A5" tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{
                background: "#171B25",
                border: "1px solid #303747",
                borderRadius: 10,
              }}
            />
            <Area
              type="monotone"
              dataKey="transactions"
              stroke="#8295E2"
              fill="url(#transactionFill)"
              strokeWidth={2.5}
            />
            <Area
              type="monotone"
              dataKey="fraud"
              stroke="#F04452"
              fill="url(#fraudFill)"
              strokeWidth={2.5}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default TransactionChart;
