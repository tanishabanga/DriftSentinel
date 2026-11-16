
import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  Filter,
  X,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";
import RiskDistribution from "../components/common/RiskDistribution";
import RiskBadge from "../components/common/RiskBadge";
import { transactions } from "../data/mockData";

function RiskAnalysis() {
  const [actionFilter, setActionFilter] = useState("All");

  const highRisk = transactions.filter((t) => t.risk === "High");
  const mediumRisk = transactions.filter((t) => t.risk === "Medium");
  const lowRisk = transactions.filter((t) => t.risk === "Low");

  const actionCounts = {
    Approve: transactions.filter((t) => t.action === "Approve").length,
    Review: transactions.filter((t) => t.action === "Review").length,
    Block: transactions.filter((t) => t.action === "Block").length,
  };

  const filteredHighRisk = useMemo(() => {
    return highRisk.filter(
      (t) => actionFilter === "All" || t.action === actionFilter
    );
  }, [highRisk, actionFilter]);

  const actionColors = {
    Approve: "#269D83",
    Review: "#E8A642",
    Block: "#F47781",
  };

  const totalTransactions = transactions.length;

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / RISK ANALYSIS"
        title="Risk analysis"
        description="Understand transaction risk levels and review potentially fraudulent activity."
      />

      <section className="metrics-grid">
        <div className="panel metric-card">
          <div className="metric-top">
            <span>High-risk records</span>
            <AlertTriangle color="#F47781" />
          </div>
          <strong className="metric-value">{highRisk.length}</strong>
          <span className="summary-label">In displayed sample</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Medium-risk records</span>
            <ShieldAlert color="#E8A642" />
          </div>
          <strong className="metric-value">{mediumRisk.length}</strong>
          <span className="summary-label">In displayed sample</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Low-risk records</span>
            <ShieldCheck color="#269D83" />
          </div>
          <strong className="metric-value">{lowRisk.length}</strong>
          <span className="summary-label">In displayed sample</span>
        </div>

        <div className="panel metric-card">
          <div className="metric-top">
            <span>Manual reviews</span>
            <ShieldAlert color="#E8A642" />
          </div>
          <strong className="metric-value">
            {actionCounts.Review}
          </strong>
          <span className="summary-label">In displayed sample</span>
        </div>
      </section>

      <section className="charts-grid">
        <RiskDistribution />

        <div className="panel risk-explanation">
          <h3>Risk interpretation</h3>
          <p>Risk labels shown here are illustrative UI categories.</p>

          <div className="risk-rule">
            <RiskBadge risk="Low" />
            <span>Lower predicted fraud probability</span>
          </div>

          <div className="risk-rule">
            <RiskBadge risk="Medium" />
            <span>Intermediate predicted fraud probability</span>
          </div>

          <div className="risk-rule">
            <RiskBadge risk="High" />
            <span>Higher predicted fraud probability</span>
          </div>

          <p className="muted-note">
            Production risk bands will be configured using the selected
            decision policy, not assumed from these demo labels.
          </p>
        </div>
      </section>

      {/* Recommended action distribution */}
      <section className="panel action-distribution-panel">
        <div className="panel-heading">
          <div>
            <h3>Recommended action distribution</h3>
            <p>Breakdown of suggested actions across demo transactions</p>
          </div>
        </div>

        <div className="action-distribution-list">
          {Object.entries(actionCounts).map(([action, count]) => {
            const percentage =
              totalTransactions > 0
                ? (count / totalTransactions) * 100
                : 0;

            return (
              <div className="action-distribution-item" key={action}>
                <div className="action-distribution-info">
                  <span>
                    <span
                      className="action-color-dot"
                      style={{ backgroundColor: actionColors[action] }}
                    />
                    {action}
                  </span>
                  <strong>
                    {count} ({percentage.toFixed(0)}%)
                  </strong>
                </div>

                <div className="action-distribution-track">
                  <div
                    className="action-distribution-fill"
                    style={{
                      width: `${percentage}%`,
                      backgroundColor: actionColors[action],
                    }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* High-risk transactions */}
      <section className="panel transactions-panel">
        <div className="panel-heading">
          <div>
            <h3>High-risk transactions</h3>
            <p>
              Showing {filteredHighRisk.length} of {highRisk.length} high-risk
              demo records
            </p>
          </div>
        </div>

        <div className="risk-filter-toolbar">
          <div className="risk-filter-label">
            <Filter size={16} />
            Filter by recommended action
          </div>

          <select
            value={actionFilter}
            onChange={(event) => setActionFilter(event.target.value)}
            aria-label="Filter high-risk transactions by action"
          >
            <option value="All">All actions</option>
            <option value="Approve">Approve</option>
            <option value="Review">Review</option>
            <option value="Block">Block</option>
          </select>

          {actionFilter !== "All" && (
            <button
              type="button"
              className="clear-filters-button"
              onClick={() => setActionFilter("All")}
            >
              <X size={14} />
              Clear filter
            </button>
          )}
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>TRANSACTION ID</th>
                <th>AMOUNT</th>
                <th>FRAUD PROBABILITY</th>
                <th>LOCATION</th>
                <th>RECOMMENDED ACTION</th>
              </tr>
            </thead>

            <tbody>
              {filteredHighRisk.map((transaction) => (
                <tr key={transaction.id}>
                  <td className="transaction-id">{transaction.id}</td>
                  <td>{transaction.amount}</td>
                  <td>{transaction.probability}</td>
                  <td>{transaction.location}</td>
                  <td>
                    <span
                      className={`action-label ${transaction.action.toLowerCase()}`}
                    >
                      {transaction.action}
                    </span>
                  </td>
                </tr>
              ))}

              {filteredHighRisk.length === 0 && (
                <tr>
                  <td colSpan="5" className="empty-state">
                    No high-risk transactions match this action.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default RiskAnalysis;
