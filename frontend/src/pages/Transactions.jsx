
import { useMemo, useState } from "react";
import { Search, X, ExternalLink, ShieldCheck, ShieldAlert } from "lucide-react";
import { useSearchParams } from "react-router-dom";

import PageHeader from "../components/common/PageHeader";
import RiskBadge from "../components/common/RiskBadge";
import { transactions } from "../data/mockData";

function Transactions() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [riskFilter, setRiskFilter] = useState("All");
  const [actionFilter, setActionFilter] = useState("All");
  const [selectedTransaction, setSelectedTransaction] = useState(null);

  const search = searchParams.get("search") || "";

  function updateSearch(value) {
    const nextParams = new URLSearchParams(searchParams);

    if (value.trim()) {
      nextParams.set("search", value);
    } else {
      nextParams.delete("search");
    }

    setSearchParams(nextParams, { replace: true });
  }

  function clearFilters() {
    setSearchParams({}, { replace: true });
    setRiskFilter("All");
    setActionFilter("All");
  }

  const filteredTransactions = useMemo(() => {
    const query = search.trim().toLowerCase();

    return transactions.filter((transaction) => {
      const searchableText = [
        transaction.id,
        transaction.location,
        transaction.amount,
        transaction.probability,
        transaction.risk,
        transaction.action,
        transaction.type,
        transaction.time,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      const matchesSearch = !query || searchableText.includes(query);
      const matchesRisk =
        riskFilter === "All" || transaction.risk === riskFilter;
      const matchesAction =
        actionFilter === "All" || transaction.action === actionFilter;

      return matchesSearch && matchesRisk && matchesAction;
    });
  }, [search, riskFilter, actionFilter]);

  const hasActiveFilters =
    search.trim() !== "" ||
    riskFilter !== "All" ||
    actionFilter !== "All";

  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / TRANSACTIONS"
        title="Transaction monitoring"
        description="Search and inspect transactions evaluated by the risk engine."
      />

      <section className="metrics-grid transaction-summary">
        <div className="panel metric-card">
          <span className="summary-label">Displayed transactions</span>
          <strong className="metric-value">
            {filteredTransactions.length}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">High risk</span>
          <strong className="metric-value">
            {filteredTransactions.filter((t) => t.risk === "High").length}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">Review required</span>
          <strong className="metric-value">
            {filteredTransactions.filter((t) => t.action === "Review").length}
          </strong>
        </div>

        <div className="panel metric-card">
          <span className="summary-label">Blocked</span>
          <strong className="metric-value">
            {filteredTransactions.filter((t) => t.action === "Block").length}
          </strong>
        </div>
      </section>

      <section className="panel transactions-panel">
        <div className="panel-heading">
          <div>
            <h3>All transactions</h3>
            <p>
              Showing {filteredTransactions.length} of {transactions.length}{" "}
              available demo records
            </p>
          </div>

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

        <div className="filter-toolbar">
          <div className="search-field">
            <Search size={17} />

            <input
              type="text"
              value={search}
              onChange={(event) => updateSearch(event.target.value)}
              placeholder="Search ID, amount, location, risk or action..."
              aria-label="Search transactions"
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
            value={riskFilter}
            onChange={(event) => setRiskFilter(event.target.value)}
            aria-label="Filter by risk level"
          >
            <option value="All">All risk levels</option>
            <option value="Low">Low risk</option>
            <option value="Medium">Medium risk</option>
            <option value="High">High risk</option>
          </select>

          <select
            value={actionFilter}
            onChange={(event) => setActionFilter(event.target.value)}
            aria-label="Filter by recommended action"
          >
            <option value="All">All actions</option>
            <option value="Approve">Approve</option>
            <option value="Review">Review</option>
            <option value="Block">Block</option>
          </select>
        </div>

        <div className="table-wrapper">
          <table>
            <thead>
              <tr>
                <th>TRANSACTION ID</th>
                <th>TYPE</th>
                <th>LOCATION</th>
                <th>AMOUNT</th>
                <th>FRAUD PROBABILITY</th>
                <th>RISK</th>
                <th>ACTION</th>
                <th>TIME</th>
              </tr>
            </thead>

            <tbody>
              {filteredTransactions.map((transaction) => (
                <tr key={transaction.id}>
                  <td>
                    <button
                      type="button"
                      className="transaction-detail-trigger"
                      onClick={() => setSelectedTransaction(transaction)}
                    >
                      {transaction.id}
                      <ExternalLink size={12} />
                    </button>
                  </td>

                  <td>{transaction.type}</td>
                  <td>{transaction.location}</td>
                  <td>{transaction.amount}</td>
                  <td>{transaction.probability}</td>
                  <td>
                    <RiskBadge risk={transaction.risk} />
                  </td>
                  <td>
                    <span
                      className={`action-label ${transaction.action.toLowerCase()}`}
                    >
                      {transaction.action}
                    </span>
                  </td>
                  <td className="time-cell">{transaction.time}</td>
                </tr>
              ))}

              {filteredTransactions.length === 0 && (
                <tr>
                  <td colSpan="8" className="empty-state">
                    <div className="empty-transactions">
                      <Search size={22} />
                      <strong>No transactions found</strong>
                      <span>
                        Try changing your search or adjusting the filters.
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
      </section>

      {selectedTransaction && (
        <section className="panel transaction-details-panel">
          <div className="panel-heading">
            <div>
              <span className="details-eyebrow">TRANSACTION DETAILS</span>
              <h3>{selectedTransaction.id}</h3>
              <p>Detailed view of the selected demo transaction.</p>
            </div>

            <button
              type="button"
              className="icon-button"
              onClick={() => setSelectedTransaction(null)}
              aria-label="Close transaction details"
            >
              <X size={18} />
            </button>
          </div>

          <div className="transaction-details-grid">
            <div className="detail-item">
              <span>Transaction amount</span>
              <strong>{selectedTransaction.amount}</strong>
            </div>

            <div className="detail-item">
              <span>Transaction type</span>
              <strong>{selectedTransaction.type || "Not available"}</strong>
            </div>

            <div className="detail-item">
              <span>Location</span>
              <strong>{selectedTransaction.location || "Not available"}</strong>
            </div>

            <div className="detail-item">
              <span>Transaction time</span>
              <strong>{selectedTransaction.time}</strong>
            </div>

            <div className="detail-item">
              <span>Fraud probability</span>
              <strong>{selectedTransaction.probability}</strong>
            </div>

            <div className="detail-item">
              <span>Risk classification</span>
              <RiskBadge risk={selectedTransaction.risk} />
            </div>
          </div>

          <div className="decision-summary">
            <div>
              {selectedTransaction.action === "Approve" ? (
                <ShieldCheck size={21} />
              ) : (
                <ShieldAlert size={21} />
              )}
              <div>
                <span>Recommended action</span>
                <strong className={`action-label ${selectedTransaction.action.toLowerCase()}`}>
                  {selectedTransaction.action}
                </strong>
              </div>
            </div>
          </div>

          <p className="demo-disclaimer">
            Demo environment: these transaction records are mock data.
            Feature-level explanations are not available in this view.
          </p>
        </section>
      )}
    </div>
  );
}

export default Transactions;
