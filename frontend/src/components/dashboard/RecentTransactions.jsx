
import { ArrowUpRight } from "lucide-react";
import { Link } from "react-router-dom";

import { transactions } from "../../data/mockData";
import RiskBadge from "../common/RiskBadge";

function RecentTransactions() {
  return (
    <section className="panel transactions-panel">
      <div className="panel-heading">
        <div>
          <h3>Recent transactions</h3>
          <p>Latest transactions processed by the risk engine</p>
        </div>

        <Link to="/transactions" className="subtle-button">
          View all <ArrowUpRight size={14} />
        </Link>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>TRANSACTION ID</th>
              <th>AMOUNT</th>
              <th>FRAUD PROBABILITY</th>
              <th>RISK LEVEL</th>
              <th>RECOMMENDED ACTION</th>
              <th>TIME</th>
            </tr>
          </thead>

          <tbody>
            {transactions.slice(0, 5).map((transaction) => (
              <tr key={transaction.id}>
                <td className="transaction-id">{transaction.id}</td>
                <td>{transaction.amount}</td>
                <td>{transaction.probability}</td>
                <td><RiskBadge risk={transaction.risk} /></td>
                <td>
                  <span className={`action-label ${transaction.action.toLowerCase()}`}>
                    {transaction.action}
                  </span>
                </td>
                <td className="time-cell">{transaction.time}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default RecentTransactions;
