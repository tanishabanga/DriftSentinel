
import {
  AlertTriangle,
  ShieldAlert,
  ShieldCheck,
  Wallet,
} from "lucide-react";

import MetricCard from "../components/dashboard/MetricCard";
import TransactionChart from "../components/common/TransactionChart";
import RiskDistribution from "../components/common/RiskDistribution";
import RecentTransactions from "../components/dashboard/RecentTransactions";
import PageHeader from "../components/common/PageHeader";


function Overview() {
  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / OVERVIEW"
        title={<>Good evening, Admin <span>✦</span></>}
        description="Here's what's happening with your transaction ecosystem."
      >
        <span className="chart-period">Last 7 days</span>
      </PageHeader>

      <section className="metrics-grid">
        <MetricCard
          title="Total transactions"
          value="39,700"
          change="+12.8%"
          positive
          icon={<Wallet size={19} />}
          description="vs. previous 7 days"
        />

        <MetricCard
          title="Fraud detected"
          value="1,850"
          change="+4.2%"
          icon={<ShieldAlert size={19} />}
          description="vs. previous 7 days"
        />

        <MetricCard
          title="High-risk transactions"
          value="284"
          change="-8.4%"
          positive
          icon={<AlertTriangle size={19} />}
          description="vs. previous 7 days"
        />

        <MetricCard
          title="Fraud detection rate"
          value="94.2%"
          change="+2.1%"
          positive
          icon={<ShieldCheck size={19} />}
          description="vs. previous 7 days"
        />
      </section>

      <section className="charts-grid">
        <TransactionChart />
        <RiskDistribution />
      </section>

      <RecentTransactions />

      <footer className="dashboard-footer">
        <span>DriftSentinel · Fraud Intelligence Platform</span>
        <span>Demo environment · Mock data</span>
      </footer>
    </div>
  );
}

export default Overview;
