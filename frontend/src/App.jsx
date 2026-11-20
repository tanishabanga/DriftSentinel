
import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import DashboardLayout from "./layouts/DashboardLayout";
import "./App.css";

const Overview = lazy(() => import("./pages/Overview"));
const Transactions = lazy(() => import("./pages/Transactions"));
const RiskAnalysis = lazy(() => import("./pages/RiskAnalysis"));
const ModelAnalytics = lazy(() => import("./pages/ModelAnalytics"));
const Prediction = lazy(() => import("./pages/Prediction"));
const MonitoringHistory = lazy(() => import("./pages/MonitoringHistory"));
const Documentation = lazy(() => import("./pages/Documentation"));

function PageLoader() {
  return (
    <div className="dashboard page-loader">
      <div className="panel">
        <p>Loading DriftSentinel...</p>
      </div>
    </div>
  );
}

function PlaceholderPage({ title }) {
  return (
    <div className="dashboard">
      <div className="panel placeholder-panel">
        <h2>{title}</h2>
        <p>This section is planned for a future development phase.</p>
      </div>
    </div>
  );
}

function App() {
  return (
    <Suspense fallback={<PageLoader />}>
      <Routes>
        <Route element={<DashboardLayout />}>
          <Route path="/" element={<Overview />} />
          <Route path="/transactions" element={<Transactions />} />
          <Route path="/risk-analysis" element={<RiskAnalysis />} />
          <Route path="/model-analytics" element={<ModelAnalytics />} />
          <Route path="/prediction" element={<Prediction />} />
          <Route path="/monitoring-history" element={<MonitoringHistory />} />

          <Route
            path="/settings"
            element={<PlaceholderPage title="Settings" />}
          />

          <Route path="/documentation" element={<Documentation />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </Suspense>
  );
}

export default App;
