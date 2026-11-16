
import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Zap,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";

const initialForm = {
  amount: "",
  paymentType: "Card payment",
  device: "Known device",
  international: "No",
  recentAttempts: "0",
};

function Prediction() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function handleChange(event) {
    const { name, value } = event.target;

    setForm((current) => ({
      ...current,
      [name]: value,
    }));

    // A previous result must not remain visible after editing inputs.
    setResult(null);
    setError("");
  }

  function runDemoAssessment(event) {
    event.preventDefault();

    const amount = Number(form.amount);

    if (
      form.amount.trim() === "" ||
      !Number.isFinite(amount) ||
      amount <= 0
    ) {
      setError("Please enter a valid transaction amount greater than zero.");
      setResult(null);
      return;
    }

    let score = 8;

    if (amount >= 50000) score += 25;
    else if (amount >= 20000) score += 15;
    else if (amount >= 10000) score += 8;

    if (form.device === "New device") score += 20;
    if (form.international === "Yes") score += 18;
    if (form.paymentType === "Online purchase") score += 8;

    score += Number(form.recentAttempts) * 10;

    score = Math.min(score, 97);

    const risk =
      score >= 70 ? "High" : score >= 35 ? "Medium" : "Low";

    const action =
      risk === "High"
        ? "Block"
        : risk === "Medium"
          ? "Review"
          : "Approve";

    setResult({
      probability: score,
      risk,
      action,
    });

    setError("");
  }

  function resetForm() {
  setForm({
    amount: "",
    paymentType: "Card payment",
    device: "Known device",
    international: "No",
    recentAttempts: "0",
  });

  setResult(null);
}

  return (
    <div className="dashboard prediction-page">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / PREDICTION"
        title="Transaction risk assessment"
        description="Submit transaction details to see a demo fraud-risk assessment."
      />

      <div className="prediction-layout">
        <section className="panel prediction-form-panel">
          <div className="panel-heading">
            <div>
              <h3>Transaction information</h3>
              <p>Enter the details for the transaction you want to assess.</p>
            </div>

            <div className="prediction-heading-icon">
              <Activity size={19} />
            </div>
          </div>

          <form onSubmit={runDemoAssessment}>
            <div className="prediction-form-grid">
              <label className="form-field">
                <span>Transaction amount (₹)</span>
                <input
                  type="number"
                  name="amount"
                  min="0.01"
                  step="0.01"
                  value={form.amount}
                  onChange={handleChange}
                  placeholder="e.g. 12500"
                  required
                />
              </label>

              <label className="form-field">
                <span>Transaction type</span>
                <select
                  name="paymentType"
                  value={form.paymentType}
                  onChange={handleChange}
                >
                  <option>Card payment</option>
                  <option>Online purchase</option>
                  <option>UPI transfer</option>
                </select>
              </label>

              <label className="form-field">
                <span>Device familiarity</span>
                <select
                  name="device"
                  value={form.device}
                  onChange={handleChange}
                >
                  <option>Known device</option>
                  <option>New device</option>
                </select>
              </label>

              <label className="form-field">
                <span>International transaction</span>
                <select
                  name="international"
                  value={form.international}
                  onChange={handleChange}
                >
                  <option>No</option>
                  <option>Yes</option>
                </select>
              </label>

              <label className="form-field">
                <span>Recent transaction attempts</span>
                <select
                  name="recentAttempts"
                  value={form.recentAttempts}
                  onChange={handleChange}
                >
                  <option value="0">0 attempts</option>
                  <option value="1">1 attempt</option>
                  <option value="2">2 attempts</option>
                  <option value="3">3 attempts</option>
                  <option value="4">4 or more attempts</option>
                </select>
              </label>
            </div>

            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}

            <div className="prediction-form-actions">
              <button
                type="button"
                className="secondary-button"
                onClick={resetForm}
              >
                <RotateCcw size={15} />
                Reset
              </button>

              <button type="submit" className="primary-button">
                <Zap size={16} />
                Assess transaction
              </button>
            </div>
          </form>
        </section>

        <section className="panel prediction-result-panel">
          <div className="panel-heading">
            <div>
              <h3>Assessment result</h3>
              <p>Risk assessment output</p>
            </div>

            <ShieldCheck size={21} className="result-shield-icon" />
          </div>

          {!result ? (
            <div className="prediction-empty">
              <div className="prediction-empty-icon">
                <ShieldAlert size={27} />
              </div>

              <strong>Waiting for transaction</strong>

              <span>
                Complete the form and run an assessment to see the result here.
              </span>
            </div>
          ) : (
            <div className="prediction-result">
              <div
                className={`prediction-risk-icon ${result.risk.toLowerCase()}`}
              >
                {result.risk === "High" ? (
                  <AlertTriangle size={25} />
                ) : result.risk === "Medium" ? (
                  <ShieldAlert size={25} />
                ) : (
                  <CheckCircle2 size={25} />
                )}
              </div>

              <span className="result-label">Demo fraud-risk score</span>

              <strong className="prediction-probability">
                {result.probability}%
              </strong>

              <div
                className="prediction-progress"
                role="progressbar"
                aria-label="Demo fraud-risk score"
                aria-valuenow={result.probability}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className={`prediction-progress-fill ${result.risk.toLowerCase()}`}
                  style={{ width: `${result.probability}%` }}
                />
              </div>

              <div className="result-detail-row">
                <span>Risk classification</span>
                <strong className={`risk-text ${result.risk.toLowerCase()}`}>
                  {result.risk} risk
                </strong>
              </div>

              <div className="result-detail-row">
                <span>Recommended action</span>
                <strong className={`action-label ${result.action.toLowerCase()}`}>
                  {result.action}
                </strong>
              </div>
            </div>
          )}

          <div className="demo-disclaimer">
            <AlertTriangle size={14} />
            <span>
              Demo only. This score is generated by illustrative frontend
              rules, not by the trained XGBoost model.
            </span>
          </div>
        </section>
      </div>
    </div>
  );
}

export default Prediction;
