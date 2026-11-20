import {
  Activity,
  AlertTriangle,
  BookOpen,
  Boxes,
  Code2,
  Database,
  GitBranch,
  Gauge,
  ShieldCheck,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";

const endpoints = [
  {
    method: "GET",
    path: "/health",
    description: "Checks API health and service availability.",
  },
  {
    method: "POST",
    path: "/predict",
    description:
      "Runs batch inference and persists prediction results in the prediction ledger.",
  },
  {
    method: "GET",
    path: "/predictions?limit=100",
    description:
      "Returns recent persisted prediction records. The limit is capped at 100.",
  },
  {
    method: "GET",
    path: "/predictions/batches/{batch_id}",
    description:
      "Retrieves the prediction records associated with a batch.",
  },
  {
    method: "GET",
    path: "/monitoring/snapshots?limit=100",
    description:
      "Lists persisted aggregate monitoring snapshots.",
  },
  {
    method: "GET",
    path: "/monitoring/snapshots/{snapshot_id}",
    description:
      "Retrieves an individual monitoring snapshot.",
  },
  {
    method: "POST",
    path: "/monitoring/snapshots",
    description:
      "Creates and persists a monitoring snapshot from a submitted monitoring payload.",
  },
];

const architecture = [
  {
    icon: Activity,
    title: "React frontend",
    description:
      "Provides prediction workflows, transaction history, risk analysis, model analytics, and monitoring dashboards.",
  },
  {
    icon: Code2,
    title: "FastAPI backend",
    description:
      "Exposes inference, prediction history, health, and monitoring endpoints.",
  },
  {
    icon: ShieldCheck,
    title: "Inference pipeline",
    description:
      "Validates the feature schema and applies XGBoost inference, probability calibration, and expected-cost decision logic.",
  },
  {
    icon: Database,
    title: "SQLite persistence",
    description:
      "Stores prediction ledger records and aggregate monitoring snapshots across API restarts.",
  },
];

function Section({ icon: Icon, title, description, children }) {
  return (
    <section className="panel docs-section">
      <div className="panel-heading">
        <div>
          <h3><Icon size={17} /> {title}</h3>
          {description && <p>{description}</p>}
        </div>
      </div>
      <div className="docs-section-content">{children}</div>
    </section>
  );
}

function Documentation() {
  return (
    <div className="dashboard">
      <PageHeader
        eyebrow="SYSTEM / DOCUMENTATION"
        title={<>DriftSentinel <span>Documentation</span></>}
        description="Understand the architecture, prediction pipeline, monitoring workflow, and API surface."
      />

      <section className="panel docs-intro">
        <div className="docs-intro-icon">
          <BookOpen size={23} />
        </div>
        <div>
          <span className="docs-kicker">PROJECT GUIDE</span>
          <h2>Fraud intelligence with explainable operational workflows.</h2>
          <p>
            DriftSentinel is a temporal fraud intelligence platform built
            around batch inference, cost-aware decisions, persisted prediction
            records, and reference-relative feature monitoring.
          </p>
        </div>
      </section>

      <div className="docs-section-grid">
        <Section
          icon={Boxes}
          title="System architecture"
          description="The main components and how they work together."
        >
          <div className="docs-architecture">
            {architecture.map((item) => {
              const Icon = item.icon;
              return (
                <article className="docs-architecture-item" key={item.title}>
                  <div className="docs-icon">
                    <Icon size={18} />
                  </div>
                  <div>
                    <h4>{item.title}</h4>
                    <p>{item.description}</p>
                  </div>
                </article>
              );
            })}
          </div>
          <div className="docs-flow">
            <span>CSV input</span>
            <b>→</b>
            <span>Frontend validation</span>
            <b>→</b>
            <span>FastAPI inference</span>
            <b>→</b>
            <span>Prediction ledger</span>
            <b>→</b>
            <span>Dashboard</span>
          </div>
        </Section>

        <Section
          icon={GitBranch}
          title="Prediction workflow"
          description="How a transaction batch becomes a set of decisions."
        >
          <ol className="docs-steps">
            <li>
              <strong>Input validation</strong>
              <p>
                The prediction interface accepts CSV input and validates it
                against the published feature schema before submission.
              </p>
            </li>
            <li>
              <strong>Feature compatibility</strong>
              <p>
                The inference pipeline expects 431 model features: 392
                numerical and 39 categorical. TransactionID and isFraud
                are not model input features.
              </p>
            </li>
            <li>
              <strong>Model inference and calibration</strong>
              <p>
                XGBoost generates fraud scores. The configured calibration
                stage produces calibrated probabilities for downstream
                decision-making.
              </p>
            </li>
            <li>
              <strong>Expected-cost decisions</strong>
              <p>
                The decision policy evaluates the costs associated with
                approving, reviewing, or blocking a transaction.
              </p>
            </li>
            <li>
              <strong>Persistence and analysis</strong>
              <p>
                Prediction records are stored with their batch identifiers,
                probabilities, actions, expected costs, and timestamps.
                Dashboard pages read this persisted history.
              </p>
            </li>
          </ol>
        </Section>

        <Section
          icon={Gauge}
          title="Decision policy"
          description="The configured cost matrix used by the inference decision logic."
        >
          <div className="docs-table-wrap">
            <table className="docs-table">
              <thead>
                <tr>
                  <th>Decision</th>
                  <th>Legitimate transaction</th>
                  <th>Fraudulent transaction</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Approve</td>
                  <td>0</td>
                  <td>100</td>
                </tr>
                <tr>
                  <td>Review</td>
                  <td>2</td>
                  <td>5</td>
                </tr>
                <tr>
                  <td>Block</td>
                  <td>10</td>
                  <td>0</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="docs-note">
            These are illustrative cost units defined by the project policy,
            not currency values. The policy compares expected costs rather
            than relying only on a fixed probability threshold.
          </p>
        </Section>

        <Section
          icon={Activity}
          title="Monitoring and drift"
          description="What monitoring snapshots measure and how to interpret them."
        >
          <ul className="docs-list">
            <li>
              <strong>Feature PSI:</strong> compares current feature
              distributions against a reference profile.
            </li>
            <li>
              <strong>Prediction summaries:</strong> record aggregate
              calibrated probability, expected cost, and action counts.
            </li>
            <li>
              <strong>Snapshot history:</strong> persists batch-level
              summaries for historical comparison.
            </li>
            <li>
              <strong>Reference profile:</strong> built from 413,379 rows
              of the processed training split across 431 features.
            </li>
          </ul>
          <div className="docs-warning">
            <AlertTriangle size={18} />
            <p>
              PSI is a distribution-shift screening measure. Elevated PSI
              does not independently prove concept drift or performance
              degradation. Confirming those requires reliable ground-truth
              labels and performance evaluation. Predicted actions are not
              confirmed fraud outcomes.
            </p>
          </div>
        </Section>

        <Section
          icon={Database}
          title="Persistence and data handling"
          description="What is stored by the current backend."
        >
          <ul className="docs-list">
            <li>
              Prediction records contain prediction IDs, batch IDs,
              timestamps, transaction identifiers when provided, raw and
              calibrated probabilities, actions, and expected costs.
            </li>
            <li>
              Monitoring records contain aggregate batch summaries,
              drift statistics, action distributions, and metadata.
            </li>
            <li>
              The prediction ledger does not store the original 431 raw
              feature values for each transaction.
            </li>
            <li>
              SQLite persistence is shared by prediction and monitoring
              storage, using separate tables.
            </li>
          </ul>
        </Section>

        <Section
          icon={Code2}
          title="API reference"
          description="Implemented backend endpoints exposed to the frontend."
        >
          <div className="docs-endpoints">
            {endpoints.map((endpoint) => (
              <article className="docs-endpoint" key={endpoint.path}>
                <div className={`docs-method docs-method-${endpoint.method.toLowerCase()}`}>
                  {endpoint.method}
                </div>
                <div>
                  <code>{endpoint.path}</code>
                  <p>{endpoint.description}</p>
                </div>
              </article>
            ))}
          </div>
          <p className="docs-note">
            Request and response schemas are defined by the running FastAPI
            application. Open the backend /docs page for the interactive
            OpenAPI reference and exact payload definitions.
          </p>
        </Section>

        <Section
          icon={ShieldCheck}
          title="Model analytics and evaluation"
          description="Understanding the saved experiment results."
        >
          <ul className="docs-list">
            <li>
              Model Analytics displays saved XGBoost validation metrics,
              threshold analysis, and calibration results.
            </li>
            <li>
              The main experiment used 413,379 training rows and 88,581
              validation rows, with 431 model features.
            </li>
            <li>
              Calibration analysis used a chronological fit partition of
              44,290 rows and a separate evaluation partition of 44,291 rows.
            </li>
            <li>
              The final test split remains untouched by the reported
              validation experiments.
            </li>
          </ul>
          <p className="docs-note">
            Offline experiment results are not the same as live API prediction
            history. They are presented separately to preserve evaluation
            integrity.
          </p>
        </Section>

        <Section
          icon={AlertTriangle}
          title="Current scope and limitations"
          description="What the current platform does not claim to provide."
        >
          <ul className="docs-list">
            <li>
              No automatic model retraining or autonomous policy adaptation
              is enabled in the current monitoring MVP.
            </li>
            <li>
              Monitoring snapshots summarize submitted batches; they do not
              independently establish real-world fraud outcomes.
            </li>
            <li>
              TransactionDT is a relative time feature, not a UTC timestamp.
            </li>
            <li>
              Concept-drift and performance monitoring require reliable
              delayed ground-truth labels, which are not inferred from
              prediction probabilities.
            </li>
            <li>
              Settings and automated live alerting are not implemented
              in the current frontend.
            </li>
          </ul>
        </Section>
      </div>

      <footer className="dashboard-footer">
        <span>DriftSentinel · Fraud Intelligence Platform</span>
        <span>Documentation · Current implementation</span>
      </footer>
    </div>
  );
}

export default Documentation;