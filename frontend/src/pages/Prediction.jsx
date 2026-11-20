import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";
import {
  AlertTriangle,
  CheckCircle2,
  FileSpreadsheet,
  LoaderCircle,
  ShieldAlert,
  Upload,
  Download,
  RefreshCw,
} from "lucide-react";

import PageHeader from "../components/common/PageHeader";
import { predictTransactions } from "../lib/api";

const MAX_BATCH_SIZE = 1000;

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatCost(value) {
  return Number(value).toFixed(4);
}

function formatAction(action) {
  return String(action ?? "").toUpperCase();
}

function Prediction() {
  const [schema, setSchema] = useState(null);
  const [schemaError, setSchemaError] = useState("");
  const [file, setFile] = useState(null);
  const [transactions, setTransactions] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [parsing, setParsing] = useState(false);

  useEffect(() => {
    let active = true;

    fetch("/feature_schema.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error("Could not load the model feature schema.");
        }
        return response.json();
      })
      .then((data) => {
        if (active) setSchema(data);
      })
      .catch((err) => {
        if (active) setSchemaError(err.message);
      });

    return () => {
      active = false;
    };
  }, []);

  const actionCounts = useMemo(() => {
    return predictions.reduce(
      (counts, prediction) => {
        const action = String(prediction.action ?? "").toLowerCase();
        if (action in counts) counts[action] += 1;
        return counts;
      },
      { approve: 0, review: 0, block: 0 },
    );
  }, [predictions]);

  const averageProbability = useMemo(() => {
    if (!predictions.length) return 0;
    return (
      predictions.reduce(
        (sum, prediction) =>
          sum + Number(prediction.calibrated_probability),
        0,
      ) / predictions.length
    );
  }, [predictions]);

  function resetResults() {
    setPredictions([]);
    setError("");
  }

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0];
    event.target.value = "";

    setFile(null);
    setTransactions([]);
    resetResults();

    if (!selectedFile) return;

    if (!selectedFile.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a CSV file.");
      return;
    }

    if (!schema) {
      setError("The model feature schema is not ready yet.");
      return;
    }

    setParsing(true);
    setFile(selectedFile);

    Papa.parse(selectedFile, {
      header: true,
      skipEmptyLines: "greedy",
      transformHeader: (header) => header.trim(),
      dynamicTyping: false,
      complete: (result) => {
        try {
          if (result.errors.length > 0) {
            const firstError = result.errors[0];
            throw new Error(
              `CSV parsing failed: ${firstError.message} (row ${firstError.row ?? "unknown"}).`,
            );
          }

          const headers = result.meta.fields ?? [];
          const uniqueHeaders = new Set(headers);

          if (uniqueHeaders.size !== headers.length) {
            throw new Error("The CSV contains duplicate column names.");
          }

          const requiredFeatures = schema.feature_names;
          const missingFeatures = requiredFeatures.filter(
            (feature) => !uniqueHeaders.has(feature),
          );

          if (missingFeatures.length > 0) {
            const preview = missingFeatures.slice(0, 12).join(", ");
            const suffix =
              missingFeatures.length > 12
                ? ` and ${missingFeatures.length - 12} more`
                : "";

            throw new Error(
              `Missing ${missingFeatures.length} required model features: ${preview}${suffix}.`,
            );
          }

          const rows = result.data.filter((row) =>
            Object.values(row).some(
              (value) => value !== null && String(value).trim() !== "",
            ),
          );

          if (rows.length === 0) {
            throw new Error("The CSV contains no transaction rows.");
          }

          if (rows.length > MAX_BATCH_SIZE) {
            throw new Error(
              `This file contains ${rows.length} transactions. The maximum batch size is ${MAX_BATCH_SIZE}.`,
            );
          }

          const featureSet = new Set(requiredFeatures);

          const cleanRows = rows.map((row) => {
            const transaction = {};

            for (const [key, value] of Object.entries(row)) {
              if (featureSet.has(key) || key === "TransactionID") {
                transaction[key] =
                  value === null || String(value).trim() === ""
                    ? null
                    : value;
              }
            }

            return transaction;
          });

          setTransactions(cleanRows);
          setError("");
        } catch (err) {
          setFile(null);
          setTransactions([]);
          setError(err.message);
        } finally {
          setParsing(false);
        }
      },
      error: (err) => {
        setFile(null);
        setTransactions([]);
        setError(`Could not read CSV: ${err.message}`);
        setParsing(false);
      },
    });
  }

  async function handlePrediction() {
    if (!transactions.length || loading) return;

    setLoading(true);
    setError("");
    setPredictions([]);

    try {
      const response = await predictTransactions({
        transactions,
      });

      if (!Array.isArray(response.predictions)) {
        throw new Error("The prediction API returned an invalid response.");
      }

      if (response.predictions.length !== transactions.length) {
        throw new Error(
          "The API returned a different number of predictions than submitted transactions.",
        );
      }

      setPredictions(response.predictions);
    } catch (err) {
      setError(err.message || "Prediction request failed.");
    } finally {
      setLoading(false);
    }
  }

  function downloadResults() {
    if (!predictions.length) return;

    const csv = Papa.unparse(
      predictions.map((prediction, index) => ({
        TransactionID: prediction.TransactionID ?? index + 1,
        raw_probability: prediction.raw_probability,
        calibrated_probability: prediction.calibrated_probability,
        action: prediction.action,
        expected_cost: prediction.expected_cost,
      })),
    );

    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "driftsentinel_predictions.csv";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function resetAll() {
    setFile(null);
    setTransactions([]);
    setPredictions([]);
    setError("");
    setParsing(false);
  }

  return (
    <div className="dashboard prediction-page">
      <PageHeader
        eyebrow="FRAUD INTELLIGENCE / PREDICTION"
        title="Transaction risk assessment"
        description="Upload transaction data and run the trained, calibrated XGBoost inference pipeline."
      />

      <div className="prediction-layout">
        <section className="panel prediction-form-panel">
          <div className="panel-heading">
            <div>
              <h3>Batch input</h3>
              <p>Upload a CSV containing the model's required transaction features.</p>
            </div>
            <FileSpreadsheet size={21} className="prediction-heading-icon" />
          </div>

          <div className="prediction-upload-area">
            <div className="prediction-upload-icon">
              <Upload size={23} />
            </div>

            <strong>Upload transaction CSV</strong>
            <span>
              Required: all {schema?.feature_count ?? 431} model features
            </span>

            <label className="primary-button prediction-upload-button">
              <Upload size={15} />
              Choose CSV file
              <input
                type="file"
                accept=".csv,text/csv"
                onChange={handleFileChange}
                disabled={!schema || parsing || loading}
              />
            </label>

            <span className="prediction-upload-hint">
              Maximum {MAX_BATCH_SIZE.toLocaleString("en-IN")} transactions per batch
            </span>
          </div>

          {schemaError && (
            <p className="form-error" role="alert">{schemaError}</p>
          )}

          {parsing && (
            <div className="prediction-status">
              <LoaderCircle size={16} className="prediction-spinner" />
              Validating CSV headers and transaction rows...
            </div>
          )}

          {file && transactions.length > 0 && (
            <div className="prediction-file-info">
              <div>
                <FileSpreadsheet size={18} />
                <div>
                  <strong>{file.name}</strong>
                  <span>
                    {transactions.length.toLocaleString("en-IN")} transactions ready
                  </span>
                </div>
              </div>
              <CheckCircle2 size={19} className="prediction-success-icon" />
            </div>
          )}

          <div className="prediction-schema-info">
            <strong>Model input contract</strong>
            <div className="prediction-schema-grid">
              <div>
                <span>Required features</span>
                <strong>{schema?.feature_count ?? 431}</strong>
              </div>
              <div>
                <span>Numerical</span>
                <strong>{schema?.numerical_columns?.length ?? 392}</strong>
              </div>
              <div>
                <span>Categorical</span>
                <strong>{schema?.categorical_columns?.length ?? 39}</strong>
              </div>
            </div>
            <p>
              Extra columns are ignored. Blank values are preserved as missing.
              The backend handles feature conversion and unseen categories.
              The target column <code>isFraud</code> is excluded.
            </p>
          </div>

          {error && (
            <div className="prediction-error" role="alert">
              <AlertTriangle size={16} />
              <span>{error}</span>
            </div>
          )}

          <div className="prediction-form-actions">
            <button
              type="button"
              className="secondary-button"
              onClick={resetAll}
              disabled={loading || parsing}
            >
              <RefreshCw size={15} />
              Reset
            </button>

            <button
              type="button"
              className="primary-button"
              onClick={handlePrediction}
              disabled={!transactions.length || loading || parsing || !schema}
            >
              {loading ? (
                <LoaderCircle size={16} className="prediction-spinner" />
              ) : (
                <ShieldAlert size={16} />
              )}
              {loading ? "Running inference..." : "Run model inference"}
            </button>
          </div>
        </section>

        <section className="panel prediction-result-panel">
          <div className="panel-heading">
            <div>
              <h3>Inference summary</h3>
              <p>Results returned by the trained model and decision policy.</p>
            </div>
            <ShieldAlert size={21} className="result-shield-icon" />
          </div>

          {!predictions.length ? (
            <div className="prediction-empty">
              <div className="prediction-empty-icon">
                <ShieldAlert size={27} />
              </div>
              <strong>Waiting for inference</strong>
              <span>
                Upload a valid transaction CSV and run model inference to see
                calibrated predictions and decisions.
              </span>
            </div>
          ) : (
            <div className="prediction-summary">
              <span className="result-label">Transactions evaluated</span>
              <strong className="prediction-probability">
                {predictions.length.toLocaleString("en-IN")}
              </strong>

              <div className="result-detail-row">
                <span>Mean calibrated fraud probability</span>
                <strong>{formatPercent(averageProbability)}</strong>
              </div>

              <div className="prediction-action-counts">
                <div className="approve">
                  <span>Approve</span>
                  <strong>{actionCounts.approve}</strong>
                </div>
                <div className="review">
                  <span>Review</span>
                  <strong>{actionCounts.review}</strong>
                </div>
                <div className="block">
                  <span>Block</span>
                  <strong>{actionCounts.block}</strong>
                </div>
              </div>

              <button
                type="button"
                className="secondary-button prediction-download-button"
                onClick={downloadResults}
              >
                <Download size={15} />
                Download predictions CSV
              </button>
            </div>
          )}
        </section>
      </div>

      {predictions.length > 0 && (
        <section className="panel prediction-table-panel">
          <div className="panel-heading">
            <div>
              <h3>Transaction predictions</h3>
              <p>Individual outputs from the inference API.</p>
            </div>
            <span className="prediction-result-count">
              {predictions.length} results
            </span>
          </div>

          <div className="prediction-table-wrap">
            <table className="prediction-table">
              <thead>
                <tr>
                  <th>Transaction ID</th>
                  <th>Raw probability</th>
                  <th>Calibrated probability</th>
                  <th>Decision</th>
                  <th>Expected cost</th>
                </tr>
              </thead>
              <tbody>
                {predictions.map((prediction, index) => {
                  const action = String(prediction.action ?? "").toLowerCase();

                  return (
                    <tr key={`${prediction.TransactionID ?? "row"}-${index}`}>
                      <td>{prediction.TransactionID ?? index + 1}</td>
                      <td>{formatPercent(prediction.raw_probability)}</td>
                      <td>{formatPercent(prediction.calibrated_probability)}</td>
                      <td>
                        <span className={`prediction-action-badge ${action}`}>
                          {formatAction(action)}
                        </span>
                      </td>
                      <td>{formatCost(prediction.expected_cost)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="prediction-model-note">
        <AlertTriangle size={14} />
        <span>
          Decisions are generated by the loaded XGBoost model, probability
          calibrator, and cost-sensitive decision policy. An action is a model
          decision, not confirmed fraud ground truth.
        </span>
      </div>
    </div>
  );
}

export default Prediction;