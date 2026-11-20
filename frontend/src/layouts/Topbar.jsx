import { useEffect, useMemo, useRef, useState } from "react";
import { Bell, Search, X } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";

import { getPredictionHistory } from "../lib/api";

const pageNames = {
  "/": "Overview",
  "/transactions": "Transactions",
  "/prediction": "Run Prediction",
  "/monitoring-history": "Monitoring History",
  "/risk-analysis": "Risk Analysis",
  "/model-analytics": "Model Analytics",
  "/settings": "Settings",
  "/documentation": "Documentation",
};

function formatProbability(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function normalizeAction(action) {
  return typeof action === "string" ? action.toLowerCase() : "unknown";
}

function getTransactionLabel(prediction) {
  const transactionId = prediction.TransactionID;

  if (transactionId === null || transactionId === undefined) {
    return prediction.prediction_id || "Prediction";
  }

  return String(transactionId);
}

function Topbar() {
  const location = useLocation();
  const navigate = useNavigate();

  const [searchOpen, setSearchOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const [predictions, setPredictions] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState("");

  const searchRef = useRef(null);
  const notificationRef = useRef(null);
  const searchInputRef = useRef(null);

  const currentPage = pageNames[location.pathname] || "Overview";

  const searchResults = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    if (!query) return [];

    return predictions
      .filter((prediction) => {
        const searchableText = [
          prediction.TransactionID,
          prediction.prediction_id,
          prediction.batch_id,
          prediction.action,
          prediction.raw_probability,
          prediction.calibrated_probability,
          prediction.expected_cost,
          prediction.predicted_at_utc,
        ]
          .filter((value) => value !== null && value !== undefined)
          .join(" ")
          .toLowerCase();

        return searchableText.includes(query);
      })
      .slice(0, 6);
  }, [predictions, searchQuery]);

  useEffect(() => {
    let active = true;

    async function loadPredictions() {
      setSearchLoading(true);
      setSearchError("");

      try {
        const response = await getPredictionHistory(100);

        if (!active) return;

        setPredictions(
          Array.isArray(response.predictions) ? response.predictions : [],
        );
      } catch (error) {
        if (!active) return;

        setSearchError(
          error.message || "Unable to load prediction history.",
        );
        setPredictions([]);
      } finally {
        if (active) setSearchLoading(false);
      }
    }

    loadPredictions();

    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function handleOutsideClick(event) {
      if (
        searchRef.current &&
        !searchRef.current.contains(event.target)
      ) {
        setSearchOpen(false);
      }

      if (
        notificationRef.current &&
        !notificationRef.current.contains(event.target)
      ) {
        setNotificationOpen(false);
      }
    }

    function handleEscape(event) {
      if (event.key === "Escape") {
        setSearchOpen(false);
        setNotificationOpen(false);
        setSearchQuery("");
      }
    }

    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);

    return () => {
      document.removeEventListener("mousedown", handleOutsideClick);
      document.removeEventListener("keydown", handleEscape);
    };
  }, []);

  useEffect(() => {
    if (searchOpen) {
      searchInputRef.current?.focus();
    }
  }, [searchOpen]);

  function toggleSearch() {
    setSearchOpen((open) => !open);
    setNotificationOpen(false);
    setSearchQuery("");
  }

  function toggleNotifications() {
    setNotificationOpen((open) => !open);
    setSearchOpen(false);
    setSearchQuery("");
  }

  function openTransactionSearch(prediction) {
    const transactionId = prediction.TransactionID;

    if (transactionId === null || transactionId === undefined) {
      return;
    }

    navigate(
      `/transactions?search=${encodeURIComponent(String(transactionId))}`,
    );

    setSearchOpen(false);
    setSearchQuery("");
  }

  return (
    <header className="topbar">
      <div className="breadcrumb">
        <span>Workspace</span>
        <span className="breadcrumb-separator">/</span>
        <strong>{currentPage}</strong>
      </div>

      <div className="topbar-actions">
        <div className="environment">
          <span className="status-dot" />
          Development
        </div>

        <div className="topbar-popover-wrapper" ref={searchRef}>
          <button
            className={`icon-button ${searchOpen ? "active" : ""}`}
            aria-label="Search predictions"
            aria-expanded={searchOpen}
            onClick={toggleSearch}
            type="button"
          >
            {searchOpen ? <X size={18} /> : <Search size={18} />}
          </button>

          {searchOpen && (
            <div className="topbar-popover search-popover">
              <div className="popover-heading">
                <div>
                  <strong>Search predictions</strong>
                  <span>Search persisted prediction records</span>
                </div>
              </div>

              <div className="global-search-input">
                <Search size={17} />
                <input
                  ref={searchInputRef}
                  type="text"
                  placeholder="Search transaction ID, batch ID or action..."
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                />

                {searchQuery && (
                  <button
                    type="button"
                    aria-label="Clear search"
                    onClick={() => setSearchQuery("")}
                  >
                    <X size={15} />
                  </button>
                )}
              </div>

              <div className="global-search-results">
                {searchLoading ? (
                  <p className="popover-empty">
                    Loading persisted predictions...
                  </p>
                ) : searchError ? (
                  <p className="popover-empty" role="alert">
                    {searchError}
                  </p>
                ) : !searchQuery.trim() ? (
                  <p className="popover-empty">
                    Try a transaction ID, prediction ID, batch ID or action.
                  </p>
                ) : searchResults.length === 0 ? (
                  <p className="popover-empty">
                    No matching predictions in the latest 100 records.
                  </p>
                ) : (
                  searchResults.map((prediction) => {
                    const action = normalizeAction(prediction.action);

                    return (
                      <button
                        type="button"
                        className="global-search-result"
                        key={prediction.prediction_id}
                        onClick={() => openTransactionSearch(prediction)}
                        disabled={
                          prediction.TransactionID === null ||
                          prediction.TransactionID === undefined
                        }
                      >
                        <div className="search-result-icon">
                          <Search size={15} />
                        </div>

                        <div className="search-result-details">
                          <strong>{getTransactionLabel(prediction)}</strong>
                          <span>
                            {formatProbability(
                              prediction.calibrated_probability,
                            )}{" "}
                            calibrated probability
                          </span>
                        </div>

                        <span className={`action-text ${action}`}>
                          {prediction.action || "Unknown"}
                        </span>
                      </button>
                    );
                  })
                )}
              </div>

              <div className="popover-footer">
                Searching the latest 100 persisted prediction records
              </div>
            </div>
          )}
        </div>

        <div className="topbar-popover-wrapper" ref={notificationRef}>
          <button
            className={`icon-button notification-button ${
              notificationOpen ? "active" : ""
            }`}
            aria-label="Notifications"
            aria-expanded={notificationOpen}
            onClick={toggleNotifications}
            type="button"
          >
            <Bell size={18} />
          </button>

          {notificationOpen && (
            <div className="topbar-popover notification-popover">
              <div className="notification-header">
                <div>
                  <strong>Notifications</strong>
                  <span>Alert integration status</span>
                </div>
              </div>

              <div className="notification-list">
                <div className="notification-item read">
                  <div className="notification-icon system">
                    <Bell size={16} />
                  </div>

                  <div className="notification-content">
                    <strong>Live alerts are not configured</strong>
                    <p>
                      DriftSentinel currently persists predictions and
                      monitoring snapshots. A live notification event system
                      has not been implemented yet.
                    </p>
                  </div>
                </div>
              </div>

              <div className="popover-footer">
                No simulated alerts · No live alert feed
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Topbar;