
import { useEffect, useMemo, useRef, useState } from "react";
import { Bell, Search, X, CheckCheck, ShieldAlert } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";

import { transactions } from "../data/mockData";

const pageNames = {
  "/": "Overview",
  "/transactions": "Transactions",
  "/prediction": "Run Prediction",
  "/risk-analysis": "Risk Analysis",
  "/model-analytics": "Model Analytics",
  "/settings": "Settings",
  "/documentation": "Documentation",
};


const initialNotifications = [
  {
    id: 1,
    title: "High-risk transaction detected",
    message: "A transaction has a high predicted fraud probability.",
    time: "2 min ago",
    read: false,
    type: "fraud",
  },
  {
    id: 2,
    title: "Manual review required",
    message: "A transaction is awaiting review.",
    time: "8 min ago",
    read: false,
    type: "review",
  },
  {
    id: 3,
    title: "Model status",
    message: "XGBoost validation model is available.",
    time: "Today",
    read: true,
    type: "system",
  },
];



function Topbar() {
  const location = useLocation();
  const navigate = useNavigate();

  const [searchOpen, setSearchOpen] = useState(false);
  const [notificationOpen, setNotificationOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [notifications, setNotifications] = useState(initialNotifications);

  const searchRef = useRef(null);
  const notificationRef = useRef(null);
  const searchInputRef = useRef(null);

  const currentPage = pageNames[location.pathname] || "Overview";

  const unreadCount = notifications.filter(
    (notification) => !notification.read
  ).length;

  const searchResults = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    if (!query) return [];

    return transactions
      .filter((transaction) => {
        const searchableText = [
          transaction.id,
          transaction.amount,
          transaction.probability,
          transaction.risk,
          transaction.action,
          transaction.time,
          transaction.type,
          transaction.location,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase();

        return searchableText.includes(query);
      })
      .slice(0, 6);
  }, [searchQuery]);

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

  function openTransactionSearch(transaction) {
    navigate(
      `/transactions?search=${encodeURIComponent(transaction.id)}`
    );

    setSearchOpen(false);
    setSearchQuery("");
  }

  function markAllAsRead() {
    setNotifications((current) =>
      current.map((notification) => ({
        ...notification,
        read: true,
      }))
    );
  }

  function markAsRead(id) {
    setNotifications((current) =>
      current.map((notification) =>
        notification.id === id
          ? { ...notification, read: true }
          : notification
      )
    );
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

        {/* Global search */}
        <div className="topbar-popover-wrapper" ref={searchRef}>
          <button
            className={`icon-button ${searchOpen ? "active" : ""}`}
            aria-label="Search transactions"
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
                  <strong>Search transactions</strong>
                  <span>Find records in the demo dataset</span>
                </div>
              </div>

              <div className="global-search-input">
                <Search size={17} />
                <input
                  ref={searchInputRef}
                  type="text"
                  placeholder="Search ID, amount, risk or action..."
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
                {!searchQuery.trim() ? (
                  <p className="popover-empty">
                    Try a transaction ID, amount, or risk level.
                  </p>
                ) : searchResults.length === 0 ? (
                  <p className="popover-empty">
                    No matching transactions found.
                  </p>
                ) : (
                  searchResults.map((transaction) => (
                    <button
                      type="button"
                      className="global-search-result"
                      key={transaction.id}
                      onClick={() => openTransactionSearch(transaction)}
                    >
                      <div className="search-result-icon">
                        <Search size={15} />
                      </div>

                      <div className="search-result-details">
                        <strong>{transaction.id}</strong>
                        <span>
                          {transaction.amount} · {transaction.risk} risk
                        </span>
                      </div>

                      <span className={`action-text ${transaction.action?.toLowerCase()}`}>
                        {transaction.action}
                      </span>
                    </button>
                  ))
                )}
              </div>

              <div className="popover-footer">
                Searching the currently loaded demo transactions
              </div>
            </div>
          )}
        </div>

        {/* Notifications */}
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
            {unreadCount > 0 && (
              <span className="notification-indicator" />
            )}
          </button>

          {notificationOpen && (
            <div className="topbar-popover notification-popover">
              <div className="notification-header">
                <div>
                  <strong>Notifications</strong>
                  <span>
                    {unreadCount} unread notification
                    {unreadCount === 1 ? "" : "s"}
                  </span>
                </div>

                <button
                  type="button"
                  className="mark-read-button"
                  onClick={markAllAsRead}
                  disabled={unreadCount === 0}
                >
                  <CheckCheck size={14} />
                  Mark all read
                </button>
              </div>

              <div className="notification-list">
                {notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`notification-item ${
                      notification.read ? "read" : "unread"
                    }`}
                  >
                    <div className={`notification-icon ${notification.type}`}>
                      <ShieldAlert size={16} />
                    </div>

                    <div className="notification-content">
                      <strong>{notification.title}</strong>
                      <p>{notification.message}</p>
                      <span>{notification.time}</span>
                    </div>

                    {!notification.read && (
                      <button
                        type="button"
                        className="mark-one-read"
                        aria-label={`Mark ${notification.title} as read`}
                        title="Mark as read"
                        onClick={() => markAsRead(notification.id)}
                      />
                    )}
                  </div>
                ))}
              </div>

              <div className="popover-footer">
                Demo notifications · Not live alerts
              </div>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}

export default Topbar;
