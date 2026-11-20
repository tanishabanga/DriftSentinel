const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "/api"
).replace(/\/+$/, "");

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: {
      Accept: "application/json",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...options.headers,
    },
  });

  if (!response.ok) {
    let message = `API request failed (${response.status})`;

    try {
      const errorData = await response.json();
      message =
        typeof errorData.detail === "string"
          ? errorData.detail
          : JSON.stringify(errorData.detail ?? errorData);
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }

    throw new Error(message);
  }

  return response.json();
}

export function getHealth() {
  return apiRequest("/health");
}

export function getMonitoringSnapshots(limit = 20) {
  const safeLimit = Math.min(100, Math.max(1, limit));

  return apiRequest(`/monitoring/snapshots?limit=${safeLimit}`);
}

export function getMonitoringSnapshot(snapshotId) {
  if (!snapshotId) {
    throw new Error("A snapshot ID is required.");
  }

  return apiRequest(
    `/monitoring/snapshots/${encodeURIComponent(snapshotId)}`,
  );
}

export function createMonitoringSnapshot(payload) {
  return apiRequest("/monitoring/snapshots", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
export function predictTransactions(payload) {
  return apiRequest("/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getPredictionHistory(limit = 100) {
  const safeLimit = Math.min(100, Math.max(1, limit));

  return apiRequest(`/predictions?limit=${safeLimit}`);
}

export function getPredictionBatches(limit = 100) {
  const safeLimit = Math.min(100, Math.max(1, limit));

  return apiRequest(`/predictions/batches?limit=${safeLimit}`);
}

export function getPredictionBatchPage(batchId, options = {}) {
  if (!batchId) {
    throw new Error("A prediction batch ID is required.");
  }

  const {
    limit = 25,
    offset = 0,
    search = "",
    action = "",
  } = options;

  const params = new URLSearchParams({
    limit: String(Math.min(100, Math.max(1, limit))),
    offset: String(Math.max(0, offset)),
  });

  if (search.trim()) {
    params.set("search", search.trim());
  }

  if (["approve", "review", "block"].includes(action)) {
    params.set("action", action);
  }

  return apiRequest(
    `/predictions/batches/${encodeURIComponent(batchId)}/page?${params.toString()}`
  );
}

export function getPredictionBatchTopRisks(batchId, options = {}) {
  if (!batchId) {
    throw new Error("A prediction batch ID is required.");
  }

  const { limit = 20, action = "" } = options;
  const params = new URLSearchParams({
    limit: String(Math.min(20, Math.max(1, limit))),
  });

  if (["approve", "review", "block"].includes(action)) {
    params.set("action", action);
  }

  return apiRequest(
    `/predictions/batches/${encodeURIComponent(batchId)}/top-risk?${params.toString()}`
  );
}

export function getPredictionBatch(batchId) {
  if (!batchId) {
    throw new Error("A prediction batch ID is required.");
  }

  return apiRequest(
    `/predictions/batches/${encodeURIComponent(batchId)}`
  );
}