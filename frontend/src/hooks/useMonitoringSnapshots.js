import { useCallback, useEffect, useRef, useState } from "react";
import { getMonitoringSnapshots } from "../lib/api";

async function fetchSnapshots(limit) {
  const response = await getMonitoringSnapshots(limit);

  if (!Array.isArray(response?.snapshots)) {
    throw new Error(
      "Monitoring API returned an invalid snapshots response.",
    );
  }

  return response.snapshots;
}

export function useMonitoringSnapshots(limit = 20) {
  const [snapshots, setSnapshots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const requestSequence = useRef(0);

  const refresh = useCallback(async () => {
    const requestId = ++requestSequence.current;

    setLoading(true);
    setError("");

    try {
      const result = await fetchSnapshots(limit);

      if (requestId === requestSequence.current) {
        setSnapshots(result);
      }
    } catch (err) {
      if (requestId === requestSequence.current) {
        setError(err.message || "Unable to load monitoring snapshots.");
      }
    } finally {
      if (requestId === requestSequence.current) {
        setLoading(false);
      }
    }
  }, [limit]);

  useEffect(() => {
    const requestId = ++requestSequence.current;

    fetchSnapshots(limit)
      .then((result) => {
        if (requestId === requestSequence.current) {
          setSnapshots(result);
          setError("");
        }
      })
      .catch((err) => {
        if (requestId === requestSequence.current) {
          setError(err.message || "Unable to load monitoring snapshots.");
        }
      })
      .finally(() => {
        if (requestId === requestSequence.current) {
          setLoading(false);
        }
      });

    return () => {
      requestSequence.current += 1;
    };
  }, [limit]);

  return { snapshots, loading, error, refresh };
}