
import {
  ArrowDownRight,
  ArrowUpRight,
} from "lucide-react";

function MetricCard({
  title,
  value,
  change,
  positive,
  icon,
  description,
}) {
  return (
    <div className="panel metric-card">
      <div className="metric-top">
        <span>{title}</span>
        <div className="metric-icon">{icon}</div>
      </div>

      <strong className="metric-value">{value}</strong>

      <div className="metric-bottom">
        <span
          className={
            positive ? "change-positive" : "change-negative"
          }
        >
          {positive ? (
            <ArrowUpRight size={14} />
          ) : (
            <ArrowDownRight size={14} />
          )}

          {change}
        </span>

        <span>{description}</span>
      </div>
    </div>
  );
}

export default MetricCard;
