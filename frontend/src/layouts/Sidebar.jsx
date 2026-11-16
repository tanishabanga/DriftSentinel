
import {
  Activity,
  ChevronDown,
  CircleHelp,
  LayoutDashboard,
  Settings,
  Shield,
  ShieldAlert,
  SlidersHorizontal,
  Zap,
} from "lucide-react";

import { NavLink } from "react-router-dom";

function Sidebar() {
  const navClass = ({ isActive }) =>
    `nav-item ${isActive ? "active" : ""}`;

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-icon"><Shield size={21} /></div>
        <div>
          <h1>DriftSentinel</h1>
          <span>FRAUD INTELLIGENCE</span>
        </div>
      </div>

      <div className="workspace-label">WORKSPACE</div>

      <nav className="navigation">
        <NavLink to="/" end className={navClass}>
          <LayoutDashboard size={18} /> Overview
        </NavLink>

        <NavLink to="/transactions" className={navClass}>
          <Activity size={18} /> Transactions
        </NavLink>

        <NavLink to="/risk-analysis" className={navClass}>
          <ShieldAlert size={18} /> Risk Analysis
        </NavLink>

        <NavLink to="/model-analytics" className={navClass}>
          <SlidersHorizontal size={18} /> Model Analytics
        </NavLink>

        <NavLink to="/prediction" className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}>
            <Zap size={18} />
            Run Prediction
        </NavLink>
      </nav>

      <div className="workspace-label system-label">SYSTEM</div>

      <nav className="navigation">
        <NavLink to="/settings" className={navClass}>
          <Settings size={18} /> Settings
        </NavLink>

        <NavLink to="/documentation" className={navClass}>
          <CircleHelp size={18} /> Documentation
        </NavLink>
      </nav>

      <div className="sidebar-bottom">
        <div className="system-status">
          <span className="status-dot" />
          <div>
            <strong>System operational</strong>
            <span>All services running</span>
          </div>
        </div>

        <div className="profile">
          <div className="avatar">DS</div>
          <div className="profile-info">
            <strong>DriftSentinel Admin</strong>
            <span>Administrator</span>
          </div>
          <ChevronDown size={16} />
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
