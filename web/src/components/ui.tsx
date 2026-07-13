import type { ReactNode } from "react";
import type { RunStatus, Team } from "../lib/types";
import { teamColor } from "../lib/colors";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`rounded-xl border border-white/8 bg-turf-900 ${className}`}>{children}</div>
  );
}

export function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">{children}</div>
  );
}

export function PageTitle({ title, sub, right }: { title: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {sub && <p className="mt-1 text-sm text-ink-400">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled,
  type = "button",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const styles = {
    primary:
      "bg-volt-400 text-turf-950 font-medium hover:bg-volt-300 disabled:bg-turf-700 disabled:text-ink-500",
    ghost:
      "border border-white/10 text-ink-100 hover:bg-turf-800 disabled:text-ink-500 disabled:hover:bg-transparent",
    danger:
      "border border-team-away/40 text-team-away hover:bg-team-away/10 disabled:opacity-40",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`rounded-lg px-3.5 py-1.5 text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-volt-400 ${styles} ${className}`}
    >
      {children}
    </button>
  );
}

const STATUS_STYLES: Record<RunStatus, { dot: string; text: string; label: string }> = {
  queued: { dot: "bg-ink-500", text: "text-ink-400", label: "Queued" },
  running: { dot: "bg-volt-400 animate-pulse-dot", text: "text-volt-400", label: "Running" },
  completed: { dot: "bg-volt-500", text: "text-ink-100", label: "Completed" },
  failed: { dot: "bg-team-away", text: "text-team-away", label: "Failed" },
};

export function StatusChip({ status }: { status: RunStatus }) {
  const s = STATUS_STYLES[status];
  return (
    <span className={`inline-flex items-center gap-1.5 text-[13px] ${s.text}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}

export function TeamDot({ team, className = "" }: { team: Team; className?: string }) {
  return (
    <span
      className={`inline-block h-2 w-2 rounded-full ${className}`}
      style={{ background: teamColor(team) }}
    />
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-12 text-ink-400 justify-center text-sm">
      <svg className="h-4 w-4 animate-spin text-volt-400" viewBox="0 0 24 24" fill="none">
        <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.2" />
        <path d="M22 12a10 10 0 0 0-10-10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
      {label ?? "Loading"}
    </div>
  );
}

export function EmptyState({ title, hint, action }: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-white/10 py-16 text-center">
      <svg viewBox="0 0 48 30" fill="none" className="h-8 w-12 text-turf-700" aria-hidden>
        <rect x="1" y="1" width="46" height="28" rx="2" stroke="currentColor" strokeWidth="2" />
        <line x1="24" y1="1" x2="24" y2="29" stroke="currentColor" strokeWidth="2" />
        <circle cx="24" cy="15" r="5" stroke="currentColor" strokeWidth="2" />
      </svg>
      <div className="text-sm text-ink-100">{title}</div>
      {hint && <div className="max-w-sm text-[13px] text-ink-400">{hint}</div>}
      {action}
    </div>
  );
}

export function ErrorNote({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-auto rounded-lg border border-team-away/30 bg-team-away/5 p-3 font-mono text-xs text-team-away">
      {children}
    </div>
  );
}

export function Tabs<T extends string>({
  tabs,
  active,
  onChange,
}: {
  tabs: { id: T; label: string; count?: number }[];
  active: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="flex items-center gap-1 border-b border-white/8 px-1">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => onChange(t.id)}
          className={`relative -mb-px border-b-2 px-3 py-2 text-[13px] transition-colors ${
            active === t.id
              ? "border-volt-400 text-ink-100"
              : "border-transparent text-ink-400 hover:text-ink-100"
          }`}
        >
          {t.label}
          {t.count != null && t.count > 0 && (
            <span className="ml-1.5 rounded-full bg-turf-800 px-1.5 py-0.5 font-mono text-[10px] text-ink-400">
              {t.count}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

export function Mono({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`font-mono text-xs ${className}`}>{children}</span>;
}
