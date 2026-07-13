import { Link, NavLink, Outlet, useLocation } from "react-router-dom";

/** Line-drawn pitch glyph — the PitchLab mark. */
export function PitchMark({ className = "h-5 w-8" }: { className?: string }) {
  return (
    <svg viewBox="0 0 48 30" fill="none" className={className} aria-hidden>
      <rect x="1" y="1" width="46" height="28" rx="2" stroke="currentColor" strokeWidth="2" />
      <line x1="24" y1="1" x2="24" y2="29" stroke="currentColor" strokeWidth="2" />
      <circle cx="24" cy="15" r="5" stroke="currentColor" strokeWidth="2" />
      <rect x="1" y="8" width="7" height="14" stroke="currentColor" strokeWidth="2" />
      <rect x="40" y="8" width="7" height="14" stroke="currentColor" strokeWidth="2" />
    </svg>
  );
}

const labTabs = [
  { to: "/lab", label: "Runs", end: true },
  { to: "/lab/new", label: "New run", end: false },
  { to: "/lab/qa", label: "QA queue", end: false },
  { to: "/lab/datasets", label: "Datasets", end: false },
];

export function Shell() {
  const { pathname } = useLocation();
  const inLab = pathname.startsWith("/lab");

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-40 border-b border-white/8 bg-turf-950/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-[1600px] items-center gap-6 px-5">
          <Link to="/" className="flex items-center gap-2.5 text-ink-100">
            <PitchMark className="h-5 w-8 text-volt-400" />
            <span className="text-[15px] font-semibold tracking-tight">
              Pitch<span className="text-volt-400">Lab</span>
            </span>
          </Link>

          {/* Mode switcher */}
          <nav className="flex items-center rounded-lg border border-white/8 bg-turf-900 p-0.5 text-[13px]">
            <NavLink
              to="/"
              className={
                !inLab
                  ? "rounded-md bg-volt-400 px-3.5 py-1 font-medium text-turf-950"
                  : "rounded-md px-3.5 py-1 text-ink-400 transition-colors hover:text-ink-100"
              }
            >
              App
            </NavLink>
            <NavLink
              to="/lab"
              className={
                inLab
                  ? "rounded-md bg-volt-400 px-3.5 py-1 font-medium text-turf-950"
                  : "rounded-md px-3.5 py-1 text-ink-400 transition-colors hover:text-ink-100"
              }
            >
              Lab
            </NavLink>
          </nav>

          {inLab && (
            <nav className="flex items-center gap-1 text-[13px]">
              {labTabs.map((t) => (
                <NavLink
                  key={t.to}
                  to={t.to}
                  end={t.end}
                  className={({ isActive }) =>
                    isActive
                      ? "rounded-md bg-turf-800 px-3 py-1.5 font-medium text-ink-100"
                      : "rounded-md px-3 py-1.5 text-ink-400 transition-colors hover:bg-turf-900 hover:text-ink-100"
                  }
                >
                  {t.label}
                </NavLink>
              ))}
            </nav>
          )}

          <div className="ml-auto font-mono text-[11px] uppercase tracking-[0.18em] text-ink-500">
            {inLab ? "pipeline lab" : "match analytics"}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1600px] px-5 py-6">
        <Outlet />
      </main>
    </div>
  );
}
