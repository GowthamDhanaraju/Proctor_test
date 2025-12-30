import { useCallback, useMemo, useRef, useState } from "react";
import "./global.css";
import IndividualProctor from "./proctor/IndividualProctor";
import TeamProctor from "./proctor/TeamProctor";
import type {
  EventCategory,
  EventKind,
  EventRecord,
  IndividualFlags,
  Severity,
  TeamFlags,
} from "./proctor/types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

type ProctorMode = "individual" | "team";

const defaultIndividualFlags: IndividualFlags = {
  audio: true,
  gaze: true,
  faces: true,
  gadgets: true,
};

const defaultTeamFlags: TeamFlags = {
  capacity: true,
  presence: true,
  gaze: false,
};

const modeMeta: Record<ProctorMode, { title: string; copy: string }> = {
  individual: {
    title: "Individual proctoring",
    copy: "YOLO object checks for gadgets, MediaPipe gaze for solo test takers.",
  },
  team: {
    title: "Team proctoring",
    copy: "Headcount cap with face presence checks for small groups.",
  },
};

const ModeToggle = ({ mode, onChange }: { mode: ProctorMode; onChange: (next: ProctorMode) => void }) => (
  <div className="mode-toggle">
    {(["individual", "team"] as ProctorMode[]).map((value) => (
      <button
        key={value}
        type="button"
        className={`mode-tab ${mode === value ? "active" : ""}`}
        onClick={() => onChange(value)}
      >
        {modeMeta[value].title}
      </button>
    ))}
  </div>
);

const FlagToggle = ({
  label,
  checked,
  onChange,
  note,
}: {
  label: string;
  checked: boolean;
  note?: string;
  onChange: (next: boolean) => void;
}) => (
  <label className="flag-toggle">
    <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
    <div>
      <p className="flag-label">{label}</p>
      {note && <p className="flag-note">{note}</p>}
    </div>
  </label>
);

function App() {
  const sessionId = useMemo(
    () => (typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}`),
    []
  );
  const [mode, setMode] = useState<ProctorMode>("individual");
  const [teamLimit, setTeamLimit] = useState(3);
  const [individualFlags, setIndividualFlags] = useState<IndividualFlags>(defaultIndividualFlags);
  const [teamFlags, setTeamFlags] = useState<TeamFlags>(defaultTeamFlags);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const lastEventRef = useRef<Record<string, number>>({});

  const isFlagEnabled = useCallback(
    (category: EventCategory) => {
      if (mode === "individual") {
        return individualFlags[category as keyof IndividualFlags] ?? true;
      }
      return teamFlags[category as keyof TeamFlags] ?? true;
    },
    [individualFlags, mode, teamFlags]
  );

  const postEvent = useCallback(
    async (key: string, category: EventCategory, kind: EventKind, severity: Severity, message: string) => {
      if (!isFlagEnabled(category)) return;
      const now = Date.now();
      const last = lastEventRef.current[key];
      if (last && now - last < 4000) {
        return;
      }
      lastEventRef.current[key] = now;

      const entry: EventRecord = {
        id: `${now}-${Math.random().toString(16).slice(2)}`,
        ts: new Date(now).toISOString(),
        message,
        severity,
        kind,
      };
      setEvents((current) => [...current.slice(-19), entry]);

      try {
        await fetch(`${API_BASE}/events`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: sessionId,
            kind,
            severity,
            message,
            ts: entry.ts,
          }),
        });
      } catch (error) {
        console.warn("Failed to send event to backend", error);
      }
    },
    [isFlagEnabled, sessionId]
  );

  const flagPanel = mode === "individual" ? (
    <div className="flag-grid">
      <FlagToggle
        label="Audio flags"
        note="Speech vs silence detection"
        checked={individualFlags.audio}
        onChange={(checked) => setIndividualFlags((prev) => ({ ...prev, audio: checked }))}
      />
      <FlagToggle
        label="Gaze flags"
        note="Yaw / head turn alerts"
        checked={individualFlags.gaze}
        onChange={(checked) => setIndividualFlags((prev) => ({ ...prev, gaze: checked }))}
      />
      <FlagToggle
        label="Face count flags"
        note="Single vs multiple faces"
        checked={individualFlags.faces}
        onChange={(checked) => setIndividualFlags((prev) => ({ ...prev, faces: checked }))}
      />
      <FlagToggle
        label="YOLO gadget flags"
        note="Phones, laptops, remotes, monitors"
        checked={individualFlags.gadgets}
        onChange={(checked) => setIndividualFlags((prev) => ({ ...prev, gadgets: checked }))}
      />
    </div>
  ) : (
    <div className="flag-grid">
      <FlagToggle
        label="Headcount cap"
        note="Raise a flag when more than allowed join"
        checked={teamFlags.capacity}
        onChange={(checked) => setTeamFlags((prev) => ({ ...prev, capacity: checked }))}
      />
      <FlagToggle
        label="Presence flags"
        note="Warn when nobody is in frame"
        checked={teamFlags.presence}
        onChange={(checked) => setTeamFlags((prev) => ({ ...prev, presence: checked }))}
      />
      <FlagToggle
        label="Gaze drift"
        note="Flag head turns for the primary speaker"
        checked={teamFlags.gaze}
        onChange={(checked) => setTeamFlags((prev) => ({ ...prev, gaze: checked }))}
      />
      <label className="flag-toggle numeric">
        <div>
          <p className="flag-label">Teammate limit</p>
          <p className="flag-note">Allowed people in frame</p>
        </div>
        <input
          type="number"
          min={1}
          max={10}
          value={teamLimit}
          onChange={(e) => {
            const next = Number(e.target.value) || 1;
            setTeamLimit(Math.min(10, Math.max(1, next)));
          }}
        />
      </label>
    </div>
  );

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Proctoring prototype</p>
          <h1>{modeMeta[mode].title}</h1>
          <p className="lede">{modeMeta[mode].copy}</p>
        </div>
        <div className="session-tag">Session {sessionId.slice(-6)}</div>
      </header>

      <section className="mode-selector">
        <ModeToggle mode={mode} onChange={setMode} />
        {flagPanel}
      </section>

      {mode === "individual" ? (
        <IndividualProctor flags={individualFlags} postEvent={postEvent} />
      ) : (
        <TeamProctor teamLimit={teamLimit} flags={teamFlags} postEvent={postEvent} />
      )}

      <section className="event-stream">
        <div className="stream-header">
          <div>
            <p className="label">Events</p>
            <h2>Recent telemetry</h2>
          </div>
          <div className="actions">
            <button type="button" className="ghost" onClick={() => setEvents([])} disabled={!events.length}>
              Clear events
            </button>
          </div>
        </div>
        {events.length === 0 ? (
          <p className="muted">No events yet. Move, speak, or add another person to trigger telemetry.</p>
        ) : (
          <ul className="event-list">
            {events
              .slice()
              .reverse()
              .map((event) => (
                <li key={event.id} className={`event-row severity-${event.severity}`}>
                  <div>
                    <p className="event-kind">{event.kind}</p>
                    <p className="event-message">{event.message}</p>
                  </div>
                  <p className="event-ts">{new Date(event.ts).toLocaleTimeString()}</p>
                </li>
              ))}
          </ul>
        )}
      </section>
    </main>
  );
}

export default App;
