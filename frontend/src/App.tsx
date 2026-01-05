import { useCallback, useMemo, useRef, useState } from "react";
import "./global.css";
import IndividualProctor from "./proctor/IndividualProctor";
import type { EventCategory, EventKind, EventRecord, IndividualFlags, Severity } from "./proctor/types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

const defaultIndividualFlags: IndividualFlags = {
  audio: true,
  gaze: true,
  faces: true,
  gadgets: true,
  overlays: true,
};

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
  const [individualFlags, setIndividualFlags] = useState<IndividualFlags>(defaultIndividualFlags);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const lastEventRef = useRef<Record<string, number>>({});

  const isFlagEnabled = useCallback(
    (category: EventCategory) => {
      return individualFlags[category as keyof IndividualFlags] ?? true;
    },
    [individualFlags]
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

  const flagPanel = (
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
      <FlagToggle
        label="Show bounding boxes"
        note="Draw face and gadget boxes on the feed"
        checked={individualFlags.overlays}
        onChange={(checked) => setIndividualFlags((prev) => ({ ...prev, overlays: checked }))}
      />
    </div>
  );

  return (
    <main className="app-shell">
      <header className="page-header">
        <div>
          <p className="eyebrow">Proctoring prototype</p>
          <h1>Individual proctoring</h1>
          <p className="lede">Backend YOLO face counts plus on-device gadget checks for solo test takers.</p>
        </div>
        <div className="session-tag">Session {sessionId.slice(-6)}</div>
      </header>

      <section className="mode-selector">
        {flagPanel}
      </section>

      <IndividualProctor flags={individualFlags} postEvent={postEvent} />

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
