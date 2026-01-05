export type StreamState = "idle" | "starting" | "active" | "error";
export type ModelState = "idle" | "loading" | "ready" | "error";
export type EventKind = "video" | "audio" | "system";
export type Severity = "info" | "warn" | "error";

export type EventRecord = {
  id: string;
  ts: string;
  message: string;
  severity: Severity;
  kind: EventKind;
};

export type EventCategory = "audio" | "gaze" | "faces" | "gadgets" | "system";

export type PostEventFn = (
  key: string,
  category: EventCategory,
  kind: EventKind,
  severity: Severity,
  message: string
) => void;

export type IndividualFlags = {
  audio: boolean;
  gaze: boolean;
  faces: boolean;
  gadgets: boolean;
  overlays: boolean;
};
