// Types matching the exact JSON schema our backend returns.

export interface MotionSegment {
  frame_range: [number, number];
  state: "moving" | "stationary";
}

export interface Interaction {
  interacted_by_person: number;
  frame_start: number;
  frame_end: number;
}

export interface DetectedObject {
  object_id: number;
  class: string;
  motion_history: MotionSegment[];
  interactions: Interaction[];
}

export interface PersonDetected {
  person_id: number;
  frames_present: [number, number] | [];
  total_frames_detected: number;
}

export interface VideoMetadata {
  duration_seconds: number;
  frame_count: number;
  resolution: string;
  fps: number;
}

export interface PipelineResult {
  videoMetadata: VideoMetadata;
  objectsDetected: DetectedObject[];
  personsDetected: PersonDetected[];
}

export type TaskStatus = "pending" | "processing" | "complete" | "failed";

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  error_message: string | null;
}