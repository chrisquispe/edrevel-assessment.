// All communication with our FastAPI backend lives here, in one place.
// Components call these functions instead of writing fetch() calls
// directly -- if the backend URL or request shape ever changes, we only
// update it here.

import type { TaskStatusResponse, PipelineResult } from "./types";

const API_BASE = "http://127.0.0.1:8000";

export async function uploadVideo(file: File): Promise<{ task_id: string; status: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/api/videos`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  const response = await fetch(`${API_BASE}/api/tasks/${taskId}`);

  if (!response.ok) {
    throw new Error(`Status check failed: ${response.statusText}`);
  }

  return response.json();
}

export async function getTaskResult(taskId: string): Promise<PipelineResult> {
  const response = await fetch(`${API_BASE}/api/tasks/${taskId}/result`);

  if (!response.ok) {
    throw new Error(`Result fetch failed: ${response.statusText}`);
  }

  return response.json();
}