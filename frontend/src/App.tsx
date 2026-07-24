import { useState, useEffect, useRef } from "react";
import { uploadVideo, getTaskStatus, getTaskResult } from "./api";
import type { PipelineResult, TaskStatus } from "./types";
import "./App.css";

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollingRef = useRef<number | null>(null);

  // Polling: once we have a taskId and it's not yet complete/failed,
  // check its status every 2 seconds. This mirrors exactly what we did
  // manually in Swagger UI -- just automated.
  useEffect(() => {
    if (!taskId || status === "complete" || status === "failed") {
      return;
    }

    pollingRef.current = window.setInterval(async () => {
      try {
        const statusResponse = await getTaskStatus(taskId);
        setStatus(statusResponse.status);

        if (statusResponse.status === "complete") {
          const resultResponse = await getTaskResult(taskId);
          setResult(resultResponse);
        } else if (statusResponse.status === "failed") {
          setError(statusResponse.error_message ?? "Processing failed.");
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    }, 2000);

    // Cleanup: stop polling if the component unmounts or taskId changes.
    return () => {
      if (pollingRef.current) window.clearInterval(pollingRef.current);
    };
  }, [taskId, status]);

  async function handleUpload() {
    if (!file) return;
    setError(null);
    setResult(null);
    try {
      const response = await uploadVideo(file);
      setTaskId(response.task_id);
      setStatus("pending");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Video Interaction Analyzer</h1>
        <p className="subtitle">
          Detects objects, tracks motion, and identifies human interaction in video footage.
        </p>
      </header>

      <section className="upload-panel">
        <input
          type="file"
          accept=".mp4,.mov,.avi,.mkv"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button onClick={handleUpload} disabled={!file || (status !== null && status !== "complete" && status !== "failed")}>
          {status === "pending" || status === "processing" ? "Analyzing..." : "Analyze Video"}
        </button>
      </section>

      {status && status !== "complete" && status !== "failed" && (
        <div className="status-banner status-processing">
          <span className="spinner" />
          Status: {status} — this may take a minute for the full pipeline to run.
        </div>
      )}

      {error && <div className="status-banner status-error">Error: {error}</div>}

      {result && <ResultsView result={result} />}
    </div>
  );
}

function ResultsView({ result }: { result: PipelineResult }) {
  const totalFrames = result.videoMetadata.frame_count;
  const person = result.personsDetected[0];

  return (
    <div className="results">
      <section className="metadata-panel">
        <div className="metadata-item">
          <span className="metadata-label">Duration</span>
          <span className="metadata-value">{result.videoMetadata.duration_seconds}s</span>
        </div>
        <div className="metadata-item">
          <span className="metadata-label">Frames</span>
          <span className="metadata-value">{totalFrames}</span>
        </div>
        <div className="metadata-item">
          <span className="metadata-label">Resolution</span>
          <span className="metadata-value">{result.videoMetadata.resolution}</span>
        </div>
        <div className="metadata-item">
          <span className="metadata-label">Person Detected</span>
          <span className="metadata-value">
            {person ? `${person.total_frames_detected}/${totalFrames} frames` : "None"}
          </span>
        </div>
      </section>

      <h2>Detected Objects ({result.objectsDetected.length})</h2>
      <div className="object-list">
        {result.objectsDetected.map((obj) => (
          <ObjectCard key={obj.object_id} obj={obj} totalFrames={totalFrames} />
        ))}
      </div>
    </div>
  );
}

function ObjectCard({ obj, totalFrames }: { obj: PipelineResult["objectsDetected"][0]; totalFrames: number }) {
  const wasTouched = obj.interactions.length > 0;

  return (
    <div className="object-card">
      <div className="object-card-header">
        <span className="object-class">{obj.class}</span>
        <span className={`interaction-badge ${wasTouched ? "touched" : "untouched"}`}>
          {wasTouched ? "Touched" : "Not touched"}
        </span>
      </div>

      <div className="timeline" title="Motion over time">
        {obj.motion_history.map((segment, i) => {
          const [start, end] = segment.frame_range;
          const widthPct = ((end - start + 1) / totalFrames) * 100;
          const leftPct = (start / totalFrames) * 100;
          return (
            <div
              key={i}
              className={`timeline-segment ${segment.state}`}
              style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              title={`Frames ${start}-${end}: ${segment.state}`}
            />
          );
        })}
        {obj.interactions.map((interaction, i) => {
          const widthPct = ((interaction.frame_end - interaction.frame_start + 1) / totalFrames) * 100;
          const leftPct = (interaction.frame_start / totalFrames) * 100;
          return (
            <div
              key={`interact-${i}`}
              className="interaction-marker"
              style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
              title={`Interaction: frames ${interaction.frame_start}-${interaction.frame_end}`}
            />
          );
        })}
      </div>

      <div className="timeline-legend">
        <span><span className="legend-dot moving" /> Moving</span>
        <span><span className="legend-dot stationary" /> Stationary</span>
        <span><span className="legend-dot interaction" /> Interaction</span>
      </div>
    </div>
  );
}

export default App;