"""
main.py

The FastAPI application. Defines three endpoints:

  POST /api/videos                 -> upload a video, creates a task, returns task_id
  GET  /api/tasks/{task_id}        -> check a task's status
  GET  /api/tasks/{task_id}/result -> get the final JSON result (once complete)

Design note (why this shape):
Video processing takes real time (seconds to minutes), and an HTTP request
shouldn't sit open that long. So POST /api/videos does the FAST part only
(save the file, create a DB row, return immediately) and hands the SLOW
part (actual processing) to a background task. The frontend polls the
GET /api/tasks/{task_id} endpoint until status == "complete", then fetches
the result.
"""

import json
import os
import shutil
import time
import uuid

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import Task

# Creates the tasks.db file and the `tasks` table if they don't exist yet.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Edrevel Video Processing Assessment")

# CORS: by default, browsers block a webpage on one origin (e.g.
# localhost:3000, where React runs) from calling an API on a different
# origin (e.g. localhost:8000, where this FastAPI server runs). This
# middleware explicitly allows that.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def run_pipeline(task_id: str):
    """
    Runs the REAL video-processing pipeline in the background after
    upload: reads the video, runs detection/tracking/motion/interaction
    analysis, and saves the resulting JSON to the task's database row.
    """
    from database import SessionLocal
    from pipeline.pipeline import run_full_pipeline

    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "processing"
        db.commit()

        result = run_full_pipeline(task.video_path)

        task.status = "complete"
        task.result_json = json.dumps(result)
        db.commit()
    except Exception as e:
        task.status = "failed"
        task.error_message = str(e)
        db.commit()
    finally:
        db.close()


@app.post("/api/videos")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Accepts a video file, saves it to disk, creates a task record with
    status "pending", schedules background processing, and immediately
    returns the task_id -- without waiting for processing to finish.
    """
    allowed_extensions = (".mp4", ".mov", ".avi", ".mkv")
    if not file.filename.lower().endswith(allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}",
        )

    task_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{task_id}_{file.filename}")

    # Stream the uploaded file to disk in chunks rather than loading the
    # whole thing into memory at once -- important for larger video files.
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    task = Task(id=task_id, status="pending", video_path=video_path)
    db.add(task)
    db.commit()

    # Schedule the pipeline to run AFTER this response is sent back to the
    # client. This is what makes the upload feel instant.
    background_tasks.add_task(run_pipeline, task_id)

    return {"task_id": task_id, "status": task.status}


@app.get("/api/tasks/{task_id}")
def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Returns the current status of a task. The frontend polls this."""
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task.id,
        "status": task.status,
        "error_message": task.error_message,
    }


@app.get("/api/tasks/{task_id}/result")
def get_task_result(task_id: str, db: Session = Depends(get_db)):
    """
    Returns the final structured JSON result. Only meaningful once
    status == "complete".
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Task is not complete yet (current status: {task.status})",
        )

    return json.loads(task.result_json)


@app.get("/api/tasks")
def list_tasks(db: Session = Depends(get_db)):
    """Lists all tasks -- convenient for a dashboard view in the frontend."""
    tasks = db.query(Task).all()
    return [{"task_id": t.id, "status": t.status, "created_at": t.created_at} for t in tasks]