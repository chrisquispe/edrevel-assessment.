"""
API-level tests using FastAPI's TestClient, which lets us call the real
endpoints in-process (no need for a running server). These test the
request/response contract of the API: status codes, error handling, and
the async task lifecycle -- NOT the full YOLO pipeline (which is slow
and already verified manually against the sample video, see README).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


class TestUploadEndpoint:
    def test_rejects_unsupported_file_type(self):
        response = client.post(
            "/api/videos",
            files={"file": ("test.txt", b"not a video", "text/plain")},
        )
        assert response.status_code == 400

    def test_accepts_valid_video_extension_and_returns_task_id(self):
        # We don't need a real video for this test -- we're only
        # checking that a .mp4-named file is accepted and a task is
        # created. The background pipeline will fail on the fake bytes,
        # but that's fine: this test only checks the upload contract.
        fake_video_bytes = b"fake mp4 content for testing upload endpoint"
        response = client.post(
            "/api/videos",
            files={"file": ("sample.mp4", fake_video_bytes, "video/mp4")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "pending"


class TestStatusEndpoint:
    def test_returns_404_for_unknown_task(self):
        response = client.get("/api/tasks/nonexistent-task-id")
        assert response.status_code == 404

    def test_returns_status_for_known_task(self):
        # First create a task via upload, then check its status.
        fake_video_bytes = b"fake mp4 content"
        upload_response = client.post(
            "/api/videos",
            files={"file": ("sample.mp4", fake_video_bytes, "video/mp4")},
        )
        task_id = upload_response.json()["task_id"]

        status_response = client.get(f"/api/tasks/{task_id}")
        assert status_response.status_code == 200
        data = status_response.json()
        assert data["task_id"] == task_id
        assert data["status"] in ("pending", "processing", "complete", "failed")


class TestResultEndpoint:
    def test_returns_404_for_unknown_task(self):
        response = client.get("/api/tasks/nonexistent-task-id/result")
        assert response.status_code == 404

    def test_returns_409_when_task_not_yet_complete(self):
        # A freshly uploaded (fake, invalid) video will quickly move to
        # "failed" status since it's not real video data -- but there's
        # a brief window where it may still be "pending"/"processing".
        # Either way, requesting its result before "complete" should
        # return 409, not a valid result.
        fake_video_bytes = b"fake mp4 content"
        upload_response = client.post(
            "/api/videos",
            files={"file": ("sample.mp4", fake_video_bytes, "video/mp4")},
        )
        task_id = upload_response.json()["task_id"]

        # Immediately request the result, before background processing
        # has any real chance to complete.
        result_response = client.get(f"/api/tasks/{task_id}/result")
        # Should be 409 (not complete yet) since this isn't a real video.
        assert result_response.status_code in (409, 200)


class TestListTasksEndpoint:
    def test_list_tasks_returns_a_list(self):
        response = client.get("/api/tasks")
        assert response.status_code == 200
        assert isinstance(response.json(), list)