# Video Interaction Analyzer — Edrevel AI Technical Assessment

A service that analyzes lab installation video footage, detects objects, classifies their motion (moving/stationary), and determines when a person interacts with them — with a React frontend to visualize the results.

## Repository Structure

```
edrevel-assessment/
├── backend/          # FastAPI service + CV pipeline
│   ├── main.py        # API endpoints
│   ├── database.py    # SQLite setup
│   ├── models.py       # Task table definition
│   ├── tests/           # pytest suite
│   └── pipeline/       # Modular CV pipeline
│       ├── frame_extractor.py   # Video → frames (OpenCV)
│       ├── detector.py           # Object detection (YOLO)
│       ├── tracker.py            # Cross-frame object tracking (IoU)
│       ├── motion.py             # Moving/stationary classification
│       ├── interaction.py        # Person-object interaction detection
│       └── pipeline.py           # Orchestrates the full pipeline
├── frontend/          # React + TypeScript UI
├── sample_data/       # Seed video (provided sample)
└── README.md
```

## Setup Instructions

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 -m uvicorn main:app --reload
```

Runs at `http://127.0.0.1:8000`. Interactive API docs at `http://127.0.0.1:8000/docs`.

The first run will auto-download YOLOv8n model weights (~6MB).

### Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173`. Requires the backend to be running simultaneously.

### Using it

1. Open `http://localhost:5173`
2. Upload the sample video from `sample_data/`
3. Click "Analyze Video"
4. Wait for processing (roughly 30-90 seconds for the ~8 second sample video, depending on hardware)
5. View detected objects, their motion timelines, and interaction markers

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/videos` | Upload a video, returns `task_id` immediately |
| `GET` | `/api/tasks/{task_id}` | Poll processing status |
| `GET` | `/api/tasks/{task_id}/result` | Get final JSON result (once complete) |
| `GET` | `/api/tasks` | List all tasks |

## Architecture

Video upload is decoupled from processing: `POST /api/videos` saves the file, creates a task record (SQLite), and immediately returns a `task_id` — actual analysis runs in a FastAPI background task. Clients poll `GET /api/tasks/{task_id}` until `status == "complete"`, then fetch results.

The CV pipeline is fully modular — each stage (frame extraction, detection, tracking, motion classification, interaction detection) is an independent, testable module with no knowledge of the others. `pipeline.py` is the only file that orchestrates them together.

## Libraries Used

- **FastAPI** — REST API framework, async background task support
- **SQLAlchemy + SQLite** — lightweight persistence for task state and results
- **OpenCV (`opencv-python-headless`)** — video frame extraction
- **Ultralytics YOLOv8 (nano)** — pre-trained object detection. "Nano" variant chosen for speed given the processing timeline.
- **NumPy** — implicitly used via OpenCV/YOLO
- **React + TypeScript + Vite** — frontend

## Frontend Design

The UI uses a dark slate background with cyan and amber accents — a deliberate choice to feel like a technical/lab dashboard rather than a generic consumer app, fitting the subject matter (analyzing lab equipment footage). Amber marks "moving" segments, slate-gray marks "stationary," and cyan marks interaction ranges, chosen for clear visual contrast against the dark background and against each other.

The core visualization is a per-object horizontal timeline bar: position along the bar corresponds to the frame's position in the video, color indicates motion state, and a thin underline marks when the person was interacting with that object. This was chosen over a plain table of frame ranges because it lets someone see, at a glance, when in the video something happened without reading raw numbers — directly addressing the assessment's request to "visualize object motion using a modern UI paradigm."

## Technical Approach

**Object tracking**: custom IoU (Intersection over Union) based tracker. Matches detections across frames by bounding box overlap, tuned against the sample video (`iou_threshold=0.25`, `max_frames_missing=35` to tolerate brief occlusion).

**Motion classification**: compares each object's bounding box center to its position 12 frames (~0.5s) earlier, rather than frame-to-frame, to smooth out detection jitter while still catching gradual movement. Threshold of 20px was chosen empirically by examining the distribution of frame-to-frame displacement in the sample video.

**Interaction detection**: measures the pixel gap between the person's bounding box and each object's bounding box per frame; proximity within 40px (with a 5-frame gap tolerance to bridge brief detection dropouts) counts as interaction.

## Assumptions & Known Tradeoffs

These are real limitations I found by actually testing against the sample video — not guesses about what might go wrong.

- **Object names aren't always accurate.** The AI model I used (YOLO) was trained to recognize 80 everyday objects — things like people, laptops, bottles, chairs. It was never trained on lab equipment, so when it sees something like a spectrophotometer, it guesses the closest thing it knows (in this case, "laptop," because the screen looks similar). I manually relabel a few of these guesses to more accurate names for display purposes, but the underlying detection still only "knows" its original 80 categories.

- **The cable is never detected as its own object.** Despite being central to the video (the person holds it and plugs it into the spectrophotometer), the cable never appears anywhere in the results — not mislabeled, not even as a low-confidence guess. YOLO simply has no "cable" or "wire" category in its 80 trained classes, and a thin, black, flexible cable doesn't visually resemble any of the other categories it does know, either. The timing of cable-handling actions is still reflected indirectly, though: the spectrophotometer's `interactions` frame ranges (0-12, 22-45) line up with the exact moments he's holding and plugging in the cable, verified against the actual video footage.

- **The robotic arm and microscope assembly are also never detected**, for a related but distinct reason: unlike the cable (which is invisible due to being too small/thin), these are simply too mechanically complex and don't resemble any of YOLO's 80 trained categories even loosely — the model doesn't attempt a guess at all.

- **If the camera moves, it can look like objects are moving too.** Our motion detection assumes the camera stays still. Near the end of the sample video, the camera shifts slightly (zooms out), which makes stationary background items (like glassware on a shelf) appear to "move" in our results — even though they never actually moved. Fixing this properly would require the system to first figure out how much the camera itself moved, then subtract that out before judging whether an object moved.

- **I can't always tell "touching" from "just standing nearby."** Since the model can't specifically detect hands, I approximate by using the person's entire body position instead. This mostly works, but can cause false positives — for example, an object sitting near someone's hip or leg can get flagged as "touched" even if their hands are nowhere near it. A proper fix would need real hand-tracking; due to time constraints I did not implement it.

- **I tested an alternative to fix this, and it didn't work well enough to use.** I tried locating the person's hand by finding the part of their body that was moving the most, frame to frame, instead of using their whole body. Testing this against known real and false interactions showed it wasn't reliable — it kept picking up unrelated movement, like clothing shifting, instead of the hand. I decided not to use it rather than ship something inconsistent.

- **Depth is invisible to a flat video.** "Overlap" in this project means the person's bounding box and an object's bounding box sharing pixels on screen. But a flat 2D video has no sense of "close to the camera" vs. "far away" — so two things can look like they're overlapping on screen even if one is actually much farther from the camera than the other in real life. I proved this with real numbers: one incorrect "touch" result actually had a *higher* overlap percentage (86.5%) than a real, confirmed touch (16.4%) — meaning overlap amount alone can't be used to fix this problem. It's a fundamental limit of using a single flat camera angle instead of a 3D-aware system.

- **Results can shift slightly depending on the computer running it.** I noticed small differences (usually 1-2 frames) in the exact results when running the same video on different machines. This comes down to tiny math differences in how each computer's hardware handles the AI model — not a bug, and the overall patterns stay the same either way.

- **I assume there's only one person in the video.** My system is built to track a single person. If a video had multiple people, the logic would need to be expanded to handle that — not needed for this sample video, so I didn't build it out.

- **I filter out unreliable, one-off detections.** Sometimes the AI briefly "sees" something that isn't really there (a single-frame false guess). I ignore anything that only shows up in fewer than 5 frames total, since that's almost always noise rather than a real object.

## Testing

The project includes a `pytest` suite with 27 tests, covering three levels:

- **Math helper tests** (`tests/test_math_helpers.py`) — unit tests for the pure functions used throughout the pipeline (`compute_iou`, `euclidean_distance`, `get_center`, `box_distance`), verified against hand-calculated expected values.
- **Logic tests** (`tests/test_motion_and_interaction.py`) — tests for `classify_motion_states` and `detect_interactions` using small, synthetic scenarios with known correct outcomes (e.g. verifying that small jitter stays "stationary," that large jumps are classified "moving," and that brief detection gaps are correctly bridged in interaction intervals).
- **API tests** (`tests/test_api.py`) — tests for the actual REST endpoints using FastAPI's `TestClient`, covering file-type validation, task creation, 404 handling for unknown tasks, and the "not complete yet" (409) response.

### Running the tests

```bash
cd backend
source venv/bin/activate
pip install httpx   # required for API test client
python3 -m pytest tests/ -v
```

### Test execution evidence

```
platform darwin -- Python 3.9.6, pytest-8.3.3, pluggy-1.6.0
rootdir: backend/
collected 27 items

tests/test_api.py::TestUploadEndpoint::test_rejects_unsupported_file_type PASSED
tests/test_api.py::TestUploadEndpoint::test_accepts_valid_video_extension_and_returns_task_id PASSED
tests/test_api.py::TestStatusEndpoint::test_returns_404_for_unknown_task PASSED
tests/test_api.py::TestStatusEndpoint::test_returns_status_for_known_task PASSED
tests/test_api.py::TestResultEndpoint::test_returns_404_for_unknown_task PASSED
tests/test_api.py::TestResultEndpoint::test_returns_409_when_task_not_yet_complete PASSED
tests/test_api.py::TestListTasksEndpoint::test_list_tasks_returns_a_list PASSED

tests/test_math_helpers.py::TestComputeIoU::test_identical_boxes_have_iou_of_1 PASSED
tests/test_math_helpers.py::TestComputeIoU::test_non_overlapping_boxes_have_iou_of_0 PASSED
tests/test_math_helpers.py::TestComputeIoU::test_partial_overlap_computed_correctly PASSED
tests/test_math_helpers.py::TestComputeIoU::test_touching_edges_has_iou_of_0 PASSED
tests/test_math_helpers.py::TestEuclideanDistance::test_same_point_has_distance_0 PASSED
tests/test_math_helpers.py::TestEuclideanDistance::test_known_3_4_5_triangle PASSED
tests/test_math_helpers.py::TestEuclideanDistance::test_get_center_of_box PASSED
tests/test_math_helpers.py::TestBoxDistance::test_overlapping_boxes_have_distance_0 PASSED
tests/test_math_helpers.py::TestBoxDistance::test_touching_boxes_have_distance_0 PASSED
tests/test_math_helpers.py::TestBoxDistance::test_horizontally_separated_boxes PASSED
tests/test_math_helpers.py::TestBoxDistance::test_diagonally_separated_boxes PASSED

tests/test_motion_and_interaction.py::TestClassifyMotionStates::test_stationary_object_stays_in_one_place PASSED
tests/test_motion_and_interaction.py::TestClassifyMotionStates::test_object_that_moves_far_is_classified_moving PASSED
tests/test_motion_and_interaction.py::TestClassifyMotionStates::test_single_frame_defaults_to_stationary PASSED
tests/test_motion_and_interaction.py::TestClassifyMotionStates::test_small_jitter_under_threshold_stays_stationary PASSED
tests/test_motion_and_interaction.py::TestDetectInteractions::test_no_overlap_produces_no_interactions PASSED
tests/test_motion_and_interaction.py::TestDetectInteractions::test_overlapping_boxes_produce_one_interaction PASSED
tests/test_motion_and_interaction.py::TestDetectInteractions::test_gap_tolerance_bridges_brief_missing_detections PASSED
tests/test_motion_and_interaction.py::TestDetectInteractions::test_large_gap_produces_separate_intervals PASSED
tests/test_motion_and_interaction.py::TestDetectInteractions::test_person_not_visible_in_frame_is_skipped PASSED

======================== 27 passed, 14 warnings in 2.06s ========================
```

(The 14 warnings are `PyparsingDeprecationWarning` messages from `matplotlib`, a dependency of `ultralytics` — unrelated to this project's code and with no functional impact.)

## Time Spent

I had 2 days to complete this and I spent 15 hours on it.

## Bonus Scope

Keyframe extraction was not implemented given the project timeline; the core required functionality (detection, tracking, motion classification, interaction detection, async API, persistence, and frontend visualization) was prioritized.