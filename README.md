# SpatialMind — Cloud Middleware

FastAPI server that bridges a Unitree Go2 robot (running DimOS) to a live web dashboard with semantic querying via AWS Bedrock.

## Architecture

```
[Laptop / DimOS]                [EC2 :8080]              [Phone / Browser]
  nav_bridge.py ── POST ──────→  /ingest/pose ←── GET ──  /pose/live
  pc_bridge.py  ── POST ──────→  /ingest/pointcloud ←── GET ── /pointcloud/live
  dimos_bridge  ── POST ──────→  /ingest       ←── GET ──  /
  cloud_client  ── POST ──────→  /frames       ← POST ──→  /query/stream
                                                   POST ──→ /navigate
```

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI server — all endpoints + dashboard HTML |
| `nav_bridge.py` | Direct LCM bridge: `/odom` to `/ingest/pose`, queued goals to `/goal_request` |
| `pc_bridge.py` | Direct LCM bridge: `/lidar` pointcloud to `/ingest/pointcloud` |
| `camera_bridge.py` | Camera frame pusher — auto-detects source (DimOS pSHM / RTSP / USB) |
| `dimos_bridge.py` | Reads DimOS MCP perception tools, pushes detected objects to cloud |
| `cloud_client.py` | `CloudClient` class — reusable push client for robot-side integration |
| `agent.py` | AWS Bedrock agent loop — natural language queries over the semantic map |
| `world_store.py` | Semantic map persistence — memory backend or S3 backend |
| `models.py` | Pydantic models shared across the API |

## Quickstart (local)

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
# Dashboard at http://localhost:8080
```

## Deploy on EC2

```bash
# On EC2 (Ubuntu 24.04, t3.small, port 8080 open in Security Group)
git clone https://github.com/your/repo.git && cd robohack_fastapi
pip install -r requirements.txt

export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-north-1
export S3_BUCKET=your-bucket-name
export USE_MEMORY_STORE=false   # omit to use in-memory store (no AWS needed)

# Run in background
screen -S fastapi
uvicorn main:app --host 0.0.0.0 --port 8080
# Ctrl+A D to detach
```

## Camera stream

`camera_bridge.py` pushes frames to `/frames` so the dashboard shows live video.

**DimOS transport is platform-dependent:**
- **Linux** → `LCMTransport` (UDP multicast) — confirmed working
- **Mac** → `pSHMTransport` (shared memory) — `--source dimos` falls back to pSHM

Must run inside the dimos venv (`source dimos/.venv/bin/activate`) for `--source dimos`.

```bash
# Auto-detect (LCM/pSHM → RTSP → USB webcam):
python camera_bridge.py

# DimOS simulation (Linux: LCM multicast, Mac: pSHM):
python camera_bridge.py --source dimos

# Real Go2 over network:
python camera_bridge.py --source rtsp --rtsp-url rtsp://192.168.123.161:8554/video

# USB webcam — no DimOS needed, good for testing:
python camera_bridge.py --source opencv --device 0

# Push to EC2:
python camera_bridge.py --cloud-url http://<ec2-ip>:8080
```

Stream visible at `/frames/go2_a/stream` (MJPEG) and `/frames/go2_a` (latest JPEG).

## Run the bridges (laptop, with DimOS running)

```bash
# Point nav/pointcloud bridges at EC2
python nav_bridge.py --cloud-url http://<ec2-public-ip>:8080
python pc_bridge.py --cloud-url http://<ec2-public-ip>:8080

# Or locally
python nav_bridge.py --cloud-url http://localhost:8080
python pc_bridge.py --cloud-url http://localhost:8080
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `USE_MEMORY_STORE` | `true` | `false` to persist map to S3 |
| `S3_BUCKET` | `robohack-map` | S3 bucket name |
| `AWS_REGION` | `eu-west-1` | AWS region (use `eu-north-1` for Stockholm) |
| `PC_ACCUM_VOXEL_CM` | `8` | Voxel size for accumulated lidar map |
| `PC_ACCUM_MAX_POINTS` | `120000` | Maximum accumulated lidar voxels served to the UI |
| `AWS_ACCESS_KEY_ID` | — | IAM credentials for Bedrock + S3 |
| `AWS_SECRET_ACCESS_KEY` | — | IAM credentials for Bedrock + S3 |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live dashboard (map + chat + objects) |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Robot pushes detected objects |
| `POST` | `/ingest/pose` | Nav bridge pushes live odometry pose |
| `POST` | `/ingest/pointcloud` | Pointcloud bridge pushes compressed lidar points |
| `GET` | `/map` | Merged semantic map (all robots) |
| `GET` | `/pose/live` | Latest robot pose by robot id |
| `GET` | `/pointcloud/live` | Accumulated lidar voxel pointcloud by robot id |
| `POST` | `/query/stream` | SSE — natural language query via Bedrock |
| `POST` | `/navigate` | Send navigation goal to robot |
| `POST` | `/frames` | Robot pushes JPEG camera frame |
| `GET` | `/frames/{robot_id}` | Latest camera frame |
| `GET` | `/frames/{robot_id}/stream` | MJPEG stream |

## Bedrock model

Uses `eu.anthropic.claude-sonnet-4-6` (cross-region inference profile, eu-north-1).
Requires IAM permissions: `bedrock:InvokeModelWithResponseStream`.
