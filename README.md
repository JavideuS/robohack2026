# SpatialMind — Cloud Middleware

FastAPI server that bridges a Unitree Go2 robot (running DimOS) to a live web dashboard with semantic querying via AWS Bedrock.

## Architecture

```
[Laptop / DimOS]                [EC2 :8080]              [Phone / Browser]
  ws_bridge.py  ── POST ──────→  /ingest/map  ←── GET ──  /map/live
  dimos_bridge  ── POST ──────→  /ingest       ←── GET ──  /
  cloud_client  ── POST ──────→  /frames       ← POST ──→  /query/stream
                                                   POST ──→ /navigate
```

## Files

| File | Purpose |
|---|---|
| `main.py` | FastAPI server — all endpoints + dashboard HTML |
| `ws_bridge.py` | Connects to DimOS Socket.IO (port 7779), relays costmap/path/pose to cloud |
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

## Run the bridge (laptop, with DimOS running)

```bash
# Point ws_bridge at EC2
python ws_bridge.py --cloud-url http://<ec2-public-ip>:8080

# Or locally
python ws_bridge.py --cloud-url http://localhost:8080

# Send a one-shot navigation goal
python ws_bridge.py --send-goal 2.0 1.5

# Start autonomous exploration
python ws_bridge.py --explore
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `USE_MEMORY_STORE` | `true` | `false` to persist map to S3 |
| `S3_BUCKET` | `robohack-map` | S3 bucket name |
| `AWS_REGION` | `eu-west-1` | AWS region (use `eu-north-1` for Stockholm) |
| `DIMOS_WS_URL` | `ws://localhost:7779` | DimOS Socket.IO URL |
| `AWS_ACCESS_KEY_ID` | — | IAM credentials for Bedrock + S3 |
| `AWS_SECRET_ACCESS_KEY` | — | IAM credentials for Bedrock + S3 |

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Live dashboard (map + chat + objects) |
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Robot pushes detected objects |
| `POST` | `/ingest/map` | Bridge pushes costmap / path / pose |
| `GET` | `/map` | Merged semantic map (all robots) |
| `GET` | `/map/live` | Raw live map state (costmap + path + pose) |
| `POST` | `/query/stream` | SSE — natural language query via Bedrock |
| `POST` | `/navigate` | Send navigation goal to robot |
| `POST` | `/frames` | Robot pushes JPEG camera frame |
| `GET` | `/frames/{robot_id}` | Latest camera frame |
| `GET` | `/frames/{robot_id}/stream` | MJPEG stream |

## Bedrock model

Uses `eu.anthropic.claude-sonnet-4-6` (cross-region inference profile, eu-north-1).
Requires IAM permissions: `bedrock:InvokeModelWithResponseStream`.
