"""
AWS Cloud Middleware — FastAPI Communication Layer

Endpoints:
  POST /ingest        — Robot pushes detections (Req 1)
  POST /query/stream  — User asks natural language question, SSE response (Req 2)
  GET  /map           — Merged world state from all robots
  GET  /              — Dashboard UI
  GET  /health        — Health check (no auth required)
"""

import base64
import os
import time
import json
import logging
import zlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, Response

from models import (
    IngestRequest, IngestResponse,
    QueryRequest, MapResponse,
    ErrorResponse, WorldState,
)
from world_store import WorldStateStore

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────

S3_BUCKET = os.environ.get("S3_BUCKET", "robohack-map")
AWS_REGION = os.environ.get("AWS_REGION", "eu-west-1")
USE_MEMORY_STORE = os.environ.get("USE_MEMORY_STORE", "true").lower() == "true"


# ── App Lifespan ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    # Use in-memory store by default (no AWS keys needed)
    # Set USE_MEMORY_STORE=false when AWS credentials are configured
    app.state.world_store = WorldStateStore(
        bucket=S3_BUCKET,
        use_memory=USE_MEMORY_STORE,
        region=AWS_REGION,
    )
    logger.info(
        f"World store initialized: {'memory' if USE_MEMORY_STORE else 'S3'} "
        f"(bucket={S3_BUCKET})"
    )
    yield


app = FastAPI(
    title="AWS Cloud Middleware — DimOS",
    description="Communication and inference hub for Unitree Go2 robot",
    lifespan=lifespan,
)


# ── Health Check ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "store": "memory" if USE_MEMORY_STORE else "s3"}


# ── Robot Endpoints (Req 1: Ingestion) ────────────────────────

@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Robot pushes detection data to the cloud.
    Validates payload, persists to world state store.
    """
    store: WorldStateStore = app.state.world_store

    # Build WorldState with current timestamp
    state = WorldState(
        robot_id=request.robot_id,
        timestamp=time.time(),
        objects=request.objects,
    )

    try:
        store.save(state)
    except Exception as e:
        logger.error(f"Failed to save world state for {request.robot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Storage error: {e}")

    return IngestResponse(
        status="saved",
        robot_id=request.robot_id,
        count=len(request.objects),
        timestamp=state.timestamp,
    )


# ── User Endpoints (Req 2: Query) ────────────────────────────

@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    User asks a natural language question.
    Streams response via SSE using Bedrock agent loop.
    """
    store: WorldStateStore = app.state.world_store

    try:
        from agent import run_agent_stream
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Agent module not available. Bedrock not configured."
        )

    def generate():
        try:
            for token in run_agent_stream(request.text, store):
                yield f"data: {json.dumps(token)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Agent stream error: {e}")
            yield f"data: {json.dumps(f'Error: {e}')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Live Map State (costmap + path + robot_pose) ─────────────

_live_map: dict = {}
_grid_buf: list[int] | None = None   # reconstructed flat grid (uint8)
_grid_shape: list[int] = [0, 0]      # [height, width]
_grid_version: int = 0               # incremented on each grid change


def _decode_grid(grid_data: dict) -> list[int] | None:
    """
    Decode DimOS OptimizedCostmapEncoder format → plain flat uint8 list.
    Handles both 'full' (full grid) and 'delta' (chunk patches) update types.
    Values: 0=free, 1-89=gradient, 90-100=occupied, 255=unknown.
    """
    global _grid_buf, _grid_shape, _grid_version

    shape = grid_data.get("shape", [0, 0])
    h, w = shape
    update_type = grid_data.get("update_type", "full")

    try:
        if update_type == "full":
            raw = zlib.decompress(base64.b64decode(grid_data["data"]))
            _grid_buf = list(raw)
            _grid_shape = shape
            _grid_version += 1
            return _grid_buf

        elif update_type == "delta":
            if _grid_buf is None or _grid_shape != shape:
                return None  # need a full frame first
            grid = list(_grid_buf)
            for chunk in grid_data.get("chunks", []):
                cy, cx = chunk["pos"]
                ch, cw = chunk["size"]
                raw = zlib.decompress(base64.b64decode(chunk["data"]))
                for y in range(ch):
                    for x in range(cw):
                        idx = (cy + y) * w + (cx + x)
                        if idx < len(grid):
                            grid[idx] = raw[y * cw + x]
            _grid_buf = grid
            _grid_shape = shape
            _grid_version += 1
            return _grid_buf

    except Exception as e:
        logger.debug(f"Costmap decode error: {e}")

    return None


@app.post("/ingest/map")
async def ingest_map(request: Request):
    """Bridge pushes raw map state (costmap, path, robot_pose)."""
    data = await request.json()

    # Decode compressed costmap grid to plain flat array
    cm = data.get("costmap")
    if cm and isinstance(cm, dict) and "grid" in cm:
        flat = _decode_grid(cm["grid"])
        if flat is not None:
            data = dict(data)
            data["costmap"] = {
                "flat": flat,
                "shape": _grid_shape,
                "resolution": cm.get("resolution", 0.05),
                "origin": cm.get("origin"),   # {"type":"vector","c":[x,y,0]}
                "v": _grid_version,
            }

    _live_map.update({k: v for k, v in data.items() if v is not None})
    _live_map["timestamp"] = time.time()
    return {"status": "ok"}


@app.get("/map/live")
async def get_live_map():
    """Return latest costmap + path + robot_pose for dashboard rendering."""
    return _live_map


# ── Camera Frames (Req: live video feed) ─────────────────────

# Store latest frame per robot (in-memory, overwritten each push)
_latest_frames: dict[str, bytes] = {}


@app.post("/frames")
async def receive_frame(request: Request):
    """Robot pushes a JPEG camera frame."""
    robot_id = request.headers.get("X-Robot-Id", "go2_a")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty frame")
    _latest_frames[robot_id] = body
    return {"status": "ok", "robot_id": robot_id, "size": len(body)}


@app.get("/frames/{robot_id}")
async def get_frame(robot_id: str):
    """Get the latest camera frame for a robot."""
    frame = _latest_frames.get(robot_id)
    if frame is None:
        raise HTTPException(404, "No frame available for this robot")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/frames/{robot_id}/stream")
async def stream_frames(robot_id: str):
    """MJPEG stream of the latest frames for a robot."""
    import asyncio

    async def generate():
        while True:
            frame = _latest_frames.get(robot_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            await asyncio.sleep(0.1)  # ~10 fps max

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Navigation — goal queue (robot polls and forwards to DimOS) ─

_goal_queue: list[dict] = []


@app.post("/navigate")
async def navigate_to_point(x: float, y: float, z: float = 0.0):
    """Queue a navigation goal — ws_bridge polls /goals/pending and forwards to DimOS."""
    _goal_queue.append({"x": x, "y": y, "z": z, "ts": time.time()})
    return {"status": "queued", "target": {"x": x, "y": y, "z": z}}


@app.get("/goals/pending")
async def get_pending_goals():
    """ws_bridge calls this to drain and forward queued goals to DimOS."""
    goals = list(_goal_queue)
    _goal_queue.clear()
    return {"goals": goals}


# ── Map Endpoint ──────────────────────────────────────────────

@app.get("/map", response_model=MapResponse)
async def get_map():
    """Return merged world state from all robots."""
    store: WorldStateStore = app.state.world_store
    states = store.load_all()
    merged = store.merge()

    return MapResponse(
        objects=merged,
        robot_count=len(states),
        timestamp=time.time(),
    )


# ── Dashboard UI ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard UI."""
    return DASHBOARD_HTML


# ── Run ───────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SpatialMind — Go2 Live Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #080c10; color: #dde; min-height: 100vh; }

    header { display: flex; align-items: center; gap: 12px; padding: 12px 16px;
             background: #0d1117; border-bottom: 1px solid #1e2d3d; }
    header h1 { font-size: 18px; color: #4fc3f7; font-weight: 600; }
    .badge { font-size: 11px; padding: 2px 8px; border-radius: 10px;
             background: #1e3a1e; color: #81c784; border: 1px solid #2e5a2e; }
    #conn-dot { width: 8px; height: 8px; border-radius: 50%; background: #555;
                transition: background 0.3s; margin-left: auto; }
    #conn-dot.live { background: #81c784; box-shadow: 0 0 6px #81c784; }

    .layout { display: grid; grid-template-columns: 1fr 340px; gap: 0; height: calc(100vh - 49px); }
    @media (max-width: 800px) {
      .layout { grid-template-columns: 1fr; grid-template-rows: 55vw 1fr; }
    }

    /* ── Map panel ── */
    #map-panel { position: relative; background: #0a0e12; overflow: hidden; }
    #map-canvas { display: block; width: 100%; height: 100%; cursor: crosshair; }
    #map-overlay { position: absolute; bottom: 10px; left: 10px; font-size: 12px; color: #4fc3f7;
                   background: rgba(0,0,0,0.6); padding: 4px 8px; border-radius: 4px; }
    #nav-status { position: absolute; top: 10px; left: 50%; transform: translateX(-50%);
                  font-size: 13px; color: #ffb74d; background: rgba(0,0,0,0.75);
                  padding: 4px 12px; border-radius: 12px; pointer-events: none;
                  opacity: 0; transition: opacity 0.3s; white-space: nowrap; }
    #nav-status.show { opacity: 1; }

    /* ── Sidebar ── */
    #sidebar { display: flex; flex-direction: column; border-left: 1px solid #1e2d3d;
               overflow: hidden; }

    .section { padding: 12px 14px; border-bottom: 1px solid #1e2d3d; }
    .section-title { font-size: 11px; font-weight: 700; color: #4fc3f7; letter-spacing: 0.1em;
                     text-transform: uppercase; margin-bottom: 8px; }

    /* Chat */
    #chat { flex: 1; overflow-y: auto; padding: 10px 14px; }
    .msg { margin-bottom: 8px; font-size: 13px; line-height: 1.5; }
    .msg-user b { color: #80cbc4; }
    .msg-robot b { color: #b0bec5; }
    #chat-input-row { display: flex; gap: 6px; padding: 10px 12px;
                      border-top: 1px solid #1e2d3d; }
    #q { flex: 1; padding: 8px 10px; background: #0d1117; border: 1px solid #2a3a4a;
         border-radius: 6px; color: #eee; font-size: 14px; outline: none; }
    #q:focus { border-color: #4fc3f7; }
    #send-btn { padding: 8px 14px; background: #1565c0; border: none; border-radius: 6px;
                color: #fff; cursor: pointer; font-weight: 600; font-size: 13px; }
    #send-btn:hover { background: #1976d2; }

    /* Objects table */
    #obj-section { overflow-y: auto; max-height: 220px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    td, th { padding: 5px 8px; text-align: left; border-bottom: 1px solid #1a2530; }
    th { color: #4fc3f7; font-weight: 600; background: #0d1117; position: sticky; top: 0; }
    .hi { color: #81c784; } .lo { color: #e57373; }
    .empty-row td { color: #555; text-align: center; padding: 14px; }

    /* Legend */
    .legend { display: flex; gap: 12px; flex-wrap: wrap; font-size: 11px; color: #888; }
    .legend span { display: flex; align-items: center; gap: 4px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  </style>
</head>
<body>
<header>
  <h1>SpatialMind</h1>
  <span class="badge">Go2 Live</span>
  <div class="legend" style="margin-left:12px">
    <span><span class="dot" style="background:#4fc3f7"></span>Robot</span>
    <span><span class="dot" style="background:#66bb6a"></span>Object</span>
    <span><span class="dot" style="background:#ff7043"></span>Goal</span>
    <span><span class="dot" style="background:#ce93d8"></span>Path</span>
  </div>
  <div id="conn-dot" title="Live data connection"></div>
</header>

<div class="layout">
  <!-- Map -->
  <div id="map-panel">
    <canvas id="map-canvas"></canvas>
    <div id="map-overlay">click to navigate</div>
    <div id="nav-status"></div>
  </div>

  <!-- Sidebar -->
  <div id="sidebar">
    <div class="section">
      <div class="section-title">Camera Feed</div>
      <img id="cam-feed" src="/frames/go2_a/stream" alt="Camera"
           style="width:100%; border-radius:6px; background:#111; min-height:120px; object-fit:cover;"
           onerror="this.style.display='none'; document.getElementById('cam-placeholder').style.display='block'">
      <div id="cam-placeholder" style="display:none; text-align:center; color:#555; padding:20px; font-size:12px;">
        No camera feed yet
      </div>
    </div>
    <div class="section">
      <div class="section-title">Ask the robot</div>
    </div>
    <div id="chat"></div>
    <div id="chat-input-row">
      <input id="q" placeholder="Where is the chair?" onkeydown="if(event.key==='Enter')send()">
      <button id="send-btn" onclick="send()">Ask</button>
    </div>
    <div class="section">
      <div class="section-title">Camera</div>
      <img id="cam-img" src="/frames/go2_a/stream"
           style="width:100%;border-radius:6px;background:#0d1117;min-height:80px;display:block"
           onerror="this.style.opacity='0.3'" alt="No feed">
    </div>
    <div class="section" id="obj-section">
      <div class="section-title">Detected objects</div>
      <table>
        <thead><tr><th>Label</th><th>Pos (m)</th><th>Conf</th></tr></thead>
        <tbody id="obj-body"><tr class="empty-row"><td colspan="3">Waiting...</td></tr></tbody>
      </table>
    </div>
  </div>
</div>

<script>
// ── Canvas setup ──────────────────────────────────────────────
const canvas = document.getElementById('map-canvas');
const ctx = canvas.getContext('2d');

function resizeCanvas() {
  const panel = document.getElementById('map-panel');
  canvas.width = panel.clientWidth;
  canvas.height = panel.clientHeight;
}
resizeCanvas();
window.addEventListener('resize', () => { resizeCanvas(); render(); });

// ── State ─────────────────────────────────────────────────────
let liveMap = {};       // costmap, path, robot_pose from /map/live
let semObjects = [];    // from /map
let clickGoal = null;
let lastLiveTs = 0;

// World coordinate range shown on canvas (meters, centered at origin)
// Auto-expands to fit costmap
let MAP_RANGE = 5;

function worldToCanvas(wx, wy) {
  const cx = (wx + MAP_RANGE) / (2 * MAP_RANGE) * canvas.width;
  const cy = (MAP_RANGE - wy) / (2 * MAP_RANGE) * canvas.height;
  return [cx, cy];
}
function canvasToWorld(cx, cy) {
  const wx = (cx / canvas.width)  * (2 * MAP_RANGE) - MAP_RANGE;
  const wy = MAP_RANGE - (cy / canvas.height) * (2 * MAP_RANGE);
  return [wx, wy];
}

// ── Costmap renderer ──────────────────────────────────────────
// Server decodes zlib+base64 → plain flat uint8 list.
// Values: 0=free, 1-89=gradient, 90-100=occupied, 255=unknown.
let costmapCache = null;   // {offscreen canvas, w, h, res, origX, origY}

function buildCostmapImage(cm) {
  const flat = cm.flat;
  if (!flat || !flat.length) return;

  const [h, w] = cm.shape || [Math.round(Math.sqrt(flat.length)), Math.round(Math.sqrt(flat.length))];
  const res = cm.resolution || 0.05;

  // origin.c[0], origin.c[1] — DimOS format {"type":"vector","c":[x,y,0]}
  const orig = cm.origin || {};
  const origX = orig.c ? orig.c[0] : (orig.x || -(w * res / 2));
  const origY = orig.c ? orig.c[1] : (orig.y || -(h * res / 2));

  // Auto-expand map range to fit costmap
  const span = Math.max(w, h) * res / 2 + Math.max(Math.abs(origX), Math.abs(origY)) + 1;
  if (span > MAP_RANGE) MAP_RANGE = Math.ceil(span);

  const img = new ImageData(w, h);
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    let r, g, b;
    if (v === 255)    { r = 95;  g = 105; b = 115; }   // unknown  → blue-gray
    else if (v === 0) { r = 215; g = 218; b = 222; }   // free     → light
    else if (v >= 90) { r = 15;  g = 18;  b = 22;  }   // occupied → dark
    else              { r = 215 - v*2; g = 218 - v*2; b = 222 - v*2; }
    img.data[i*4]   = r;
    img.data[i*4+1] = g;
    img.data[i*4+2] = b;
    img.data[i*4+3] = 255;
  }

  // Draw to offscreen canvas once — drawCostmap() just blits it scaled
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  off.getContext('2d').putImageData(img, 0, 0);

  costmapCache = {off, w, h, res, origX, origY};
}

function drawCostmap() {
  if (!costmapCache) return;
  const {off, w, h, res, origX, origY} = costmapCache;

  // DimOS costmap row-major: row 0 = bottom of world (y = origY)
  // Canvas Y is flipped vs world Y, so:
  //   world top-left  = (origX, origY + h*res)
  //   world bot-right = (origX + w*res, origY)
  const [x0, y0] = worldToCanvas(origX,         origY + h * res);
  const [x1, y1] = worldToCanvas(origX + w*res, origY);
  const dw = x1 - x0, dh = y1 - y0;
  if (dw <= 0 || dh <= 0) return;

  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, x0, y0, dw, dh);
  ctx.restore();
}

// ── Main render ───────────────────────────────────────────────
function render() {
  ctx.fillStyle = '#0a0e12';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Occupancy grid
  drawCostmap();

  // Subtle grid overlay
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  const step = canvas.width / (MAP_RANGE * 2);
  const [ox, oy] = worldToCanvas(0, 0);
  for (let x = ox % step; x < canvas.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = oy % step; y < canvas.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }

  // Origin axes
  ctx.strokeStyle = 'rgba(79,195,247,0.25)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(ox-12, oy); ctx.lineTo(ox+12, oy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(ox, oy-12); ctx.lineTo(ox, oy+12); ctx.stroke();

  // Navigation path
  const path = liveMap.path;
  if (path && Array.isArray(path.points) && path.points.length > 1) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(206,147,216,0.7)';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    const [px0, py0] = worldToCanvas(path.points[0][0], path.points[0][1]);
    ctx.moveTo(px0, py0);
    for (let i = 1; i < path.points.length; i++) {
      const [px, py] = worldToCanvas(path.points[i][0], path.points[i][1]);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Semantic objects
  for (const obj of semObjects) {
    if (!obj.pose) continue;
    if (obj.label === 'robot_position' || obj.label.startsWith('nav_goal')) continue;
    const [cx, cy] = worldToCanvas(obj.pose.x, obj.pose.y);
    const conf = obj.confidence || 0.5;

    ctx.fillStyle = conf > 0.7 ? '#66bb6a' : '#ef9a9a';
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI*2); ctx.fill();

    ctx.fillStyle = '#ccc';
    ctx.font = '11px sans-serif';
    ctx.fillText(obj.label, cx + 7, cy + 4);
  }

  // Click goal ring
  if (clickGoal) {
    const [gx, gy] = worldToCanvas(clickGoal.x, clickGoal.y);
    ctx.strokeStyle = '#ff7043';
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(gx, gy, 11, 0, Math.PI*2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(gx-6, gy); ctx.lineTo(gx+6, gy);
    ctx.moveTo(gx, gy-6); ctx.lineTo(gx, gy+6);
    ctx.stroke();
  }

  // Robot — from live map (robot_pose) or from semantic objects
  let robotPos = null;
  const rp = liveMap.robot_pose;
  if (rp && Array.isArray(rp.c) && rp.c.length >= 2) {
    robotPos = {x: rp.c[0], y: rp.c[1], heading: rp.c[2] || 0};
  } else {
    const ro = semObjects.find(o => o.label === 'robot_position');
    if (ro && ro.pose) robotPos = {x: ro.pose.x, y: ro.pose.y, heading: 0};
  }

  if (robotPos) {
    const [rx, ry] = worldToCanvas(robotPos.x, robotPos.y);
    const angle = -robotPos.heading;  // canvas Y is flipped
    const sz = 10;

    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(angle);

    // Body circle
    ctx.fillStyle = '#4fc3f7';
    ctx.beginPath(); ctx.arc(0, 0, sz, 0, Math.PI*2); ctx.fill();

    // Heading arrow
    ctx.fillStyle = '#0d47a1';
    ctx.beginPath();
    ctx.moveTo(0, -sz - 5);
    ctx.lineTo(-4, -sz + 2);
    ctx.lineTo(4, -sz + 2);
    ctx.closePath();
    ctx.fill();

    ctx.restore();

    // Pos label
    ctx.fillStyle = 'rgba(79,195,247,0.7)';
    ctx.font = '10px monospace';
    ctx.fillText('(' + robotPos.x.toFixed(1) + ', ' + robotPos.y.toFixed(1) + ')', rx + 13, ry + 4);
  }
}

// ── Click to navigate ─────────────────────────────────────────
canvas.addEventListener('click', async (e) => {
  const rect = canvas.getBoundingClientRect();
  const [wx, wy] = canvasToWorld(
    (e.clientX - rect.left) * (canvas.width / rect.width),
    (e.clientY - rect.top)  * (canvas.height / rect.height)
  );
  clickGoal = {x: wx, y: wy};
  showNavStatus('Sending goal (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')...');
  render();

  try {
    const r = await fetch('/navigate?x=' + wx.toFixed(3) + '&y=' + wy.toFixed(3), {method:'POST'});
    const d = await r.json();
    showNavStatus(r.ok ? 'Goal sent → (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')' : 'Failed: ' + d.detail, !r.ok);
  } catch(e) {
    showNavStatus('Error: ' + e.message, true);
  }
});

function showNavStatus(msg, error=false) {
  const el = document.getElementById('nav-status');
  el.textContent = msg;
  el.style.color = error ? '#ef9a9a' : '#ffb74d';
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3000);
}

// ── Data polling ──────────────────────────────────────────────
let lastCostmapV = -1;

async function pollLiveMap() {
  try {
    const r = await fetch('/map/live');
    const d = await r.json();

    // Rebuild costmap image only when server reports a new grid version
    if (d.costmap && d.costmap.flat && d.costmap.v !== lastCostmapV) {
      lastCostmapV = d.costmap.v;
      buildCostmapImage(d.costmap);
    }

    liveMap = d;
    if (d.timestamp && d.timestamp !== lastLiveTs) {
      lastLiveTs = d.timestamp;
      document.getElementById('conn-dot').classList.add('live');
    }
  } catch(e) {}
}

async function pollSemMap() {
  try {
    const r = await fetch('/map');
    const d = await r.json();
    semObjects = d.objects || [];

    const tbody = document.getElementById('obj-body');
    const disp = semObjects.filter(o => o.label !== 'robot_position' && !o.label.startsWith('nav_goal'));
    if (!disp.length) {
      tbody.innerHTML = '<tr class="empty-row"><td colspan="3">No objects yet</td></tr>';
    } else {
      tbody.innerHTML = disp.map(o => {
        const conf = ((o.confidence||0)*100).toFixed(0);
        const cls = conf > 70 ? 'hi' : 'lo';
        return '<tr><td>' + o.label + '</td><td>(' + (o.pose?.x||0).toFixed(1) + ',' + (o.pose?.y||0).toFixed(1) + ')</td>' +
               '<td class="' + cls + '">' + conf + '%</td></tr>';
      }).join('');
    }
  } catch(e) {}
}

// Render loop — 20fps
function loop() { render(); requestAnimationFrame(loop); }
loop();

// Data polls
setInterval(pollLiveMap, 500);   // costmap/path/pose at 2Hz
setInterval(pollSemMap,  2000);  // semantic objects at 0.5Hz
pollLiveMap(); pollSemMap();

// ── Chat ──────────────────────────────────────────────────────
let robotMsgEl = null;

async function send() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  addMsg('You', q, 'msg-user');
  document.getElementById('q').value = '';
  robotMsgEl = addMsg('Go2', '', 'msg-robot');

  try {
    const resp = await fetch('/query/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: q})
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\\n')) {
        if (!line.startsWith('data: ')) continue;
        const tok = line.slice(6);
        if (tok === '[DONE]') break;
        try { robotMsgEl.querySelector('span').innerHTML += JSON.parse(tok); } catch(e) {}
        document.getElementById('chat').scrollTop = 99999;
      }
    }
  } catch(e) { robotMsgEl.querySelector('span').innerHTML += ' [error]'; }
}

function addMsg(who, text, cls) {
  const chat = document.getElementById('chat');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = '<b>' + who + ':</b> <span>' + text + '</span>';
  chat.appendChild(d);
  chat.scrollTop = 99999;
  return d;
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
