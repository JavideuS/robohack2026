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
    store: WorldStateStore = app.state.world_store
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
    store: WorldStateStore = app.state.world_store
    try:
        from agent import run_agent_stream
    except ImportError:
        raise HTTPException(status_code=503, detail="Agent module not available.")

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
_grid_buf: list[int] | None = None
_grid_shape: list[int] = [0, 0]
_grid_version: int = 0


def _decode_grid(grid_data: dict) -> list[int] | None:
    """
    Decode DimOS OptimizedCostmapEncoder → flat uint8 list.
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
                return None
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
    cm = data.get("costmap")
    if cm and isinstance(cm, dict) and "grid" in cm:
        flat = _decode_grid(cm["grid"])
        if flat is not None:
            data = dict(data)
            data["costmap"] = {
                "b64": base64.b64encode(bytes(flat)).decode(),
                "shape": _grid_shape,
                "resolution": cm.get("resolution", 0.05),
                "origin": cm.get("origin"),
                "v": _grid_version,
            }
    _live_map.update({k: v for k, v in data.items() if v is not None})
    _live_map["timestamp"] = time.time()
    return {"status": "ok"}


@app.get("/map/live")
async def get_live_map():
    return _live_map


# ── Camera Frames ─────────────────────────────────────────────

_latest_frames: dict[str, bytes] = {}


@app.post("/frames")
async def receive_frame(request: Request):
    robot_id = request.headers.get("X-Robot-Id", "go2_a")
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty frame")
    _latest_frames[robot_id] = body
    return {"status": "ok", "robot_id": robot_id, "size": len(body)}


@app.get("/frames/{robot_id}")
async def get_frame(robot_id: str):
    frame = _latest_frames.get(robot_id)
    if frame is None:
        raise HTTPException(404, "No frame available")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/frames/{robot_id}/stream")
async def stream_frames(robot_id: str):
    import asyncio
    async def generate():
        while True:
            frame = _latest_frames.get(robot_id)
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── Navigation — goal queue ───────────────────────────────────

_goal_queue: list[dict] = []


@app.post("/navigate")
async def navigate_to_point(x: float, y: float, z: float = 0.0):
    """Queue a navigation goal — ws_bridge polls /goals/pending and forwards to DimOS."""
    _goal_queue.append({"x": x, "y": y, "z": z, "ts": time.time()})
    return {"status": "queued", "target": {"x": x, "y": y, "z": z}}


@app.get("/goals/pending")
async def get_pending_goals():
    goals = list(_goal_queue)
    _goal_queue.clear()
    return {"goals": goals}


# ── Map Endpoint ──────────────────────────────────────────────

@app.get("/map", response_model=MapResponse)
async def get_map():
    store: WorldStateStore = app.state.world_store
    states = store.load_all()
    merged = store.merge()
    return MapResponse(objects=merged, robot_count=len(states), timestamp=time.time())


# ── Dashboard UI ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


# ── Dashboard HTML ────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>SpatialMind — Go2 3D Live</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script type="importmap">
  {
    "imports": {
      "three": "https://cdn.jsdelivr.net/npm/three@0.161/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.161/examples/jsm/"
    }
  }
  </script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', sans-serif; background: #080c10; color: #dde;
           height: 100vh; overflow: hidden; }

    header { display: flex; align-items: center; gap: 10px; padding: 9px 16px;
             background: #0d1117; border-bottom: 1px solid #1e2d3d; height: 44px; }
    header h1 { font-size: 16px; color: #4fc3f7; font-weight: 600; }
    .badge { font-size: 10px; padding: 2px 8px; border-radius: 10px;
             background: #1e3a1e; color: #81c784; border: 1px solid #2e5a2e; }
    #robot-pos { font-size: 11px; color: #4fc3f7; font-family: monospace;
                 opacity: 0.8; margin-left: 6px; }
    #conn-dot { width: 8px; height: 8px; border-radius: 50%; background: #444;
                transition: background 0.3s; margin-left: auto; }
    #conn-dot.live { background: #81c784; box-shadow: 0 0 6px #81c784; }

    .layout { display: grid; grid-template-columns: 1fr 310px;
              height: calc(100vh - 44px); }
    @media (max-width: 768px) {
      .layout { grid-template-columns: 1fr; grid-template-rows: 50vh 1fr; }
    }

    #map-panel { position: relative; background: #080c10; overflow: hidden; }
    #map-panel canvas { display: block; }
    #nav-status { position: absolute; top: 12px; left: 50%; transform: translateX(-50%);
                  font-size: 12px; color: #ffb74d; background: rgba(0,0,0,0.85);
                  padding: 5px 14px; border-radius: 14px; pointer-events: none;
                  opacity: 0; transition: opacity 0.3s; white-space: nowrap; z-index: 10; }
    #nav-status.show { opacity: 1; }
    #view-hint { position: absolute; bottom: 8px; left: 10px; font-size: 10px;
                 color: rgba(255,255,255,0.22); pointer-events: none; z-index: 10; }

    #sidebar { display: flex; flex-direction: column; border-left: 1px solid #1e2d3d;
               overflow: hidden; }
    .section { padding: 9px 12px; border-bottom: 1px solid #1e2d3d; flex-shrink: 0; }
    .section-title { font-size: 10px; font-weight: 700; color: #4fc3f7;
                     letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }

    #chat { flex: 1; overflow-y: auto; padding: 8px 12px; min-height: 0; }
    .msg { margin-bottom: 6px; font-size: 12px; line-height: 1.5; }
    .msg-user b { color: #80cbc4; }
    .msg-robot b { color: #b0bec5; }
    #chat-input-row { display: flex; gap: 6px; padding: 8px 10px;
                      border-top: 1px solid #1e2d3d; flex-shrink: 0; }
    #q { flex: 1; padding: 7px 10px; background: #0d1117; border: 1px solid #2a3a4a;
         border-radius: 6px; color: #eee; font-size: 13px; outline: none; }
    #q:focus { border-color: #4fc3f7; }
    #send-btn { padding: 7px 12px; background: #1565c0; border: none; border-radius: 6px;
                color: #fff; cursor: pointer; font-weight: 600; font-size: 12px; }
    #send-btn:hover { background: #1976d2; }

    #cam-img { width: 100%; display: block; border-radius: 4px; background: #0a0e14;
               min-height: 60px; max-height: 110px; object-fit: cover; }
    #obj-section { overflow-y: auto; flex: 0 0 auto; max-height: 150px; }
    table { width: 100%; border-collapse: collapse; font-size: 11px; }
    td, th { padding: 4px 6px; text-align: left; border-bottom: 1px solid #1a2530; }
    th { color: #4fc3f7; font-weight: 600; background: #0d1117; position: sticky; top: 0; }
    .hi { color: #81c784; } .lo { color: #e57373; }
    .empty-row td { color: #555; text-align: center; padding: 10px; }
  </style>
</head>
<body>

<header>
  <h1>SpatialMind</h1>
  <span class="badge">Go2 · 3D Live</span>
  <span id="robot-pos"></span>
  <div id="conn-dot"></div>
</header>

<div class="layout">
  <div id="map-panel">
    <div id="nav-status"></div>
    <div id="view-hint">drag to orbit · scroll to zoom · click floor to navigate</div>
  </div>
  <div id="sidebar">
    <div class="section"><div class="section-title">Ask the robot</div></div>
    <div id="chat"></div>
    <div id="chat-input-row">
      <input id="q" placeholder="Where is the chair?"
             onkeydown="if(event.key==='Enter')window._chat()">
      <button id="send-btn" onclick="window._chat()">Ask</button>
    </div>
    <div class="section">
      <div class="section-title">Camera</div>
      <img id="cam-img" src="/frames/go2_a/stream"
           onerror="this.style.opacity='0.12'" alt="">
    </div>
    <div class="section" id="obj-section">
      <div class="section-title">Objects</div>
      <table>
        <thead><tr><th>Label</th><th>Pos (m)</th><th>Conf</th></tr></thead>
        <tbody id="obj-body">
          <tr class="empty-row"><td colspan="3">Waiting for robot...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
/* Chat — must be global for onclick */
window._chat = async function() {
  const inp = document.getElementById('q');
  const q = inp.value.trim();
  if (!q) return;
  _msg('You', q, 'msg-user');
  inp.value = '';
  const el = _msg('Go2', '', 'msg-robot');
  try {
    const resp = await fetch('/query/stream', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text: q})
    });
    const reader = resp.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      for (const line of dec.decode(value).split('\\n')) {
        if (!line.startsWith('data: ')) continue;
        const tok = line.slice(6);
        if (tok === '[DONE]') break;
        try { el.querySelector('span').innerHTML += JSON.parse(tok); } catch(e) {}
        document.getElementById('chat').scrollTop = 99999;
      }
    }
  } catch(e) { el.querySelector('span').textContent += ' [error]'; }
};
function _msg(who, text, cls) {
  const chat = document.getElementById('chat');
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.innerHTML = '<b>' + who + ':</b> <span>' + text + '</span>';
  chat.appendChild(d); chat.scrollTop = 99999; return d;
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Renderer ──────────────────────────────────────────────────
const panel = document.getElementById('map-panel');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(panel.clientWidth, panel.clientHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
panel.appendChild(renderer.domElement);

// ── Scene ─────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080c10);
scene.fog = new THREE.FogExp2(0x080c10, 0.03);

// ── Camera ────────────────────────────────────────────────────
const camera = new THREE.PerspectiveCamera(55, panel.clientWidth / panel.clientHeight, 0.01, 400);
camera.position.set(0, 10, 14);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.maxPolarAngle = Math.PI * 0.47;
controls.minDistance = 0.3;
controls.maxDistance = 150;

let rotTimer;
renderer.domElement.addEventListener('pointerdown', () => {
  controls.autoRotate = false;
  clearTimeout(rotTimer);
  rotTimer = setTimeout(() => { controls.autoRotate = true; }, 7000);
});

// ── Lights ────────────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0x223355, 0x080c10, 2.5));
const sun = new THREE.DirectionalLight(0x99aacc, 3.5);
sun.position.set(20, 40, 25);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x334466, 0.5);
fill.position.set(-15, -5, -15);
scene.add(fill);

// ── Floor & grid ──────────────────────────────────────────────
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(400, 400),
  new THREE.MeshStandardMaterial({ color: 0x0a1018, roughness: 1 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.002;
scene.add(floor);
scene.add(new THREE.GridHelper(400, 400, 0x111d2a, 0x0e1824));

// ── Voxel InstancedMesh ───────────────────────────────────────
const MAX_V = 80000;
const WALL_H = 0.35;
const GRAD_H = 0.12;

const unitBox = new THREE.BoxGeometry(1, 1, 1);

const wallMesh = new THREE.InstancedMesh(
  unitBox,
  new THREE.MeshStandardMaterial({ color: 0x2a3a52, roughness: 0.55, metalness: 0.45 }),
  MAX_V
);
wallMesh.count = 0;
scene.add(wallMesh);

const gradMesh = new THREE.InstancedMesh(
  unitBox,
  new THREE.MeshStandardMaterial({ color: 0xc05318, roughness: 0.8, transparent: true, opacity: 0.5 }),
  MAX_V
);
gradMesh.count = 0;
scene.add(gradMesh);

const dummy = new THREE.Object3D();
let lastV = -1, cameraFitted = false;

function b64ToUint8(b64) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}

function rebuildVoxels(cm) {
  if (!cm || (!cm.flat && !cm.b64) || cm.v === lastV) return;
  lastV = cm.v;
  const flat = cm.b64 ? b64ToUint8(cm.b64) : cm.flat;
  const shape = cm.shape || [0, 0];
  const rows = shape[0], cols = shape[1];
  if (!rows || !cols) return;
  const res = cm.resolution || 0.05;
  const oc = cm.origin || {};
  const ox = oc.c ? oc.c[0] : 0;
  const oy = oc.c ? oc.c[1] : 0;

  let wc = 0, gc = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = flat[r * cols + c];
      const wx = ox + (c + 0.5) * res;
      const wy = oy + (r + 0.5) * res;
      if (v >= 90 && v !== 255) {
        if (wc < MAX_V) {
          dummy.position.set(wx, WALL_H * 0.5, -wy);
          dummy.scale.set(res, WALL_H, res);
          dummy.updateMatrix();
          wallMesh.setMatrixAt(wc++, dummy.matrix);
        }
      } else if (v > 5 && v < 90) {
        if (gc < MAX_V) {
          const h = GRAD_H * (v / 89);
          dummy.position.set(wx, h * 0.5, -wy);
          dummy.scale.set(res, h, res);
          dummy.updateMatrix();
          gradMesh.setMatrixAt(gc++, dummy.matrix);
        }
      }
    }
  }
  wallMesh.count = wc; gradMesh.count = gc;
  wallMesh.instanceMatrix.needsUpdate = true;
  gradMesh.instanceMatrix.needsUpdate = true;

  if (!cameraFitted) {
    cameraFitted = true;
    const mx = ox + cols * res / 2;
    const my = oy + rows * res / 2;
    const span = Math.max(cols, rows) * res;
    controls.target.set(mx, 0, -my);
    camera.position.set(mx, span * 0.65, -my + span * 0.8);
    controls.update();
  }
}

// ── Robot ─────────────────────────────────────────────────────
const robotGrp = new THREE.Group();

const body = new THREE.Mesh(
  new THREE.SphereGeometry(0.18, 16, 12),
  new THREE.MeshStandardMaterial({
    color: 0x4fc3f7, emissive: 0x4fc3f7, emissiveIntensity: 0.5, roughness: 0.2
  })
);
robotGrp.add(body);

const arrowMesh = new THREE.Mesh(
  new THREE.ConeGeometry(0.07, 0.28, 8),
  new THREE.MeshStandardMaterial({ color: 0x80d8ff, emissive: 0x4fc3f7, emissiveIntensity: 0.3 })
);
arrowMesh.rotation.z = -Math.PI / 2;
arrowMesh.position.x = 0.27;
robotGrp.add(arrowMesh);

const ringMesh = new THREE.Mesh(
  new THREE.RingGeometry(0.22, 0.32, 32),
  new THREE.MeshBasicMaterial({ color: 0x4fc3f7, transparent: true, opacity: 0.28, side: THREE.DoubleSide })
);
ringMesh.rotation.x = -Math.PI / 2;
robotGrp.add(ringMesh);

robotGrp.visible = false;
scene.add(robotGrp);

function updateRobot(rp) {
  if (!rp || !rp.c) return;
  const x = rp.c[0], y = rp.c[1], theta = rp.c[2] || 0;
  robotGrp.position.set(x, 0.18, -y);
  robotGrp.rotation.y = -theta;
  robotGrp.visible = true;
  document.getElementById('robot-pos').textContent = '(' + x.toFixed(2) + ', ' + y.toFixed(2) + ')';
}

// ── Path ──────────────────────────────────────────────────────
const pathGeo = new THREE.BufferGeometry();
const pathLine = new THREE.Line(
  pathGeo,
  new THREE.LineBasicMaterial({ color: 0xce93d8, transparent: true, opacity: 0.85 })
);
scene.add(pathLine);

function updatePath(pd) {
  if (!pd || !pd.points || !pd.points.length) { pathGeo.setFromPoints([]); return; }
  pathGeo.setFromPoints(pd.points.map(function(p) {
    return new THREE.Vector3(p[0], 0.05, -p[1]);
  }));
}

// ── Semantic objects ──────────────────────────────────────────
const objPool = [];
const oGeo = new THREE.SphereGeometry(0.12, 8, 6);
const C_HI = new THREE.Color(0x66bb6a);
const C_LO = new THREE.Color(0xef9a9a);

for (let i = 0; i < 80; i++) {
  const m = new THREE.Mesh(
    oGeo,
    new THREE.MeshStandardMaterial({ roughness: 0.4 })
  );
  m.visible = false;
  scene.add(m);
  objPool.push(m);
}

function updateObjects(objs) {
  const disp = objs.filter(function(o) {
    return o.pose && o.label !== 'robot_position' && o.label.indexOf('nav_goal') !== 0;
  });
  for (let i = 0; i < objPool.length; i++) {
    if (i < disp.length) {
      const o = disp[i];
      objPool[i].position.set(o.pose.x, 0.15, -o.pose.y);
      const col = (o.confidence || 0) > 0.7 ? C_HI : C_LO;
      objPool[i].material.color.copy(col);
      objPool[i].material.emissive.copy(col).multiplyScalar(0.15);
      objPool[i].visible = true;
    } else {
      objPool[i].visible = false;
    }
  }
  const tbody = document.getElementById('obj-body');
  if (!disp.length) {
    tbody.innerHTML = '<tr class="empty-row"><td colspan="3">No objects</td></tr>';
    return;
  }
  tbody.innerHTML = disp.map(function(o) {
    const c = ((o.confidence || 0) * 100).toFixed(0);
    const cls = c > 70 ? 'hi' : 'lo';
    return '<tr><td>' + o.label + '</td>'
         + '<td>(' + (o.pose.x || 0).toFixed(1) + ', ' + (o.pose.y || 0).toFixed(1) + ')</td>'
         + '<td class="' + cls + '">' + c + '%</td></tr>';
  }).join('');
}

// ── Click → navigate ──────────────────────────────────────────
const ray = new THREE.Raycaster();
const m2 = new THREE.Vector2();
const navPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const navPt = new THREE.Vector3();

renderer.domElement.addEventListener('click', async function(e) {
  const r = renderer.domElement.getBoundingClientRect();
  m2.x =  ((e.clientX - r.left) / r.width)  * 2 - 1;
  m2.y = -((e.clientY - r.top)  / r.height) * 2 + 1;
  ray.setFromCamera(m2, camera);
  if (!ray.ray.intersectPlane(navPlane, navPt)) return;
  const wx = navPt.x, wy = -navPt.z;
  _showNav('Sending (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')...');
  try {
    const res = await fetch('/navigate?x=' + wx.toFixed(3) + '&y=' + wy.toFixed(3), {method: 'POST'});
    const d = await res.json();
    _showNav(res.ok
      ? 'Goal sent (' + wx.toFixed(2) + ', ' + wy.toFixed(2) + ')'
      : 'Failed: ' + (d.detail || 'error'), !res.ok);
  } catch(err) { _showNav('Error: ' + err.message, true); }
});

function _showNav(msg, err) {
  const el = document.getElementById('nav-status');
  el.textContent = msg;
  el.style.color = err ? '#ef9a9a' : '#ffb74d';
  el.classList.add('show');
  clearTimeout(el._t);
  el._t = setTimeout(function() { el.classList.remove('show'); }, 3000);
}

// ── Polling ───────────────────────────────────────────────────
let lastTs = 0;

async function pollLive() {
  try {
    const d = await (await fetch('/map/live')).json();
    if (d.costmap) rebuildVoxels(d.costmap);
    if (d.robot_pose) updateRobot(d.robot_pose);
    if (d.path) updatePath(d.path);
    if (d.timestamp && d.timestamp !== lastTs) {
      lastTs = d.timestamp;
      document.getElementById('conn-dot').classList.add('live');
    }
  } catch(e) {}
}

async function pollSem() {
  try {
    const d = await (await fetch('/map')).json();
    updateObjects(d.objects || []);
  } catch(e) {}
}

setInterval(pollLive, 500);
setInterval(pollSem, 2000);
pollLive(); pollSem();

// ── Resize ────────────────────────────────────────────────────
window.addEventListener('resize', function() {
  camera.aspect = panel.clientWidth / panel.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(panel.clientWidth, panel.clientHeight);
});

// ── Render loop ───────────────────────────────────────────────
(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();
</script>

</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
