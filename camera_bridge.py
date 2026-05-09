"""
Camera Bridge — grabs frames from the robot and pushes to FastAPI /frames.

Source priority (auto mode, first that works wins):
  1. DimOS LCM   — color_image UDP multicast (Linux, confirmed working)
               or pSHM fallback (Mac, see unitree_go2_basic._mac_transports)
  2. RTSP        — rtsp://<host>:8554/video (real Go2 over network)
  3. OpenCV dev  — /dev/video<N> (USB / v4l2 camera)

ROS2 stub is commented at the bottom for future integration.

Usage:
    python camera_bridge.py                          # auto-detect
    python camera_bridge.py --source dimos           # DimOS pSHM (dimos venv)
    python camera_bridge.py --source rtsp            # real Go2
    python camera_bridge.py --source opencv          # USB webcam (testing)
    python camera_bridge.py --cloud-url http://<ec2>:8080
"""

import argparse
import logging
import os
import time

import cv2
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_FPS      = 8
DEFAULT_CLOUD    = os.environ.get("CLOUD_URL", "http://localhost:8080")
DEFAULT_ROBOT_ID = os.environ.get("ROBOT_ID",  "go2_a")
DEFAULT_RTSP     = "rtsp://192.168.123.161:8554/video"
JPEG_QUALITY     = 72


# ── Shared helpers ────────────────────────────────────────────

def push_frame(cloud_url: str, robot_id: str, jpeg_bytes: bytes) -> bool:
    try:
        r = httpx.post(
            f"{cloud_url}/frames",
            content=jpeg_bytes,
            headers={"Content-Type": "image/jpeg", "X-Robot-Id": robot_id},
            timeout=1.0,
        )
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def encode_jpeg(frame) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def check_cloud(cloud_url: str) -> bool:
    try:
        return httpx.get(f"{cloud_url}/health", timeout=2.0).status_code == 200
    except Exception:
        return False


# ── Source 1: DimOS LCM ───────────────────────────────────────
# DimOS transport is platform-dependent:
#   Linux → LCMTransport (UDP multicast, confirmed working)
#   Mac   → pSHMTransport (shared memory, see _mac_transports in unitree_go2_basic.py)
# On Linux, subscribe to the LCM channel directly with lcm_decode().

LCM_CHANNEL = "/color_image#sensor_msgs.Image"


def run_dimos_lcm(cloud_url: str, robot_id: str, fps: int):
    try:
        import lcm as lcmlib
        from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
    except ImportError:
        raise RuntimeError("lcm/dimos not importable — run inside the dimos venv")

    log.info(f"[LCM] Subscribing to '{LCM_CHANNEL}'...")
    lc = lcmlib.LCM()
    latest = [None]

    def _on_image(channel, data):
        try:
            latest[0] = Image.lcm_decode(data)
        except Exception as e:
            log.debug(f"LCM decode error: {e}")

    lc.subscribe(LCM_CHANNEL, _on_image)
    log.info("[LCM] Subscribed. Waiting for frames from DimOS...")

    interval = 1.0 / fps
    last_push = 0.0
    empty_ticks = 0

    while True:
        # Non-blocking LCM handle (50ms timeout)
        lc.handle_timeout(50)

        img = latest[0]
        now = time.time()
        if img is not None and now - last_push >= interval:
            latest[0] = None
            last_push = now
            empty_ticks = 0
            try:
                arr = img.data
                if img.format == ImageFormat.RGB:
                    arr = arr[:, :, ::-1]  # RGB → BGR for cv2
                jpeg = encode_jpeg(arr)
                if jpeg:
                    ok = push_frame(cloud_url, robot_id, jpeg)
                    log.debug(f"pushed {len(jpeg)//1024}KB {'ok' if ok else 'FAIL'}")
            except Exception as e:
                log.debug(f"encode error: {e}")
        else:
            empty_ticks += 1
            if empty_ticks == 100:
                log.warning("[LCM] No frames yet — is DimOS simulation running?")


# ── Source 2: RTSP (real Go2 / IP camera) ────────────────────

def run_rtsp(cloud_url: str, robot_id: str, fps: int, rtsp_url: str):
    log.info(f"[RTSP] Opening {rtsp_url} ...")
    interval = 1.0 / fps
    last_push = 0.0

    while True:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning(f"[RTSP] Cannot open {rtsp_url} — retrying in 3s")
            time.sleep(3.0)
            continue

        log.info("[RTSP] Stream open.")
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("[RTSP] Lost stream — reconnecting...")
                break
            now = time.time()
            if now - last_push >= interval:
                last_push = now
                jpeg = encode_jpeg(frame)
                if jpeg:
                    push_frame(cloud_url, robot_id, jpeg)

        cap.release()
        time.sleep(2.0)


# ── Source 3: OpenCV device (USB / v4l2) ─────────────────────

def run_opencv(cloud_url: str, robot_id: str, fps: int, device: int):
    log.info(f"[OpenCV] Opening device {device} ...")
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    log.info("[OpenCV] Camera open.")

    interval = 1.0 / fps
    last_push = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        now = time.time()
        if now - last_push >= interval:
            last_push = now
            jpeg = encode_jpeg(frame)
            if jpeg:
                push_frame(cloud_url, robot_id, jpeg)


# ── Source 4: ROS2 (future) ───────────────────────────────────
#
#   import rclpy
#   from rclpy.node import Node
#   from sensor_msgs.msg import Image as RosImage
#   from cv_bridge import CvBridge
#
#   class CameraNode(Node):
#       def __init__(self, cloud_url, robot_id, fps):
#           super().__init__('camera_bridge')
#           self._cloud_url = cloud_url
#           self._robot_id  = robot_id
#           self._interval  = 1.0 / fps
#           self._last_push = 0.0
#           self._bridge    = CvBridge()
#           self.create_subscription(
#               RosImage, '/camera/color/image_raw', self._cb, 10)
#
#       def _cb(self, msg):
#           now = time.time()
#           if now - self._last_push < self._interval:
#               return
#           self._last_push = now
#           frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
#           jpeg = encode_jpeg(frame)
#           if jpeg:
#               push_frame(self._cloud_url, self._robot_id, jpeg)
#
#   def run_ros2(cloud_url, robot_id, fps):
#       rclpy.init()
#       rclpy.spin(CameraNode(cloud_url, robot_id, fps))
#       rclpy.shutdown()


# ── Auto-detect ───────────────────────────────────────────────

def run_auto(cloud_url: str, robot_id: str, fps: int, rtsp_url: str):
    # 1. DimOS LCM (Linux default transport for color_image)
    try:
        import lcm as lcmlib  # noqa: F401
        import dimos  # noqa: F401
        log.info("[auto] dimos+lcm found → LCM")
        run_dimos_lcm(cloud_url, robot_id, fps)
        return
    except (ImportError, RuntimeError) as e:
        log.info(f"[auto] LCM skipped: {e}")

    # 2. Go2 RTSP
    probe = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if probe.isOpened():
        probe.release()
        log.info(f"[auto] RTSP reachable → {rtsp_url}")
        run_rtsp(cloud_url, robot_id, fps, rtsp_url)
        return
    probe.release()
    log.info("[auto] RTSP not reachable")

    # 3. USB camera
    for dev in range(4):
        probe = cv2.VideoCapture(dev)
        if probe.isOpened():
            probe.release()
            log.info(f"[auto] Found USB camera device {dev}")
            run_opencv(cloud_url, robot_id, fps, dev)
            return
        probe.release()

    log.error("No camera source found. Specify --source explicitly.")
    log.error("  --source dimos   (dimos venv, LCM multicast, simulation running)")
    log.error("  --source rtsp    (real Go2: --rtsp-url rtsp://192.168.123.161:8554/video)")
    log.error("  --source opencv  (USB webcam: --device 0)")


# ── Entry point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Camera bridge → FastAPI /frames")
    parser.add_argument("--cloud-url", default=DEFAULT_CLOUD)
    parser.add_argument("--robot-id",  default=DEFAULT_ROBOT_ID)
    parser.add_argument("--source",    choices=["auto", "dimos", "rtsp", "opencv"],
                        default="auto")
    parser.add_argument("--rtsp-url",  default=DEFAULT_RTSP)
    parser.add_argument("--device",    type=int, default=0)
    parser.add_argument("--fps",       type=int, default=DEFAULT_FPS)
    args = parser.parse_args()

    log.info(f"cloud={args.cloud_url}  robot={args.robot_id}  fps={args.fps}  source={args.source}")
    if not check_cloud(args.cloud_url):
        log.warning("Cloud not reachable yet — will retry on each push")

    try:
        if args.source == "auto":
            run_auto(args.cloud_url, args.robot_id, args.fps, args.rtsp_url)
        elif args.source == "dimos":
            run_dimos_lcm(args.cloud_url, args.robot_id, args.fps)
        elif args.source == "rtsp":
            run_rtsp(args.cloud_url, args.robot_id, args.fps, args.rtsp_url)
        elif args.source == "opencv":
            run_opencv(args.cloud_url, args.robot_id, args.fps, args.device)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()
