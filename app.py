import csv
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import requests

# =========================
# CONFIG
# =========================
LOG_FILE = "seedling_log.csv"
SNAPSHOT_DIR = Path("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Seedling Health Detection System", layout="wide")

# =========================
# TELEGRAM CONFIG
# =========================
try:
    TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
except Exception:
    TELEGRAM_TOKEN = "YOUR_TOKEN"

# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    if os.path.exists("best12s.pt"):
        return YOLO("best12s.pt"), "best12s.pt (custom)"
    return YOLO("yolo12s.pt"), "fallback (yolov12s)"

model, model_status = load_model()

# =========================
# LOG SYSTEM
# =========================
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow([
                "timestamp", "healthy", "unhealthy",
                "confidence", "latency_ms", "screenshot"
            ])

def read_log():
    init_log()
    return pd.read_csv(LOG_FILE)

def append_log(ts, h, u, conf, lat, filename):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, h, u, conf, lat, filename])

def clear_log():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp","healthy","unhealthy","confidence","latency_ms","screenshot"
        ])

def save_screenshot(frame, ts):
    filename = ts.replace(":", "-").replace(" ", "_") + ".jpg"
    path = SNAPSHOT_DIR / filename
    cv2.imwrite(str(path), frame)
    return filename

# =========================
# DETECTION ENGINE
# =========================
def run_detection(frame):
    result = model.predict(
        frame,
        conf=conf_thres,
        verbose=False
    )[0]

    annotated = result.plot()

    healthy = 0
    unhealthy = 0
    max_conf = 0.0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            max_conf = max(max_conf, conf)

            class_name = str(model.names[cls]).lower().strip()

            if "unhealthy" in class_name:
                unhealthy += 1
            elif "healthy" in class_name:
                healthy += 1

    return annotated, healthy, unhealthy, max_conf

# =========================
# STATE
# =========================
init_log()

if "running" not in st.session_state:
    st.session_state.running = False

if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = 0

# =========================
# HEADER
# =========================
st.title("🌱 Seedling Health Detection System")
st.info(f"Model Loaded: {model_status}")

# =========================
# KPI
# =========================
df = read_log()

if len(df):
    df["healthy"] = pd.to_numeric(df["healthy"], errors="coerce").fillna(0)
    df["unhealthy"] = pd.to_numeric(df["unhealthy"], errors="coerce").fillna(0)
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce").fillna(0)

c1, c2, c3, c4 = st.columns(4)

c1.metric("📦 Total Logs", len(df))
c2.metric("🌿 Healthy", int(df["healthy"].sum()) if len(df) else 0)
c3.metric("⚠️ Unhealthy", int(df["unhealthy"].sum()) if len(df) else 0)
c4.metric(
    "⏱ Avg Latency",
    f"{df['latency_ms'].mean():.0f} ms" if len(df) else "0 ms"
)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    cam_index = st.number_input("Camera Index", 0, 10, 0)
    conf_thres = st.slider("Confidence", 0.0, 1.0, 0.5)
    fps_limit = st.slider("FPS", 1, 60, 30)
    log_interval = st.slider("Log Interval (seconds)", 1, 60, 5)

    st.divider()

    telegram_enabled = st.toggle("Telegram Alerts")

    chat_id = ""
    if telegram_enabled:
        chat_id = st.text_input("Chat ID", type="password")

    input_mode = st.radio(
        "Detection Mode",
        ["Live Camera", "Image Upload", "Video Upload"]
    )

    uploaded_image = None
    if input_mode == "Image Upload":
        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"]
        )

    uploaded_video = None
    if input_mode == "Video Upload":
        uploaded_video = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"]
        )

# =========================
# IMAGE MODE
# =========================
if input_mode == "Image Upload" and uploaded_image is not None:

    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    annotated, healthy, unhealthy, conf = run_detection(frame)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.success(f"🌿 Healthy: {healthy} | ⚠️ Unhealthy: {unhealthy}")

# =========================
# CONTROLS
# =========================
col1, col2, col3 = st.columns(3)

if col1.button("Start"):
    st.session_state.running = True
    st.rerun()

if col2.button("Stop"):
    st.session_state.running = False
    st.rerun()

if col3.button("Clear Log"):
    clear_log()
    st.success("Log cleared")

# =========================
# VIDEO MODE
# =========================
if input_mode == "Video Upload" and uploaded_video is not None:

    temp_path = "temp_video.mp4"
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_path)

    stframe = st.empty()
    progress = st.progress(0)

    total_h = 0
    total_u = 0
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        annotated, h, u, max_conf = run_detection(frame)

        total_h += h
        total_u += u

        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                      use_container_width=True)

        if total_frames > 0:
            progress.progress(min(frame_count / total_frames, 1.0))

    cap.release()

    st.subheader("🎬 Video Analysis Complete")
    c1, c2 = st.columns(2)

    c1.metric("🌿 Healthy Detections", total_h)
    c2.metric("⚠️ Unhealthy Detections", total_u)

    st.success(
        f"Finished {frame_count} frames | Healthy: {total_h} | Unhealthy: {total_u}"
    )

# =========================
# LIVE MODE
# =========================
if st.session_state.running and input_mode == "Live Camera":

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    frame_id = 0
    frame_box = st.empty()

    while st.session_state.running:

        t0 = time.perf_counter()

        ok, frame = cap.read()
        if not ok:
            st.error("Camera not detected")
            break

        frame_id += 1
        if frame_id % 2 != 0:
            continue

        start = time.perf_counter()

        annotated, healthy, unhealthy, max_conf = run_detection(frame)

        latency_ms = (time.perf_counter() - start) * 1000

        frame_box.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # =========================
        # LOGGING (ALL FRAMES)
        # =========================
        if time.time() - st.session_state.last_log_time > log_interval:

            # Update interval timer regardless of detection result
            st.session_state.last_log_time = time.time()

            # Log only when Healthy and/or Unhealthy is detected
            if healthy > 0 or unhealthy > 0:

                filename = save_screenshot(annotated, ts)

                append_log(
                    ts,
                    healthy,
                    unhealthy,
                    max_conf,
                    latency_ms,
                    filename
                )

        # =========================
        # TELEGRAM ALERTS
        # =========================
        if telegram_enabled and TELEGRAM_TOKEN and chat_id and unhealthy > 0:
            caption = f"🌱 UNHEALTHY DETECTED\nH:{healthy} U:{unhealthy}"

            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                data={"chat_id": chat_id, "caption": caption},
                files={"photo": cv2.imencode('.jpg', annotated)[1].tobytes()}
            )

        elapsed = time.perf_counter() - t0
        if elapsed < 1 / fps_limit:
            time.sleep((1 / fps_limit) - elapsed)

    cap.release()
    st.rerun()

# =========================
# LOG VIEWER
# =========================
st.divider()
st.subheader("📊 Detection Log")

df = read_log()

if len(df) == 0:
    st.info("No logs yet.")
else:
    df = df.fillna("")

    # Ensure numeric columns
    df["healthy"] = pd.to_numeric(df["healthy"], errors="coerce").fillna(0).astype(int)
    df["unhealthy"] = pd.to_numeric(df["unhealthy"], errors="coerce").fillna(0).astype(int)

    st.dataframe(df, use_container_width=True)

    st.subheader("🖼️ Screenshot Viewer")

    # Create readable dropdown labels
    df["label"] = (
        df["timestamp"].astype(str)
        + " | H:" + df["healthy"].astype(str)
        + " | U:" + df["unhealthy"].astype(str)
    )

    selected = st.selectbox(
        "Select log entry",
        options=df.index.tolist(),
        format_func=lambda i: df.loc[i, "label"]
    )

    row = df.loc[selected]

    screenshot_name = str(row["screenshot"]).strip()
    img_path = SNAPSHOT_DIR / screenshot_name

    if screenshot_name and img_path.exists():

        img = cv2.imread(str(img_path))

        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            st.image(
                img,
                caption=(
                    f"{row['timestamp']} | "
                    f"Healthy: {row['healthy']} | "
                    f"Unhealthy: {row['unhealthy']}"
                ),
                use_container_width=True
            )
        else:
            st.error("Failed to load image.")

    else:
        st.warning("No screenshot available.")
