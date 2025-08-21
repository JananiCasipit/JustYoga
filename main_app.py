# webrtc_main_app.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import os
import sys
import time
import threading
import re

from nicegui import ui, app
from starlette.responses import StreamingResponse

from pose_estimation_logic import process_frame_for_pose
from pose_corrector import provide_feedback, calculate_angle
from llm_feedback import get_llm_feedback

# =======================
# Performance Settings
# =======================
TARGET_FPS = 30
STREAM_FPS = 30
STREAM_WIDTH = 640
STREAM_HEIGHT = 480
JPEG_QUALITY = 85

# Thread-safe frame sharing
latest_processed_frame = None
latest_landmarks = None
frame_lock = threading.Lock()
processing_active = True

# UI refs
pose_label = None
feedback_text_area = None
choreo_label = None
timer_label = None
music_file_display = None
score_display = None
start_button = None  # single toggle

# Load classifier
classifier_model = None
try:
    classifier_model = joblib.load('random_forest_pose_model.pkl')
    print("Console: Pose classification model loaded successfully.")
except FileNotFoundError:
    print("Console Error: Pose classification model not found.")
    sys.exit(1)

# Music
pygame.mixer.init()
DEFAULT_MUSIC_FILE = 'spiritual_song.mp3'

# =======================
# Game state
# =======================
YOGA_ROUTINE = [
    {'pose': 'downward_dog', 'duration': 15},
    {'pose': 'tree_pose', 'duration': 15},
    {'pose': 'warrior_ii', 'duration': 15},
]
current_score = 0
current_pose_index = 0
pose_start_time = 0.0
routine_in_progress = False

# New: per-pose flags
ai_feedback_sent_for_pose = False        # ensure one LLM tip per pose
countdown_notified = set()               # remembers {3,2,1} already shown

# =======================
# Label Canonicalization
# =======================
def normalize_pose_label(label: str) -> str:
    if not label:
        return ''
    s = re.sub(r'[^a-z0-9]+', '_', label.strip().lower())
    s = re.sub(r'_+', '_', s).strip('_')
    return s

POSE_SYNONYMS = {
    'warrior_2': 'warrior_ii',
    'warriorii': 'warrior_ii',
    'down_dog': 'downward_dog',
    'downwarddog': 'downward_dog',
    'tree': 'tree_pose',
}

def canonical_pose(label: str) -> str:
    n = normalize_pose_label(label)
    return POSE_SYNONYMS.get(n, n)

def friendly_pose(label_can: str) -> str:
    SPECIAL_CASES = {
        "warrior_ii": "Warrior II",
        "warrior_2": "Warrior II",
    }
    if label_can in SPECIAL_CASES:
        return SPECIAL_CASES[label_can]
    return label_can.replace('_', ' ').title()

# =======================
# MJPEG Video Stream
# =======================
def generate_video_stream():
    global latest_processed_frame, frame_lock, processing_active
    while processing_active:
        try:
            with frame_lock:
                if latest_processed_frame is not None:
                    frame = latest_processed_frame.copy()
                else:
                    frame = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(frame, "Connecting...", (200, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0 / STREAM_FPS)
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(0.1)

@app.get('/video_stream')
def video_stream():
    return StreamingResponse(
        generate_video_stream(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

# =======================
# Camera Processing Thread
# =======================
def video_processor():
    global latest_processed_frame, latest_landmarks, frame_lock, processing_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam in processor thread")
        return
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Video processor started")

    while processing_active:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01); continue
            frame = cv2.flip(frame, 1)  # mirror
            frame_with_landmarks, landmarks = process_frame_for_pose(frame)
            with frame_lock:
                latest_processed_frame = frame_with_landmarks
                latest_landmarks = landmarks
            time.sleep(1.0 / TARGET_FPS)
        except Exception as e:
            print(f"Video processing error: {e}")
            time.sleep(0.1)
    cap.release()
    print("Video processor stopped")

# =======================
# Game Logic
# =======================
def start_routine():
    global routine_in_progress, current_pose_index, pose_start_time, current_score
    global ai_feedback_sent_for_pose, countdown_notified

    if routine_in_progress:
        return
    routine_in_progress = True
    current_pose_index = 0
    pose_start_time = time.time()
    current_score = 0
    ai_feedback_sent_for_pose = False
    countdown_notified = set()

    if score_display: score_display.set_text('0')
    if feedback_text_area: feedback_text_area.set_text("Nice! Hold steady and follow the cues.")
    if start_button: start_button.set_text('⏹ EXIT ROUTINE')

def stop_routine():
    global routine_in_progress, current_pose_index, pose_start_time, current_score
    global ai_feedback_sent_for_pose, countdown_notified

    routine_in_progress = False
    current_pose_index = 0
    pose_start_time = 0.0
    current_score = 0
    ai_feedback_sent_for_pose = False
    countdown_notified = set()

    if timer_label: timer_label.set_text('0s')
    if choreo_label: choreo_label.set_text('Press START to begin your yoga journey!')
    if feedback_text_area: feedback_text_area.set_text("Routine stopped. Press START when you're ready.")
    if score_display: score_display.set_text('0')
    if pose_label: pose_label.set_text('Ready')
    if start_button: start_button.set_text('🚀 START ROUTINE')

def toggle_routine():
    if not routine_in_progress: start_routine()
    else: stop_routine()

def update_feedback_async(feedback_text):
    if feedback_text_area:
        feedback_text_area.set_text(feedback_text)

async def update_game_logic():
    global current_pose_index, pose_start_time, routine_in_progress
    global current_score, latest_landmarks
    global ai_feedback_sent_for_pose, countdown_notified

    try:
        if score_display:
            score_display.set_text(str(current_score))

        if routine_in_progress and current_pose_index < len(YOGA_ROUTINE):
            # Time remaining
            elapsed = time.time() - pose_start_time
            remaining = max(0, int(YOGA_ROUTINE[current_pose_index]['duration'] - elapsed))
            if timer_label: timer_label.set_text(f'{remaining}s')

            # Bottom-left countdown toasts (once each)
            if remaining in (3, 2, 1) and remaining not in countdown_notified:
                ui.notify(f"🔔 {remaining} " + ("second" if remaining == 1 else "seconds") + "!",
                          type='warning', position='bottom-left')
                countdown_notified.add(remaining)

            # Transition to next pose
            if remaining <= 0:
                current_pose_index += 1
                ai_feedback_sent_for_pose = False
                countdown_notified = set()
                if current_pose_index >= len(YOGA_ROUTINE):
                    routine_in_progress = False
                    if timer_label: timer_label.set_text('0s')
                    if feedback_text_area:
                        feedback_text_area.set_text(f"🎊 FANTASTIC! Final Score: {current_score} points! 🏆")
                    if choreo_label:
                        choreo_label.set_text("🎉 You're a Yoga Champion! 🎉")
                    if start_button:
                        start_button.set_text('🚀 START ROUTINE')
                    return
                pose_start_time = time.time()
                next_pose_name = friendly_pose(YOGA_ROUTINE[current_pose_index]['pose'])
                ui.notify(f"🔄 Transition to: {next_pose_name}", type='info', position='bottom-left')

            # HUD text
            current_pose_name = friendly_pose(YOGA_ROUTINE[current_pose_index]['pose'])
            upcoming = [friendly_pose(p['pose']) for p in YOGA_ROUTINE[current_pose_index + 1:]]
            if choreo_label:
                choreo_label.set_text(
                    f"🎯 {current_pose_name} — " + ("Up next: " + " → ".join(upcoming) if upcoming else "FINAL POSE! 🏁")
                )

            # Classify & score
            with frame_lock:
                landmarks = latest_landmarks

            if (landmarks is not None) and (classifier_model is not None):
                try:
                    predicted_pose_raw = classifier_model.predict([landmarks])[0]
                    required_pose_raw = YOGA_ROUTINE[current_pose_index]['pose']

                    predicted_pose_can = canonical_pose(predicted_pose_raw)
                    required_pose_can = canonical_pose(required_pose_raw)

                    if pose_label:
                        pose_label.set_text(f"🧘 {friendly_pose(predicted_pose_can)}")

                    if predicted_pose_can == required_pose_can:
                        # continuous scoring while correct
                        current_score += 1

                        # ONE LLM tip per pose
                        if not ai_feedback_sent_for_pose:
                            ai_feedback_sent_for_pose = True
                            def get_feedback_async():
                                try:
                                    issues = provide_feedback(landmarks, predicted_pose_can)
                                    return get_llm_feedback(friendly_pose(predicted_pose_can), issues)
                                except Exception as e:
                                    print(f"LLM Error: {e}")
                                    return "🔥 Great pose! Keep holding it!"
                            threading.Thread(
                                target=lambda: update_feedback_async(get_feedback_async()),
                                daemon=True
                            ).start()
                    else:
                        # keep coaching prompt stable; no LLM here
                        pass
                except Exception as e:
                    print(f"Classification error: {e}")
            else:
                if feedback_text_area:
                    feedback_text_area.set_text("📍 Step into the frame so I can see your pose!")
        else:
            # Idle mode
            if timer_label: timer_label.set_text('0s')
            with frame_lock:
                landmarks = latest_landmarks
            if (landmarks is not None) and (classifier_model is not None):
                try:
                    predicted_pose_raw = classifier_model.predict([landmarks])[0]
                    if pose_label:
                        pose_label.set_text(f"🧘 {friendly_pose(canonical_pose(predicted_pose_raw))}")
                except Exception as e:
                    print(f"Idle classification error: {e}")

    except Exception as e:
        print(f"Game logic error: {e}")

# =======================
# Music
# =======================
async def play_default_music():
    if DEFAULT_MUSIC_FILE and os.path.exists(DEFAULT_MUSIC_FILE):
        try:
            await ui.run_javascript(
                f'document.getElementById("audioPlayer").src = "{DEFAULT_MUSIC_FILE}";'
                'document.getElementById("audioPlayer").load();'
                'document.getElementById("audioPlayer").play();'
            )
            ui.notify(f"🎵 Now Playing: {os.path.basename(DEFAULT_MUSIC_FILE)}",
                      type='positive', position='bottom-left')
        except Exception as e:
            ui.notify(f"❌ Music Error: {e}", type='negative', position='bottom-left')
    else:
        ui.notify("🎵 Add spiritual_song.mp3 to enable music!",
                  type='warning', position='bottom-left')

def stop_music():
    ui.run_javascript('document.getElementById("audioPlayer").pause();')
    ui.notify("🔇 Music stopped", type='info', position='bottom-left')

# =======================
# Main UI
# =======================
@ui.page('/')
async def main_page():
    global pose_label, feedback_text_area, choreo_label, timer_label
    global music_file_display, score_display, start_button

    threading.Thread(target=video_processor, daemon=True).start()
        ui.label('JUST YOGA').classes('font-extrabold titlebar').style('font-size:20px;')

        with ui.column().classes('absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2').style(
            'width:min(92vw, 960px); align-items:center; z-index:20;'
        ):
            ui.html('''
                <div class="video-container">
                    <img class="video-stream" src="/video_stream" alt="Live Video Feed">
                </div>
            ''')

        # Score
        with ui.column().classes('hud').style('top:16px; left:16px;'):
            with ui.column().classes('panel').style('padding:10px 14px; min-width:110px; text-align:center;'):
                ui.label('🏆 SCORE').style('color:white; opacity:.85; font-weight:700; font-size:12px;')
                score_display = ui.label('0').style('color:#fde047; font-weight:900; font-size:26px;')

        # Timer
        with ui.column().classes('hud').style('top:16px; right:16px; text-align:center;'):
            with ui.column().classes('panel').style('padding:10px 14px; min-width:86px;'):
                ui.label('⏱️ TIME').style('color:white; opacity:.85; font-weight:700; font-size:12px;')
                timer_label = ui.label('0s').style('color:white; font-weight:900; font-size:26px;')

        # Detected pose
        with ui.column().classes('hud').style('top:60px; left:50%; transform:translateX(-50%); text-align:center;'):
            with ui.column().classes('panel').style('padding:8px 16px;'):
                ui.label('🧘 DETECTED POSE').style('color:#93c5fd; font-weight:800; font-size:12px;')
                pose_label = ui.label('Ready').style('color:white; font-weight:900; font-size:18px;')

        # Next pose line
        with ui.column().classes('hud').style('left:50%; transform:translateX(-50%); bottom:22px; text-align:center;'):
            with ui.column().classes('panel').style('padding:14px 18px; min-width:min(92vw, 800px);'):
                ui.label('🎯 NEXT POSE').style('color:white; font-weight:800; font-size:14px; margin-bottom:6px;')
                choreo_label = ui.label('Press START to begin your yoga journey!').style('color:white; font-weight:900; font-size:20px;')

        # Coach + controls
        with ui.column().classes('hud').style('right:16px; bottom:16px; gap:12px; align-items:flex-end;'):
            with ui.column().classes('panel').style('padding:12px 14px; max-width:400px; min-width:320px;'):
                ui.label('💬 AI COACH (ChatGPT)').style('color:#93c5fd; font-weight:800; font-size:14px; margin-bottom:6px;')
                feedback_text_area = ui.label(
                    "Step into the camera view and press START to begin your AI-guided yoga routine!"
                ).style('color:white; font-size:14px; line-height:1.4; word-wrap:break-word; white-space:normal;')

            with ui.column().classes('panel').style('padding:12px 14px; min-width:240px;'):
                music_file_display = ui.label(
                    f"🎵 {os.path.basename(DEFAULT_MUSIC_FILE) if DEFAULT_MUSIC_FILE and os.path.exists(DEFAULT_MUSIC_FILE) else 'No music found'}"
                ).style('color:white; opacity:.8; font-size:13px; margin-bottom:8px; text-align:center;')
                with ui.row().classes('items-center justify-center').style('gap:8px;'):
                    ui.button('🎵 PLAY', on_click=play_default_music).classes(
                        'bg-blue-600 hover:bg-blue-700 text-white font-bold px-3 py-2 rounded-lg transition-colors'
                    )
                    ui.button('🔇 STOP', on_click=stop_music).classes(
                        'bg-rose-600 hover:bg-rose-700 text-white font-bold px-3 py-2 rounded-lg transition-colors'
                    )

            with ui.column().classes('panel').style('padding:12px 14px; min-width:240px;'):
                with ui.row().classes('items-center justify-center').style('gap:8px;'):
                    global start_button
                    start_button = ui.button('🚀 START ROUTINE', on_click=toggle_routine).classes(
                        'bg-fuchsia-600 hover:bg-fuchsia-700 text-white font-bold px-3 py-2 rounded-lg transition-colors'
                    )

    ui.html('<audio id="audioPlayer" loop></audio>')
    ui.timer(0.1, update_game_logic)  # 10 fps logic

# =======================
# Alternative WebRTC demo
# =======================
@ui.page('/webrtc')
async def webrtc_page():
    ui.add_head_html('''
    <style>
        .video-container { 
            width: 100%; max-width: 960px; border-radius: 24px; overflow: hidden;
            box-shadow: 0 20px 40px rgba(0,0,0,0.35);
        }
        .webrtc-video { width: 100%; height: auto; transform: scaleX(-1); }
    </style>
    ''')
    with ui.column().classes('items-center p-4'):
        ui.label('🎥 WebRTC Live Feed').classes('text-2xl font-bold text-white mb-4')
        ui.html('''
            <div class="video-container">
                <video id="webrtcVideo" class="webrtc-video" autoplay muted playsinline></video>
            </div>
            <script>
                async function startWebRTC() {
                    try {
                        const video = document.getElementById('webrtcVideo');
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            video: { width: 640, height: 480, frameRate: 30 } 
                        });
                        video.srcObject = stream;
                        console.log('WebRTC stream started');
                    } catch (error) {
                        console.error('WebRTC error:', error);
                    }
                }
                startWebRTC();
            </script>
        ''')

# =======================
# Shutdown
# =======================
@app.on_shutdown
def handle_shutdown():
    global processing_active
    processing_active = False
    pygame.mixer.quit()
    print("Console: Application closed cleanly. Namaste! 🙏")

# =======================
# Run
# =======================
print("Starting Just Yoga with WebRTC streaming...")
ui.run(title="Just Yoga – WebRTC Stream", dark=True, port=8080, show=False)
