# main_app.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
import os
import sys
import time
import base64 # Used for image encoding to base64


from nicegui import ui, app
from pose_estimation_logic import process_frame_for_pose
from pose_corrector import provide_feedback, calculate_angle
from llm_feedback import get_llm_feedback


# --- Global Variables and ML Components ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils # Not directly used here, but good to keep if needed elsewhere


classifier_model = None
try:
    classifier_model = joblib.load('random_forest_pose_model.pkl')
    print("Console: Pose classification model loaded successfully.")
except FileNotFoundError:
    print("Console Error: Pose classification model 'random_forest_pose_model.pkl' not found.")
    print("Console: Please run 'python pose_classifier.py' after collecting data.")
    sys.exit(1)


# --- Music Setup ---
pygame.mixer.init()
music_playing = False # Keep track of music state


# --- DEFINE YOUR DEFAULT MUSIC FILE HERE ---
# Place your spiritual music MP3 file in your project directory (e.g., next to main_app.py)
# and update this path. If it's in a subfolder, e.g., 'music/my_song.mp3', specify that.
DEFAULT_MUSIC_FILE = 'spiritual_song.mp3' # <<< REPLACE WITH YOUR ACTUAL SONG FILENAME


# Check if the default music file exists at startup
if not os.path.exists(DEFAULT_MUSIC_FILE):
    print(f"Console Warning: Default music file '{DEFAULT_MUSIC_FILE}' not found. Music playback may not work.")
    # Set to None if not found to prevent errors later when trying to play
    DEFAULT_MUSIC_FILE = None


# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Console Error: Could not open webcam. Please ensure your webcam is connected and not in use by another application.")
    sys.exit(1)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --- NiceGUI UI Definition ---
@ui.page('/')
async def main_page():
    # UI Elements
    ui.label("Just Yoga AI Posture Pal").classes('text-4xl font-bold text-white text-center w-full my-4')
    ui.separator().classes('w-full bg-gray-700 h-1')


    live_feed_img = ui.image('https://placehold.co/640x480/000000/FFFFFF?text=Loading+Webcam').classes('w-full max-w-xl rounded-lg shadow-lg')
    ui.separator().classes('w-full bg-gray-700 h-1 my-4')


    ui.label("Detected Pose:").classes('text-xl font-semibold text-gray-300')
    pose_label = ui.label("None").classes('text-3xl font-extrabold text-yellow-400 mt-1')


    ui.label("Guidance:").classes('text-xl font-semibold text-gray-300 mt-4')
    feedback_text_area = ui.label("Step into the frame to begin your yoga journey...").classes('text-lg text-white-700 bg-gray-800 p-4 rounded-lg mt-2 w-full max-w-xl text-center')


    ui.separator().classes('w-full bg-gray-700 h-1 my-4')


    # Music Controls (Simplified)
    ui.label("Spiritual Music:").classes('text-xl font-semibold text-gray-300')
    ui.html('<audio id="audioPlayer" loop></audio>') # HTML audio element
   
    # Display the name of the default music file
    music_file_display = ui.label(f"Default: {os.path.basename(DEFAULT_MUSIC_FILE) if DEFAULT_MUSIC_FILE else 'No music file found.'}").classes('text-base text-gray-400 mt-2')
   
    async def play_default_music():
        if DEFAULT_MUSIC_FILE:
            try:
                # NiceGUI can serve files from the project root by default if they exist
                # The browser will request this file from the NiceGUI server
                await ui.run_javascript(f'document.getElementById("audioPlayer").src = "{DEFAULT_MUSIC_FILE}"; document.getElementById("audioPlayer").load(); document.getElementById("audioPlayer").play();')
                ui.notify(f"Playing: {os.path.basename(DEFAULT_MUSIC_FILE)}", type='positive')
                print(f"Console: Playing default music file: '{DEFAULT_MUSIC_FILE}'.")
            except Exception as e:
                ui.notify(f"Error playing default music: {e}. Ensure file is valid and accessible.", type='negative')
                print(f"Console Error: Default music playback failed: {e}")
        else:
            ui.notify("No default music file configured or found.", type='warning')


    # Replaced 'Browse & Play Music' with 'Play Default Music'
    ui.button('Play Default Music', on_click=play_default_music).classes('bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg shadow-md')
    ui.button('Stop Music', on_click=lambda: ui.run_javascript('document.getElementById("audioPlayer").pause();')).classes('bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg shadow-md ml-2')


    ui.separator().classes('w-full bg-gray-700 h-1 my-4')
    ui.button('Exit Application', on_click=app.shutdown).classes('bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg shadow-md')


    # --- Real-time Webcam and ML Processing Loop ---
    last_feedback_time = time.time()
    feedback_interval = 3


    async def update_feed():
        nonlocal last_feedback_time
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_with_landmarks, landmarks = process_frame_for_pose(frame)


            # Convert frame to base64 for NiceGUI image update
            # Using JPEG encoding for potentially better performance and smaller size
            _, buffer = cv2.imencode('.jpeg', frame_with_landmarks, [cv2.IMWRITE_JPEG_QUALITY, 70]) # Quality 70 (0-100)
            img_base64 = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
            live_feed_img.set_source(img_base64)


            if classifier_model and landmarks and classifier_model.n_features_in_ is not None and len(landmarks) == classifier_model.n_features_in_:
                try:
                    predicted_pose = classifier_model.predict([landmarks])[0]
                    pose_label.set_text(predicted_pose.replace('_', ' ').title())
                    posture_issues = provide_feedback(landmarks, predicted_pose)
                    current_time = time.time()
                    if current_time - last_feedback_time > feedback_interval:
                        llm_response = get_llm_feedback(predicted_pose, posture_issues)
                        feedback_text_area.set_text(llm_response)
                        last_feedback_time = current_time
                    elif not posture_issues and "Excellent" in feedback_text_area.text:
                        pass
                    elif posture_issues and "Focus on grounding" not in feedback_text_area.text:
                         llm_response = get_llm_feedback(predicted_pose, posture_issues)
                         feedback_text_area.set_text(llm_response)
                         last_feedback_time = current_time
                except Exception as e:
                    pose_label.set_text("Error in classification")
                    feedback_text_area.set_text(f"Error: {e}")
            else:
                pose_label.set_text("No Pose Detected")
                feedback_text_area.set_text("Ensure full body is visible in frame.")
        else:
            pose_label.set_text("Webcam Error")
            feedback_text_area.set_text("Check webcam connection.")
            app.shutdown()


    ui.timer(0.05, update_feed)


# --- App Shutdown Hook ---
@app.on_shutdown
def handle_shutdown():
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("Console: Just Yoga application closed.")


# --- Run the NiceGUI app ---
ui.run(title="Just Yoga", dark=True, port=8080)




