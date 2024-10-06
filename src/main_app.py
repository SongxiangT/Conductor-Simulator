# main_app.py

import cv2
import mediapipe as mp
import numpy as np
import threading
import time
from collections import deque
import vlc  # VLC for audio playback
from filterpy.kalman import KalmanFilter
from scipy.ndimage import gaussian_filter1d  # For Gaussian smoothing
import sys
import os

# ================================
# Hand Tracking and Filtering
# ================================

# Kalman Filter class for smoothing hand landmarks
class HandKalmanFilter:
    def __init__(self):
        # 4D state [x, y, v_x, v_y]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # Time step

        # State transition matrix
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Covariance matrices
        self.kf.P *= 1000.0  # Initial uncertainty
        self.kf.R = np.eye(2) * 5  # Measurement noise
        self.kf.Q = np.eye(4) * 0.1  # Process noise

        self.kf.x = np.zeros((4, 1))

    def update(self, x, y):
        self.kf.predict()
        self.kf.update([x, y])
        x_filtered = self.kf.x[0]
        y_filtered = self.kf.x[1]
        return x_filtered, y_filtered

# ================================
# Utility Functions
# ================================

def initialize_kalman_filter(initial_value, measurement_noise=1.0, process_noise=1.0):
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([[initial_value], [0]])  # Initial state (value and its rate)
    kf.F = np.array([[1, 1],
                     [0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0]])  # Measurement function
    kf.P *= 1000.0  # Covariance matrix
    kf.R = measurement_noise  # Measurement noise
    kf.Q = np.array([[process_noise, 0],
                     [0, process_noise]])  # Process noise
    return kf

def exponential_smooth(current_value, previous_value, alpha=0.05):
    if previous_value is None:
        return current_value
    return previous_value + alpha * (current_value - previous_value)

def count_extended_fingers(hand_landmarks, frame_width, frame_height, hand_label, mp_hands):
    """
    Counts the number of extended fingers.
    """
    count = 0

    # Define finger tip and pip landmarks
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]

    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP
    ]

    # Calculate if fingers are extended
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y

        # For thumb, compare x coordinates instead of y
        if tip == mp_hands.HandLandmark.THUMB_TIP:
            tip_x = hand_landmarks.landmark[tip].x
            pip_x = hand_landmarks.landmark[pip].x
            if hand_label == 'Right':
                if tip_x > pip_x:
                    count += 1
            else:
                if tip_x < pip_x:
                    count += 1
        else:
            if tip_y < pip_y:
                count += 1

    return count

# Additional utility functions like `is_pause_posture`, `is_continue_posture`, etc., can be moved here for better organization.

# ================================
# Main Application Class
# ================================

class OrchestraConductorApp:
    def __init__(self, music_path, music_bpm):
        self.MUSIC_FILE_PATH = music_path
        self.MUSIC_BPM = music_bpm

        # Initialize variables
        self.volume = 50  # VLC volume range is 0 to 100
        self.tempo = 1.0  # Playback rate (1.0 = normal speed)
        self.stop_music = False
        self.music_thread = None
        self.player = None  # VLC player instance
        self.cue_triggered = False  # Flag to track if a cue has been triggered
        self.last_cue_time = 0  # Timestamp of the last cue
        self.BUFFER_TIME = 2.0  # Buffer time in seconds between cues

        # Define the required hold duration for the Pause gesture in seconds
        self.PAUSE_HOLD_DURATION = 1.0  # seconds

        # Define the average distance thresholds for the Pause gesture in pixels
        self.MIN_AVERAGE_DISTANCE = 20  # Minimum average distance
        self.MAX_AVERAGE_DISTANCE = 60  # Maximum average distance

        # Define extensivity thresholds for the Pause gesture in pixels
        self.MIN_EXTENSIVITY = 30  # Minimum extensivity
        self.MAX_EXTENSIVITY = 100  # Maximum extensivity

        # Initialize a deque to track the timestamps when Pause posture is detected
        self.pause_posture_timestamps = deque()

        # Initialize Kalman Filters for volume and tempo
        self.volume_kf = initialize_kalman_filter(self.volume, measurement_noise=1.0, process_noise=1.0)
        self.tempo_kf = initialize_kalman_filter(self.tempo, measurement_noise=0.01, process_noise=1.0)

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # For visualization
        self.volume_history = deque(maxlen=100)
        self.tempo_history = deque(maxlen=100)

        # Initialize variables for peak detection (for tempo control)
        self.last_velocity = None
        self.peak_times = []
        self.last_peak_time = None

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Gesture control variables
        self.left_wrist_positions = deque(maxlen=20)  # For tempo control
        self.left_hand_filter = HandKalmanFilter()

        # Initialize variables for 'Continue' gesture detection
        self.right_wrist_positions = deque(maxlen=20)  # For waving detection

    def play_music(self):
        # Create VLC instance
        instance = vlc.Instance()
        self.player = instance.media_player_new()
        media = instance.media_new(self.MUSIC_FILE_PATH)
        self.player.set_media(media)

        # Set initial volume
        self.player.audio_set_volume(int(self.volume))

        # Play media
        self.player.play()
        time.sleep(0.5)  # Wait for playback to start

        while not self.stop_music:
            # Apply Kalman filter to volume
            self.volume_kf.predict()
            self.volume_kf.update([self.volume])
            smoothed_volume = self.volume_kf.x[0]
            self.player.audio_set_volume(int(smoothed_volume))

            # Apply Kalman filter to tempo
            self.tempo_kf.predict()
            self.tempo_kf.update([self.tempo])
            smoothed_tempo = self.tempo_kf.x[0]
            self.player.set_rate(smoothed_tempo)

            # Handle 'cue' action
            if self.cue_triggered:
                current_time = time.time()
                # Toggle play/pause
                if self.player.is_playing():
                    self.player.pause()
                    print("Music Paused")
                else:
                    self.player.play()
                    print("Music Playing")
                # Update the last cue time AFTER performing the action
                self.last_cue_time = current_time
                # Reset the flag after action
                self.cue_triggered = False

            time.sleep(0.2)  # Delay to prevent high CPU usage

        self.player.stop()
        self.player.release()
        instance.release()

    def run(self):
        # Start the music playback thread
        if self.music_thread is None or not self.music_thread.is_alive():
            self.stop_music = False
            self.music_thread = threading.Thread(target=self.play_music)
            self.music_thread.start()

        # Create windows for video and curves
        cv2.namedWindow('Orchestra Conductor - Video')
        cv2.namedWindow('Orchestra Conductor - Curves')

        with self.mp_hands.Hands(max_num_hands=2,
                                 min_detection_confidence=0.7,
                                 min_tracking_confidence=0.7) as hands:
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    continue

                frame = cv2.flip(frame, 1)  # Mirror the image
                frame_height, frame_width, _ = frame.shape

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                current_time = time.time()

                if results.multi_hand_landmarks:
                    right_hand_landmarks = None
                    left_hand_landmarks = None
                    cue_detected = False  # Flag to check if 'cue' is detected

                    # Separate hands based on label
                    for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = hand_info.classification[0].label
                        # Default drawing
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        if hand_label == 'Right':
                            right_hand_landmarks = hand_landmarks
                        elif hand_label == 'Left':
                            left_hand_landmarks = hand_landmarks

                    # ====== Right Hand: Cue Posture Detection ======
                    if right_hand_landmarks:
                        # Check for 'Pause' posture (Stop Gesture)
                        if self.is_pause_posture(right_hand_landmarks, frame_width, frame_height, 'Right'):
                            # Check if enough time has passed since the last cue
                            if current_time - self.last_cue_time >= self.BUFFER_TIME:
                                cue_detected = True
                                self.cue_triggered = True  # Set flag to trigger cue action
                                # Do NOT update last_cue_time here
                                print("Pause Cue Gesture Detected")

                                # Highlight landmarks for visual feedback
                                self.mp_drawing.draw_landmarks(
                                    frame,
                                    right_hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                                )

                                # Draw a progress bar to indicate hold duration
                                hold_time = self.PAUSE_HOLD_DURATION  # seconds
                                progress_width = 200
                                progress_height = 20
                                progress_x = 10
                                progress_y = frame_height - 30

                                # Calculate how much time has been held
                                if self.pause_posture_timestamps:
                                    elapsed_time = current_time - self.pause_posture_timestamps[-1]
                                    progress = min(elapsed_time / hold_time, 1.0)
                                else:
                                    progress = 0.0

                                # Draw background rectangle
                                cv2.rectangle(frame, (progress_x, progress_y),
                                              (progress_x + progress_width, progress_y + progress_height),
                                              (50, 50, 50), 2)

                                # Draw progress
                                cv2.rectangle(frame, (progress_x, progress_y),
                                              (int(progress_x + progress_width * progress), progress_y + progress_height),
                                              (0, 255, 255), -1)

                                # Display countdown timer
                                countdown = max(0, hold_time - elapsed_time)
                                countdown_text = f"Holding: {countdown:.1f}s"
                                cv2.putText(frame, countdown_text, (progress_x, progress_y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Check for 'Continue' posture (Waving Gesture)
                        elif self.is_continue_posture(right_hand_landmarks, frame_width, frame_height, 'Right',
                                                     self.right_wrist_positions, current_time):
                            # Check if enough time has passed since the last cue
                            if current_time - self.last_cue_time >= self.BUFFER_TIME:
                                cue_detected = True
                                self.cue_triggered = True  # Set flag to trigger cue action
                                # Do NOT update last_cue_time here
                                print("Continue Cue Gesture Detected")

                                # Highlight landmarks for visual feedback
                                self.mp_drawing.draw_landmarks(
                                    frame,
                                    right_hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                                )

                    # ====== Left Hand: Tempo and Volume Control ======
                    if left_hand_landmarks:
                        wrist = left_hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                        x_raw = wrist.x * frame_width
                        y_raw = wrist.y * frame_height
                        x_filtered, y_filtered = self.left_hand_filter.update(x_raw, y_raw)
                        self.left_wrist_positions.append((x_filtered, y_filtered, current_time))

                        # Peak detection for waving time (tempo control)
                        if len(self.left_wrist_positions) >= 2:
                            y1 = self.left_wrist_positions[-2][1]
                            y2 = self.left_wrist_positions[-1][1]
                            t1 = self.left_wrist_positions[-2][2]
                            t2 = self.left_wrist_positions[-1][2]
                            dt = t2 - t1
                            if dt > 0:
                                vy = (y2 - y1) / dt

                                if self.last_velocity is not None:
                                    if self.last_velocity > 0 and vy <= 0:
                                        # Peak detected
                                        peak_time = t1
                                        self.peak_times.append(peak_time)
                                        self.last_peak_time = peak_time

                                        if len(self.peak_times) >= 2:
                                            period = self.peak_times[-1] - self.peak_times[-2]
                                            if period > 0:
                                                user_bpm = 60 / period
                                                self.adjust_tempo(user_bpm)
                                                print(f"Detected User BPM: {user_bpm:.2f}")
                                self.last_velocity = vy

                        # Reset tempo if no peaks detected recently
                        if self.last_peak_time is not None and current_time - self.last_peak_time > 2.0:
                            self.adjust_tempo(None)

                        # Calculate extensivity for volume control
                        extensivity = self.calculate_extensivity(left_hand_landmarks, frame_width, frame_height)

                        # Map extensivity to volume range (adjust min and max based on your setup)
                        min_extensivity = 50  # Minimum possible extensivity
                        max_extensivity = 200  # Maximum possible extensivity

                        # Clamp extensivity
                        extensivity = max(min_extensivity, min(max_extensivity, extensivity))

                        # Map extensivity to volume
                        target_volume = (extensivity - min_extensivity) / (max_extensivity - min_extensivity) * 100

                        # Apply exponential smoothing to volume
                        self.volume = exponential_smooth(target_volume, self.volume, alpha=0.05)

                        # Append to history
                        self.volume_history.append(self.volume)
                        self.tempo_history.append(self.tempo)

                else:
                    # No hands detected, reset tempo after 2 seconds
                    if self.last_peak_time is not None and current_time - self.last_peak_time > 2.0:
                        self.adjust_tempo(None)

                # Handle cue actions outside the hand detection block
                if cue_detected:
                    self.cue_triggered = True  # Set flag to trigger cue action
                    self.last_cue_time = current_time  # Update last_cue_time after performing the action

                # Display the video feed with landmarks
                cv2.imshow('Orchestra Conductor - Video', frame)

                # Visualize the volume and tempo curves in a separate window
                graph_frame = self.visualize_volume_and_tempo(self.volume_history, self.tempo_history)
                cv2.imshow('Orchestra Conductor - Curves', graph_frame)

                # Exit on pressing 'ESC'
                if cv2.waitKey(5) & 0xFF == 27:
                    self.stop_music = True
                    break

        self.cap.release()
        cv2.destroyAllWindows()

    # Define other methods like `is_pause_posture`, `is_continue_posture`, `calculate_extensivity`, `adjust_tempo`, `visualize_volume_and_tempo` here.

    def is_pause_posture(self, hand_landmarks, frame_width, frame_height, hand_label):
        """
        Determines if the hand is showing the 'Pause' posture (Stop Gesture)
        and has been held steadily for the required duration.
        """
        # Define 'Pause' as all five fingers extended and fingertips close together
        extended_fingers = count_extended_fingers(hand_landmarks, frame_width, frame_height, hand_label, self.mp_hands)

        if extended_fingers != 5:
            return False

        # Calculate the distances between all fingertips to ensure they are close
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]

        # Convert normalized coordinates to pixel coordinates
        fingertip_coords = np.array([[tip.x * frame_width, tip.y * frame_height] for tip in fingertips])

        # Calculate pairwise distances
        distances = []
        for i in range(len(fingertip_coords)):
            for j in range(i + 1, len(fingertip_coords)):
                distance = np.linalg.norm(fingertip_coords[i] - fingertip_coords[j])
                distances.append(distance)

        # Calculate the average distance between fingertips
        average_distance = np.mean(distances)

        # Define thresholds for the average distance
        if not (self.MIN_AVERAGE_DISTANCE < average_distance < self.MAX_AVERAGE_DISTANCE):
            return False

        # Additionally, calculate extensivity to ensure it's within a reasonable range
        extensivity = self.calculate_extensivity(hand_landmarks, frame_width, frame_height)

        # Define extensivity thresholds
        if not (self.MIN_EXTENSIVITY < extensivity < self.MAX_EXTENSIVITY):
            return False

        # Define a threshold for the maximum average distance (adjust as needed)
        if self.MIN_AVERAGE_DISTANCE < average_distance < self.MAX_AVERAGE_DISTANCE:
            # Append current timestamp to the deque
            self.pause_posture_timestamps.append(time.time())

            # Remove timestamps older than PAUSE_HOLD_DURATION
            while self.pause_posture_timestamps and (time.time() - self.pause_posture_timestamps[0] > self.PAUSE_HOLD_DURATION):
                self.pause_posture_timestamps.popleft()

            # Check if the posture has been held steadily for the required duration
            # Assuming the function is called approximately every 0.1 seconds
            if len(self.pause_posture_timestamps) * 0.1 >= self.PAUSE_HOLD_DURATION:
                # Clear the deque to prevent multiple triggers
                self.pause_posture_timestamps.clear()
                return True
        else:
            # Reset the deque if the posture is not held
            self.pause_posture_timestamps.clear()

        return False

    def is_continue_posture(self, hand_landmarks, frame_width, frame_height, hand_label, previous_positions, current_time):
        """
        Determines if the hand is showing the 'Continue' posture (Waving Gesture).
        """
        # Track horizontal wrist movement for waving
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        x = wrist.x * frame_width
        y = wrist.y * frame_height

        # Append current x position to previous_positions
        previous_positions.append((x, current_time))

        # Analyze movement over the last few positions
        if len(previous_positions) < 6:
            return False  # Not enough data to determine waving

        # Calculate the number of direction changes in horizontal movement
        directions = []
        for i in range(1, len(previous_positions)):
            if previous_positions[i][0] > previous_positions[i - 1][0]:
                directions.append(1)  # Moving right
            elif previous_positions[i][0] < previous_positions[i - 1][0]:
                directions.append(-1)  # Moving left
            else:
                directions.append(0)  # No movement

        # Count the number of times the direction changes
        direction_changes = 0
        for i in range(1, len(directions)):
            if directions[i] != 0 and directions[i] != directions[i - 1]:
                direction_changes += 1

        # Define a threshold for direction changes to qualify as waving
        if direction_changes >= 2:
            return True
        else:
            return False

    def calculate_extensivity(self, hand_landmarks, frame_width, frame_height):
        """
        Calculates the extensivity based on finger distances.
        """
        # Calculate distances between fingertips and palm center
        palm_coords = np.array([hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * frame_width,
                                hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * frame_height])

        finger_tips = [self.mp_hands.HandLandmark.THUMB_TIP,
                       self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                       self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                       self.mp_hands.HandLandmark.RING_FINGER_TIP,
                       self.mp_hands.HandLandmark.PINKY_TIP]

        distances = []
        for tip in finger_tips:
            tip_coords = np.array([hand_landmarks.landmark[tip].x * frame_width,
                                   hand_landmarks.landmark[tip].y * frame_height])
            distance = np.linalg.norm(tip_coords - palm_coords)
            distances.append(distance)

        # Calculate average distance (extensivity)
        extensivity = np.mean(distances)
        return extensivity

    def adjust_tempo(self, user_bpm):
        if self.MUSIC_BPM == 0:
            target_tempo = 1.0
        else:
            # Calculate the ratio of user BPM to music BPM
            target_tempo = user_bpm / self.MUSIC_BPM

        # Clamp the target tempo to reasonable limits
        min_tempo = 0.8
        max_tempo = 1.2
        target_tempo = max(min_tempo, min(max_tempo, target_tempo))

        # Apply exponential smoothing to the tempo
        self.tempo = exponential_smooth(target_tempo, self.tempo, alpha=0.05)

    def visualize_volume_and_tempo(self, volume_history, tempo_history):
        graph_height = 400
        graph_width = 600
        graph_frame = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

        max_volume = 100
        min_volume = 0
        max_tempo = 1.2
        min_tempo = 0.8

        volume_color = (0, 255, 0)  # Green
        tempo_color = (0, 0, 255)  # Blue

        margin_top = 20
        margin_bottom = 40

        # Adjusted plotting height
        plot_height = graph_height - margin_top - margin_bottom

        # Apply Gaussian smoothing to histories
        if len(volume_history) > 1:
            volume_history_smoothed = gaussian_filter1d(volume_history, sigma=2)
        else:
            volume_history_smoothed = list(volume_history)
        if len(tempo_history) > 1:
            tempo_history_smoothed = gaussian_filter1d(tempo_history, sigma=2)
        else:
            tempo_history_smoothed = list(tempo_history)

        # Plot volume history
        for i in range(1, len(volume_history_smoothed)):
            if len(volume_history_smoothed) > 1:
                x1 = int((i - 1) / (len(volume_history_smoothed) - 1) * graph_width)
                y1 = int((1 - (volume_history_smoothed[i - 1] - min_volume) / (max_volume - min_volume)) * plot_height) + margin_top
                x2 = int(i / (len(volume_history_smoothed) - 1) * graph_width)
                y2 = int(
                    (1 - (volume_history_smoothed[i] - min_volume) / (max_volume - min_volume)) * plot_height) + margin_top
                cv2.line(graph_frame, (x1, y1), (x2, y2), volume_color, 2)

        # Plot tempo history
        for i in range(1, len(tempo_history_smoothed)):
            if len(tempo_history_smoothed) > 1:
                x1 = int((i - 1) / (len(tempo_history_smoothed) - 1) * graph_width)
                y1 = int(
                    (1 - (tempo_history_smoothed[i - 1] - min_tempo) / (max_tempo - min_tempo)) * plot_height) + margin_top
                x2 = int(i / (len(tempo_history_smoothed) - 1) * graph_width)
                y2 = int((1 - (tempo_history_smoothed[i] - min_tempo) / (max_tempo - min_tempo)) * plot_height) + margin_top
                cv2.line(graph_frame, (x1, y1), (x2, y2), tempo_color, 2)

        # Display numerical values at the bottom
        font = cv2.FONT_HERSHEY_SIMPLEX
        if volume_history:
            volume_text = f"Volume: {int(volume_history[-1])}%"
            cv2.putText(graph_frame, volume_text, (10, graph_height - 15), font, 0.7, volume_color, 2)
        if tempo_history:
            tempo_text = f"Tempo: {tempo_history[-1]:.2f}x"
            cv2.putText(graph_frame, tempo_text, (graph_width - 200, graph_height - 15), font, 0.7, tempo_color, 2)

        return graph_frame
