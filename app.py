import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, Response, request, jsonify
import threading
import io
import time # Import the time module

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Configuration and Setup ---

# Initialize YOLOv8 model
try:
    model = YOLO('yolov8n.pt')
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have a working internet connection or 'yolov8n.pt' is present.")
    exit()

# --- Email Alert Configuration ---
SENDER_EMAIL = ""    # The email you are sending from (e.g., "your-email@gmail.com")
SENDER_PASSWORD = "" # The 16-character App Password you generated
RECIPIENT_EMAIL = "" # The email you are sending alerts to

# --- Alert Cooldown Configuration ---
ALERT_COOLDOWN_SECONDS = 60 
last_fall_alert_time = None
last_boundary_alert_time = None

# --- Global State Variables for Server ---
cap = None                  # OpenCV VideoCapture object
video_source_path = None    # Store the path for reconnection
boundary_polygon = None     # User-defined boundary
person_states = {}          # Dictionary to track people
processing_thread = None    # Thread for background processing
output_frame = None         # The latest processed frame
lock = threading.Lock()     # Thread lock for output_frame
monitoring_active = False   # Flag to control the processing loop


# --- Email Sending Function (from your script) ---
def send_email_with_image(subject, body, image_path):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL]):
        print("Email credentials are not configured. Skipping email alert.")
        return

    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg.attach(MIMEText(body, 'plain'))
        image = MIMEImage(img_data, name=os.path.basename(image_path))
        msg.attach(image)
        
        print("Connecting to SMTP server...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(SENDER_EMAIL, SENDER_PASSWORD)
            s.send_message(msg)
        print(f"Email alert with image sent successfully to {RECIPIENT_EMAIL}")
    except Exception as e:
        print(f"Failed to send email: {e}")
    finally:
        # Clean up the saved image file after sending
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Cleaned up snapshot: {image_path}")
        except Exception as e:
            print(f"Error cleaning up file {image_path}: {e}")


# --- Detection Logic (from your script) ---
def is_inside_boundary(center_point, boundary_polygon):
    if boundary_polygon is None or len(boundary_polygon) < 3:
        return False
    return cv2.pointPolygonTest(boundary_polygon, center_point, False) >= 0

# --- REMOVED is_fall_detected function ---
# The logic is now inside process_frame for more robust, stateful detection.

# --- Frame Processing (from your script) ---
def process_frame(frame, boundary_polygon, person_states):
    global last_boundary_alert_time, last_fall_alert_time

    # Use model.track() for persistent object tracking
    results = model.track(frame, persist=True, classes=[0], verbose=False) # Class 0 is 'person'
    
    # Draw the defined boundary on the frame *before* processing
    if boundary_polygon is not None and len(boundary_polygon) > 2:
        cv2.polylines(frame, [boundary_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

    if results[0].boxes.id is None:
        return frame # No persons tracked

    current_frame_person_ids = set()
    tracked_boxes = results[0].boxes.cpu().numpy()

    # Make a copy of the frame for alerts, *before* drawing all boxes
    alert_frame_base = frame.copy()

    for box in tracked_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_id = int(box.id[0])
        current_frame_person_ids.add(person_id)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        current_width, current_height = x2 - x1, y2 - y1

        # --- Boundary Crossing Logic (Unchanged) ---
        is_currently_inside = is_inside_boundary((center_x, center_y), boundary_polygon)
        
        # Get the previous full state for this person
        previous_state = person_states.get(person_id, {})
        previous_status = previous_state.get('status', 'outside')
        
        # Initialize or update the state
        person_states[person_id] = previous_state # Start with old state
        person_states[person_id]['bbox'] = (x1, y1, x2, y2) # Update bbox

        if previous_status == 'inside' and not is_currently_inside:
            current_time = datetime.datetime.now()
            if last_boundary_alert_time is None or (current_time - last_boundary_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                last_boundary_alert_time = current_time
                alert_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"ALERT: Person #{person_id} crossed the boundary at {alert_time_str}!")
                
                # --- START: Boundary Alert Email Logic ---
                snapshot_filename = f"alert_boundary_{person_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                alert_frame = alert_frame_base.copy()
                cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(alert_frame, f"#{person_id} Crossed!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(snapshot_filename, alert_frame)
                subject = "Surveillance Alert: Boundary Crossing Detected"
                body = f"A person (ID #{person_id}) was detected crossing the defined boundary at {alert_time_str}."
                email_thread = threading.Thread(target=send_email_with_image, args=(subject, body, snapshot_filename))
                email_thread.daemon = True
                email_thread.start()
                # --- END: Boundary Alert Email Logic ---

            person_states[person_id]['status'] = 'crossed'
        else:
             person_states[person_id]['status'] = 'inside' if is_currently_inside else 'outside'
        
        # --- NEW: Improved Fall Detection Logic ---
        fall_alerted = previous_state.get('fall_alerted', False)
        
        # Get metrics from the *previous* frame for comparison
        last_center_y = previous_state.get('last_center_y', center_y)
        last_height = previous_state.get('last_height', current_height)

        is_fall = False
        if current_height > 0 and last_height > 0:
            # Condition 1: Check for "Fallen" state (aspect ratio)
            # Person is horizontal (width > height)
            aspect_ratio = current_width / current_height
            is_horizontal = aspect_ratio > 1.4 # Lowered threshold for sensitivity

            # Condition 2: Check for "Falling" motion (rapid collapse)
            # Vertical center dropped significantly (positive velocity)
            vertical_velocity = center_y - last_center_y
            # Height shrunk significantly (negative change)
            height_change = current_height - last_height
            
            # A "collapse" is a large downward velocity AND a large height shrinkage,
            # relative to the person's previous height.
            is_collapsing = (vertical_velocity > last_height * 0.3) and (height_change < -last_height * 0.3)

            # Trigger alert if not already alerted AND (they are collapsing OR they are horizontal)
            is_fall = (is_horizontal or is_collapsing)

        if not fall_alerted and is_fall:
            current_time = datetime.datetime.now()
            if last_fall_alert_time is None or (current_time - last_fall_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                last_fall_alert_time = current_time
                alert_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"ALERT: Fall detected for person #{person_id} at {alert_time_str}!")

                # --- START: Fall Alert Email Logic ---
                snapshot_filename = f"alert_fall_{person_id}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                alert_frame = alert_frame_base.copy()
                cv2.rectangle(alert_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(alert_frame, f"#{person_id} Fall Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imwrite(snapshot_filename, alert_frame)
                
                subject = "Surveillance Alert: Fall Detected"
                body = f"A potential fall was detected for person (ID #{person_id}) at {alert_time_str}."

                email_thread = threading.Thread(target=send_email_with_image, args=(subject, body, snapshot_filename))
                email_thread.daemon = True
                email_thread.start()
                # --- END: Fall Alert Email Logic ---

                person_states[person_id]['fall_alerted'] = True

        # Store current metrics for the *next* frame's comparison
        person_states[person_id]['last_center_y'] = center_y
        person_states[person_id]['last_height'] = current_height
        # --- END: New Fall Logic ---


    # --- Drawing Logic (on the main 'frame' to be streamed) ---
    for pid, state_info in person_states.items():
        if 'bbox' not in state_info: continue
        x1, y1, x2, y2 = state_info['bbox']
        status = state_info.get('status', 'outside')
        fall_alerted = state_info.get('fall_alerted', False)

        if fall_alerted:
            color, label = (0, 165, 255), f"#{pid} Fall Detected!"
        elif status == 'crossed':
            color, label = (0, 0, 255), f"#{pid} Crossed!"
        elif status == 'inside':
            color, label = (0, 255, 0), f"#{pid} Inside"
        else: 
            color, label = (255, 0, 0), f"#{pid} Person"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    for pid in list(person_states.keys()):
        if pid not in current_frame_person_ids:
            del person_states[pid]
            
    return frame

# --- Background Processing Thread ---
def processing_loop():
    """
    Main background thread for video processing.
    Includes reconnection logic for unstable streams.
    """
    global cap, output_frame, lock, monitoring_active, person_states, video_source_path

    print(f"Processing loop started for source: {video_source_path}")
    
    while monitoring_active:
        try:
            # --- 1. Check if capture is open ---
            if cap is None or not cap.isOpened():
                print(f"Capture not open. Attempting to connect to: {video_source_path}")
                # Re-initialize capture
                if isinstance(video_source_path, str) and video_source_path.isdigit():
                    source_input = int(video_source_path)
                else:
                    source_input = video_source_path
                
                cap = cv2.VideoCapture(source_input)
                
                if not cap.isOpened():
                    print("Failed to open video source. Retrying in 2s...")
                    time.sleep(2) # Wait 2 seconds before retrying
                    continue # Go to the next iteration
                print("Successfully re-connected to video source.")

            # --- 2. Try to read a frame ---
            success, frame = cap.read()
            if not success:
                # This is where the "Stream ends prematurely" error happens
                print("Frame read failed (stream end?). Re-opening connection...")
                cap.release()
                cap = None
                time.sleep(2) # Wait 2 seconds before retrying
                continue # Go to the next iteration of the while loop

            # --- 3. Process the frame ---
            processed_frame = process_frame(frame.copy(), boundary_polygon, person_states)
            
            # Add timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(processed_frame, timestamp, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # --- 4. Store the processed frame ---
            with lock:
                output_frame = processed_frame.copy()

        except Exception as e:
            # Handle any other unexpected errors
            print(f"Error in processing loop: {e}")
            if cap:
                cap.release()
            cap = None
            time.sleep(5) # Wait 5 seconds before trying to recover # Changed from time.stop
    
    # --- Loop has been stopped ---
    if cap:
        cap.release()
        cap = None
    print("Processing loop stopped.")


# --- Video Stream Generator ---
def generate_stream():
    global output_frame, lock, monitoring_active

    while monitoring_active:
        time.sleep(0.03) # Limit to ~30fps, reduces CPU load
        with lock:
            if output_frame is None:
                # Send a placeholder if no frame is ready
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Loading...", (220, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                (flag, encoded_image) = cv2.imencode(".jpg", placeholder)
                if not flag:
                    continue
            else:
                # Encode frame as JPEG
                (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
                if not flag:
                    continue
        
        # Yield the frame in multipart format
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded_image) + b'\r\n')

# --- Flask API Endpoints ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    # We use render_template to serve the HTML file from the 'templates' folder.
    # Make sure 'surveillance_app.html' is in a folder named 'templates'
    # For simplicity, let's just send the file from the current directory.
    return app.send_static_file('surveillance_app.html')

@app.route('/api/load_source', methods=['POST'])
def load_source():
    """Loads the video source and sends back the first frame."""
    global cap, video_source_path, monitoring_active
    
    # Stop any previous monitoring
    if monitoring_active:
        stop_monitoring()

    data = request.json
    source_path = data.get('source')

    if not source_path:
        return jsonify({"error": "No source path provided"}), 400

    # Release previous capture if it exists
    if cap:
        cap.release()
        cap = None

    # Create new capture
    if source_path.isdigit():
        source_input = int(source_path) # For webcam
    else:
        source_input = source_path # For video file path
        
    video_source_path = source_path # Store for reconnection
        
    try:
        print(f"Attempting to open source: {source_input}")
        cap = cv2.VideoCapture(source_input)
        if not cap.isOpened():
            raise Exception(f"Could not open video source: {source_path}")
    except Exception as e:
        print(f"Error opening source {source_input}: {e}")
        video_source_path = None
        return jsonify({"error": str(e)}), 500

    ret, frame = cap.read()
    if not ret:
        cap.release()
        cap = None
        video_source_path = None
        return jsonify({"error": "Could not read first frame"}), 500
    
    # Encode frame as JPEG
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        cap.release()
        cap = None
        video_source_path = None
        return jsonify({"error": "Could not encode frame"}), 500
    
    # Send frame back as a response
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Starts the background processing thread."""
    global boundary_polygon, person_states, processing_thread, monitoring_active, cap, video_source_path
    
    if cap is None or not cap.isOpened():
        if video_source_path is None:
             return jsonify({"error": "No video source loaded. Please load a source first."}), 400
        # Try to reconnect if cap was released
        try:
            if isinstance(video_source_path, str) and video_source_path.isdigit():
                source_input = int(video_source_path)
            else:
                source_input = video_source_path
            cap = cv2.VideoCapture(source_input)
            if not cap.isOpened():
                raise Exception("Failed to re-open video source")
        except Exception as e:
            return jsonify({"error": f"Failed to start monitoring: {e}"}), 500


    data = request.json
    points = data.get('boundary_points')
    if not points or len(points) < 3:
        return jsonify({"error": "Invalid boundary points"}), 400
        
    boundary_polygon = np.array(points, np.int32)
    person_states = {}
    
    # Stop thread if it is somehow still running
    if processing_thread and processing_thread.is_alive():
        monitoring_active = False
        processing_thread.join()

    monitoring_active = True
    
    # Start the background thread
    processing_thread = threading.Thread(target=processing_loop)
    processing_thread.daemon = True
    processing_thread.start()
    
    print("Monitoring started.")
    return jsonify({"status": "monitoring_started"})

@app.route('/api/video_stream')
def video_stream():
    """Returns the live video stream."""
    if not monitoring_active:
        print("Video stream request received, but monitoring is not active.")
        return "Monitoring not active.", 404
        
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stops the monitoring thread and releases resources."""
    global monitoring_active, processing_thread, cap, output_frame, boundary_polygon, person_states
    
    print("Received stop monitoring request.")
    monitoring_active = False
    
    if processing_thread:
        print("Waiting for processing thread to join...")
        processing_thread.join() # Wait for thread to finish
        processing_thread = None
        print("Processing thread joined.")
        
    if cap:
        print("Releasing video capture...")
        cap.release()
        cap = None
        print("Video capture released.")
        
    # Clear state variables
    output_frame = None
    boundary_polygon = None
    person_states = {}
    # Keep video_source_path so user can restart monitoring without reloading
    
    print("Monitoring stopped and resources partially released.")
    return jsonify({"status": "monitoring_stopped"})

# --- Run the App ---
if __name__ == "__main__":
    # Note: For this to find 'surveillance_app.html',
    # run this script from the same directory where the HTML file is saved.
    # Flask will automatically create a 'static' route.
    app.run(debug=True, host='0.0.0.0', threaded=True, use_reloader=False)

