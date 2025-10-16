import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import datetime
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# --- Configuration and Setup ---

# Initialize YOLOv8 model
try:
    # Use the YOLOv8 model for object detection.
    # It will be downloaded automatically if you don't have it.
    model = YOLO('yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure you have a working internet connection to download the model or that the file 'yolov8n.pt' is in your directory.")
    exit()

# --- Email Alert Configuration ---
# IMPORTANT: Fill in your credentials below to enable email alerts.
# NOTE: You MUST use a Gmail "App Password" if you have 2-Factor Authentication enabled.
# Go to your Google Account -> Security -> 2-Step Verification -> App passwords.
SENDER_EMAIL = ""    # The email you are sending from (e.g., "your-email@gmail.com")
SENDER_PASSWORD = "" # The 16-character App Password you generated
RECIPIENT_EMAIL = "" # The email you are sending alerts to

# --- Alert Cooldown Configuration ---
ALERT_COOLDOWN_SECONDS = 60 # Cooldown period in seconds to prevent spamming alerts
last_fall_alert_time = None
last_boundary_alert_time = None

# --- Global variables for boundary drawing ---
boundary_points = []
drawing_complete = False

def draw_boundary_callback(event, x, y, flags, param):
    """Mouse callback function to capture points for the boundary."""
    global boundary_points, drawing_complete
    if drawing_complete:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        boundary_points.append((x, y))
        print(f"Point added: {(x, y)}. {len(boundary_points)} point(s) in total.")

def send_email_with_image(subject, body, image_path):
    """Sends an email with an attached image using Gmail SMTP."""
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
        if os.path.exists(image_path):
            os.remove(image_path)

def is_inside_boundary(center_point, boundary_polygon):
    """Checks if a point is inside the defined polygon boundary."""
    if boundary_polygon is None or len(boundary_polygon) < 3:
        return False
    # Use cv2.pointPolygonTest to determine if the point is inside the contour
    return cv2.pointPolygonTest(boundary_polygon, center_point, False) >= 0

def is_fall_detected(bbox, fall_threshold=1.5):
    """Detects a fall based on the aspect ratio of the bounding box."""
    try:
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        if height == 0: return False
        # If width is significantly greater than height, it might be a fall
        return (width / height) > fall_threshold
    except Exception as e:
        print(f"Error in fall detection logic: {e}")
        return False

def process_frame(frame, boundary_polygon, person_states):
    """
    Processes a single frame for person tracking, boundary crossing, and fall detection.
    
    Args:
        frame (np.ndarray): The input video frame.
        boundary_polygon (np.ndarray): The vertices of the defined boundary.
        person_states (dict): A dictionary to maintain the state of each tracked person.
    """
    global last_boundary_alert_time, last_fall_alert_time

    # Use model.track() for persistent object tracking
    results = model.track(frame, persist=True, classes=[0], verbose=False) # Class 0 is 'person'
    
    if results[0].boxes.id is None:
        return frame # No persons tracked in this frame

    current_frame_person_ids = set()
    tracked_boxes = results[0].boxes.cpu().numpy()

    for box in tracked_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_id = int(box.id[0])
        current_frame_person_ids.add(person_id)

        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        is_currently_inside = is_inside_boundary((center_x, center_y), boundary_polygon)
        
        # Get the previous state of the person, default to 'outside' if new
        previous_state = person_states.get(person_id, {}).get('status', 'outside')

        # --- Boundary Crossing Alert Logic ---
        if previous_state == 'inside' and not is_currently_inside:
            current_time = datetime.datetime.now()
            if last_boundary_alert_time is None or (current_time - last_boundary_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                last_boundary_alert_time = current_time
                alert_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"ALERT: Person #{person_id} crossed the boundary at {alert_time_str}!")
                
                # Save frame for email alert
                image_filename = f"boundary_cross_alert_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_filename, frame)
                
                subject = "Security Alert: Person Crossed Boundary"
                body = f"A person (ID: {person_id}) was detected crossing the designated boundary at {alert_time_str}."
                send_email_with_image(subject, body, image_filename)
            
            # Update state to 'crossed' to show a different color
            person_states[person_id] = {'status': 'crossed', 'bbox': (x1, y1, x2, y2)}
        else:
             person_states[person_id] = {'status': 'inside' if is_currently_inside else 'outside', 'bbox': (x1, y1, x2, y2)}
        
        # --- Fall Detection Logic ---
        if person_states[person_id].get('fall_alerted') is not True and is_fall_detected((x1, y1, x2, y2)):
            current_time = datetime.datetime.now()
            if last_fall_alert_time is None or (current_time - last_fall_alert_time).total_seconds() > ALERT_COOLDOWN_SECONDS:
                last_fall_alert_time = current_time
                alert_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"ALERT: Fall detected for person #{person_id} at {alert_time_str}!")
                
                image_filename = f"fall_alert_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(image_filename, frame)
                subject = "Safety Alert: Fall Detected"
                body = f"A potential fall was detected for person (ID: {person_id}) at {alert_time_str}."
                send_email_with_image(subject, body, image_filename)
                
                person_states[person_id]['fall_alerted'] = True


    # --- Drawing Logic ---
    for pid, state_info in person_states.items():
        x1, y1, x2, y2 = state_info['bbox']
        status = state_info.get('status', 'outside')
        fall_alerted = state_info.get('fall_alerted', False)

        if fall_alerted:
            color, label = (0, 165, 255), f"#{pid} Fall Detected!" # Orange
        elif status == 'crossed':
            color, label = (0, 0, 255), f"#{pid} Crossed!" # Red
        elif status == 'inside':
            color, label = (0, 255, 0), f"#{pid} Inside" # Green
        else: # outside
            color, label = (255, 0, 0), f"#{pid} Person" # Blue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    # Clean up states for people who have left the frame
    for pid in list(person_states.keys()):
        if pid not in current_frame_person_ids:
            del person_states[pid]
            
    return frame

def main():
    global boundary_points, drawing_complete
    parser = argparse.ArgumentParser(description="Fall and Boundary Crossing Detection using YOLOv8.")
    parser.add_argument("--input", type=str, default="0", help="Path to input video file or '0' for webcam.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video file (e.g., output.mp4).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.input}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    window_name = "Define Monitoring Zone"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, draw_boundary_callback)

    print("\n--- Define Monitoring Zone ---")
    print("Click at least 3 points to form a polygon.")
    print("Press 'c' to CONFIRM the zone.")
    print("Press 'r' to RESET the points.")
    print("---------------------------\n")

    while not drawing_complete:
        frame_to_show = first_frame.copy()
        for point in boundary_points:
            cv2.circle(frame_to_show, point, 5, (0, 0, 255), -1)
        if len(boundary_points) > 1:
            cv2.polylines(frame_to_show, [np.array(boundary_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        
        cv2.imshow(window_name, frame_to_show)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if len(boundary_points) < 3:
                print("Error: Please select at least 3 points.")
                continue
            drawing_complete = True
        elif key == ord('r'):
            boundary_points = []
            print("Points reset.")

    cv2.destroyWindow(window_name)
    boundary_polygon = np.array(boundary_points, np.int32)
    print("Zone confirmed. Starting monitoring...")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video to the beginning
    
    person_states = {} # Main dictionary to track person states
    
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Draw the defined boundary on the frame
        cv2.polylines(frame, [boundary_polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Process the frame for detection and alerts
        processed_frame = process_frame(frame.copy(), boundary_polygon, person_states)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(processed_frame, timestamp, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(processed_frame)
        cv2.imshow("Monitoring", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Finished processing.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
