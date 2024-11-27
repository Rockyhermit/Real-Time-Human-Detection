import os
import datetime
import cv2
import argparse
import pygame
from ultralytics import YOLO
import supervision as sv
import numpy as np
import ctypes

# Alarm sound initialization
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(r"sound_small.mp3")

ZONE_POLYGON = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Create 'Rec' folder if it doesn't exist
    if not os.path.exists('Rec'):
        os.makedirs('Rec')

    # Initialize video capture object
    cap = cv2.VideoCapture(1)  # External webcam (or built-in webcam if not available)
    if not cap.read()[0]:
        cap = cv2.VideoCapture(0)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    
    WINDOW_NAME = 'Human Detection System'
    cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    model = YOLO("yolov8n.pt")
    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.65
    )

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_dem = [width, height]
    zone_polygon = (ZONE_POLYGON * np.array(vid_dem)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))

    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Initialize recording flag and frame counter
    recording = False
    output_file = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cl1 = clahe.apply(gray)
        frame = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
        frame = cv2.flip(frame, 1)

        # Run object detection
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ in detections
        ]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Trigger sound when human is detected
        if len(detections) > 0:
            alarm_sound.play()

        # Add text on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (20, 50)  # Starting position for text (top-left corner)
        font_scale = 0.65
        font_color = (0, 0, 0)  # Black color
        line_thickness = 2

        # Add "Humans Detected" text
        cv2.putText(frame, "Humans Detected: ", position, font, font_scale, font_color, line_thickness)
        num_humans_in_zone = len(detections)
        cv2.putText(frame, str(num_humans_in_zone), (position[0] + 200, position[1]), font, font_scale, (0, 255, 0), line_thickness)

        # Display Date and Time
        now = datetime.datetime.now()
        date_position = (int(0.5 * frame.shape[1]), 50)
        time_position = (int(0.5 * frame.shape[1]), 80)
        
        cv2.putText(frame, f"Date: {now.date()}", date_position, font, font_scale, font_color, line_thickness)
        cv2.putText(frame, f"Time: {now.time()}", time_position, font, font_scale, font_color, line_thickness)

        # Show recording status
        if recording:
            cv2.putText(frame, 'Recording: ON', (position[0], position[1] + 50), font, font_scale, (0, 0, 255), line_thickness)
        else:
            cv2.putText(frame, 'Recording: OFF', (position[0], position[1] + 50), font, font_scale, font_color, line_thickness)

        # Resize the frame to fit the screen
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        screen_height = ctypes.windll.user32.GetSystemMetrics(1)
        scaleWidth = float(screen_width) / float(frame.shape[1])
        scaleHeight = float(screen_height) / float(frame.shape[0])
        imgScale = min(scaleWidth, scaleHeight)
        new_x, new_y = int(frame.shape[1] * imgScale), int(frame.shape[0] * imgScale)
        frame = cv2.resize(frame, (new_x, new_y))

        cv2.imshow(WINDOW_NAME, frame)

        # Handle keypress events
        key = cv2.waitKey(1)
        
        if key == ord('r'):  # Start/Stop recording
            if recording:
                output_file.release()
                recording = False
                print('Recording stopped.')
            else:
                now = datetime.datetime.now()
                # Save the video file inside the 'rec' folder
                filename = os.path.join('Rec', now.strftime("%Y-%m-%d_%H-%M-%S") + ".avi")
                output_file = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
                recording = True
                print(f'Recording started. File saved in "Rec" folder as: {filename}')

        if recording:
            output_file.write(frame)

        if key == ord('q'):  # Quit
            break

    # Release resources
    cap.release()
    if output_file:
        output_file.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
