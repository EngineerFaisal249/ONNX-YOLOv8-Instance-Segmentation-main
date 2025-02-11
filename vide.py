import cv2
import onnxruntime as ort
from yoloseg import YOLOSeg

# Check and initialize ONNX runtime session with GPU
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
session_options = ort.SessionOptions()

# Path to the ONNX model
model_path = r"models\yolov8m-seg.onnx"

# Initialize YOLOv8 Instance Segmentator with GPU
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Initialize video
video_path = r'vid 3.mp4'  # Replace this with the path to your local video
cap = cv2.VideoCapture(video_path)

# Optionally, set the starting time of the video in seconds
start_time = 22  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)

# Define display and output resolution
output_width, output_height = 640, 480

# Video writer for saving output video at the specified resolution
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_640x480.mp4", fourcc, fps, (output_width, output_height))

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)

while cap.isOpened():
    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

    # Read frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Update object localizer
    boxes, scores, class_ids, masks = yoloseg(frame)

    # Draw the masks on the frame
    combined_img = yoloseg.draw_masks(frame, mask_alpha=0.4)
    
    # Resize the frame for saving and display
    resized_frame = cv2.resize(combined_img, (output_width, output_height))
    
    # Save the resized frame to the output video
    out.write(resized_frame)
    
    # Display the frame
    cv2.imshow("Detected Objects", resized_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
