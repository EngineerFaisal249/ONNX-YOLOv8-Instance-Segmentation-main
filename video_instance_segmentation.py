import cv2
from yoloseg import YOLOSeg

# Initialize video
video_path = r'vid 3.mp4'  # Replace this with the path to your local video
cap = cv2.VideoCapture(video_path)

# Initialize YOLOv8 Instance Segmentator
model_path = r"models\yolov8m-seg.onnx"
yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)

# Optionally, set the starting time of the video in seconds
start_time = 22  # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time * cap.get(cv2.CAP_PROP_FPS))

# Get original video properties
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define display resolution
display_width, display_height = 640, 480

# Video writer for saving output video at original resolution
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (original_width, original_height))

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
    
    # Save the frame to the output video
    out.write(combined_img)
    
    # Resize the frame for display
    display_frame = cv2.resize(combined_img, (display_width, display_height))
    
    # Display the frame
    cv2.imshow("Detected Objects", display_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
