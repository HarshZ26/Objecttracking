import cv2
import numpy as np
import sys
import glob
import time
import torch
import torchvision.transforms as transforms
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.detection import Detection
from collections import Counter

transform = transforms.Compose([transforms.ToTensor()])

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
my_tracker = DeepSort(max_age=5,max_cosine_distance=0.5, nms_max_overlap=1.0)

# Open video file
cap = cv2.VideoCapture(1) # Change the camera index as per system
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480))  
# Define colors for different tracks
color_list = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
# Loop through frames
while True:
    # Read frame
    ret, frame = cap.read()
    start = time.perf_counter()
    if not ret:
        break

    # Detect objects with YOLOv5
    results = model(frame)
    # Extract features for each object
    detections = []
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj.cpu().numpy()
        if cls==0:
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]
            resized = cv2.resize(cropped, (224, 224))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).cuda()
            with torch.no_grad():
                feature = model.forward(tensor)
                feature = feature.squeeze().cpu().numpy()
            detection = Detection(np.array([x1, y1, int(x2-x1), int(y2-y1)]), conf, feature,class_name='person')
            detections.append(detection)

    # Associate objects across frames with SORT and DeepSORT
    ids = []
    total = 0
    if len(detections) > 0:
        # Run SORT tracker
        # print("here",len(detections))
        # Run DeepSORT tracker
        deepsort_detections = []
        for detection in detections:
            detection = (list(detection.get_ltwh()), detection.confidence, detection.class_name)
            deepsort_detections.append(detection)
        deepsort_tracks = my_tracker.update_tracks(raw_detections = deepsort_detections, frame = frame)
        # Draw bounding boxes and track IDs on frame
        for track in deepsort_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            ids.append(int(track_id))
            bbox = ltrb
            
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(color_list[int(track_id)%1000].tolist()),2)
            cv2.putText(frame, "ID: " + str(track_id), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (color_list[int(track_id)%1000].tolist()), 2)
    # Display output frame
        total = len(Counter(ids).keys())
    end = time.perf_counter()
    totaltime = end - start
    fps = 1 / totaltime
    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(frame, f'Count: {int(total)}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow('frame', frame)
    out.write(frame)  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
