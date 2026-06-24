"""
capture_frame.py

Extracts a frame from a traffic video for ROI calibration.

The saved image can be opened in any image editor
(Paint, Paint.NET, GIMP, etc.) to record pixel
coordinates (x, y) for ROI polygons.
"""

import cv2

cap = cv2.VideoCapture("videos/east.avi")

ret, frame = cap.read()

if ret:
    cv2.imwrite("captured_frames/east_frame.jpg", frame)

cap.release()