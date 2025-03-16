import cv2

cam = cv2.VideoCapture("../../detection/det_bench_416.mp4")
_, frame = cam.read()
print(type(frame))