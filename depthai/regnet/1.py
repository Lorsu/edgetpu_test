import time
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse
import os

### 33.29  33.967 49.29 68.1 131    311             mem err         24.6            29.3                   38.80                50.8            92.4         176                500
##                              05_regnet_y_16_gf      06      07_regnet_x_400mf    08_regnet_x_800mf 09_regnet_x_1_6gf 10_regnet_x_3_2gf 11_regnet_x_8gf 12_regnet_x_16gf 13_regnet_x_32gf                                                         

parser = argparse.ArgumentParser(description="Benchmark with given version of regnet")
parser.add_argument("--model_id", type=int, help="0, 1, 2, 3 etc")
args = parser.parse_args()

if args.model_id>= 10:
    model_prefix = str(args.model_id)
else:
    model_prefix = "0"+str(args.model_id)


for model_name in os.listdir("./export/"):
    if model_name.find(model_prefix) != -1:
        model_path = "./export/"+model_name

print("model chosen:", model_path)

pipeline = dai.Pipeline()

# nodes
xinFrame = pipeline.create(dai.node.XLinkIn)
nn = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

# properties
xinFrame.setStreamName("inFrame")
nn.setBlobPath(Path(model_path))
nnOut.setStreamName("nn")

# links
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

device = dai.Device(pipeline)
qIn = device.getInputQueue(name="inFrame")
qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

fps = 60
duration_sec = 10
num_frames = fps * duration_sec
img = dai.ImgFrame()

res = 224
########################################################### warmup
cam = cv2.VideoCapture("benchmark_224.mp4") #warmup
for _ in range(30):
    _, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (res,res)).transpose(2, 0, 1)
    frame = frame.flatten()
    img.setData(frame)
    img.setWidth(res)
    img.setHeight(res)
    qIn.send(img)
    inDet = qDet.get()
cam.release()

################################## benchmark
cam = cv2.VideoCapture("benchmark_224.mp4")
inference_times = np.zeros((num_frames,))

start_time = time.perf_counter()
for frame_id in range(num_frames):
    ret, frame = cam.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (res,res)).transpose(2, 0, 1)
        frame = frame.flatten()
        img.setData(frame)
        img.setWidth(res)
        img.setHeight(res)
        start_inference = time.perf_counter()
        qIn.send(img)
        inDet = qDet.get()
        end_inference = time.perf_counter()
        logits = np.array(inDet.getData()).view(np.float16)
        inference_times[frame_id] = (end_inference - start_inference)*1000

    else:
        break
cam.release()

end_time = time.perf_counter()
elapsed_ms = (end_time - start_time) * 1000
print(f"time:{elapsed_ms / num_frames}ms")
print(f"average inference time:{np.mean(inference_times)}ms")
