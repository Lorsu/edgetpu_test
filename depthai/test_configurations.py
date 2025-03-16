import time
import cv2
import depthai as dai
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Benchmark with given version of efficientnet")
parser.add_argument("version", help="efficientnetb-x version")
args = parser.parse_args()
int_version = int(args.version)

efficientnet_resolutions = [224, 240, 288, 300, 380, 456]
res = efficientnet_resolutions[int_version]

pipeline = dai.Pipeline()

# nodes
xinFrame = pipeline.create(dai.node.XLinkIn)
nn = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

# properties
xinFrame.setStreamName("inFrame")
nn.setBlobPath(Path("models/efficientnets 6 shaves/efficientnet-b"+args.version+".blob"))
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

########################################################### warmup
cam = cv2.VideoCapture(f"../efficientnet/benchmark_{res}.mp4") #warmup
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
cam = cv2.VideoCapture(f"../efficientnet/benchmark_{res}.mp4")
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
        logits = np.array(inDet.getData()).view(np.float16)
        end_inference = time.perf_counter()
        inference_times[frame_id] = (end_inference - start_inference)*1000

    else:
        break
cam.release()

end_time = time.perf_counter()
elapsed_ms = (end_time - start_time) * 1000
print(f"time:{elapsed_ms / num_frames}ms")
print(f"average inference time:{np.mean(inference_times)}ms")
