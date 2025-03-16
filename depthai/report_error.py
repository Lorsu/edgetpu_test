import cv2
import depthai as dai
import numpy as np
from pathlib import Path
from utils import topk, mysoftmax
import json
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

DO_NORM = False
pipeline = dai.Pipeline()

# nodes
xinFrame = pipeline.create(dai.node.XLinkIn)
nn = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

# properties
xinFrame.setStreamName("inFrame")
nn.setBlobPath(Path("resnet50_norm.blob"))
nnOut.setStreamName("nn")

# links
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

with dai.Device(pipeline) as device:
    qIn = device.getInputQueue(name="inFrame")
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
    
    frame = cv2.imread("shark4.png")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224,224)).transpose(2, 0, 1)
    if DO_NORM:
        frame = frame / 225
        frame[0] = (frame[0] - 0.485) / 0.229
        frame[1] = (frame[1] - 0.456) / 0.224
        frame[2] = (frame[2] - 0.406) / 0.225
    frame = frame.flatten()

    img = dai.ImgFrame()
    img.setData(frame)
    img.setWidth(224)
    img.setHeight(224)
    qIn.send(img)
    
    inDet = qDet.get()
    logits = np.array(inDet.getData()).view(np.float16)
    logits = logits.reshape((1,1000))

    preds = topk(logits, k=5).indices[1]
    for idx in preds:
        label = labels_map[idx]
        prob = mysoftmax(logits.flatten())[idx]
        print('{:<75} ({:.2f}%)'.format(label, prob*100))
