#import cv2
import depthai as dai
import numpy as np
from pathlib import Path

#nodes
pipeline = dai.Pipeline()
camRgb = pipeline.create(dai.node.ColorCamera)
nnOut = pipeline.create(dai.node.XLinkOut)
nn = pipeline.create(dai.node.NeuralNetwork)

#properties
nn.setBlobPath(Path("models/different shaves/efficientnet-b0_shave6.blob"))
nnOut.setStreamName("nn")
camRgb.setPreviewSize(224, 224)
camRgb.setInterleaved(False) ##false for channelfirst of torch
nn.input.setBlocking(False)

#links
camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)

with dai.Device(pipeline) as device:
    while True:
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        inDet = qDet.get()
        output = np.array(inDet.getData()).view(np.float16)
        print(output.shape)
