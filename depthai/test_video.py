#!/usr/bin/env python3

from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from time import monotonic
import os
import json
from utils import topk, mysoftmax

labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

img_folder = "sharks"

pipeline = dai.Pipeline()

# nodi
xinFrame = pipeline.create(dai.node.XLinkIn)
nn = pipeline.create(dai.node.NeuralNetwork)
nnOut = pipeline.create(dai.node.XLinkOut)

#propieta
xinFrame.setStreamName("inFrame")
xinFrame.setMaxDataSize(224*224*3)
nn.setBlobPath("densenet121.blob")
nnOut.setStreamName("nn")

#collegamenti
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

with dai.Device(pipeline) as device:

    qIn = device.getInputQueue(name="inFrame")
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    
    frame = None
    pred_class = ""
    
    
    def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()
    
    def displayFrame(name, frame, det_class):
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50) 
        fontScale = 1
        color = (255, 0, 0) 
        thickness = 2
        frame = cv2.putText(frame, det_class, org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow(name,frame)

    for i, img_name in enumerate(os.listdir(img_folder)):
        img_path = img_folder+"/"+img_name
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        img = dai.ImgFrame()
        img.setData(to_planar(frame, (224, 224)))
        img.setTimestamp(monotonic())
        img.setWidth(224)
        img.setHeight(224)
        qIn.send(img)
        inDet = qDet.tryGet()

        if inDet is not None:
            print(i)
            logits = inDet.getData()
            print(logits.shape)
            #preds = topk(logits, k=5).indices[1]

            break
        
        """

        preds = topk(logits, k=5).indices[1]

        for idx in preds:
            label = labels_map[idx]
            prob = mysoftmax(logits)[idx]
            print('{:<75} ({:.2f}%)'.format(label, prob*100)
        """
        
end=1

