import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import time
import numpy as np

#folder_path = "/home/models/v14"
#model_file  = folder_path + "/ultralytics/" + "yolo11s-seg_320_edgetpu.tflite"
model_file = "/home/models/yolo8/14/yolov8m_416_edgetpu.tflite"


interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()
size = common.input_size(interpreter)
print("detected input size:", size)


BENCHMARK_SAMPLES_NO = 600
inference_times = np.zeros((BENCHMARK_SAMPLES_NO,))

for i in range(BENCHMARK_SAMPLES_NO):
	image_input = np.random.randint(low=0, high=200, size=1*3*size[0]*size[1], dtype=np.uint8).reshape(size[0],size[1],3)
	
	start_time = time.time()
	common.set_input(interpreter, image_input)
	interpreter.invoke()
	inference_times[i] = time.time() - start_time
	
print("tempo medio:", np.average(inference_times))
