# 차선 인식을 openVINO 를 통해서 진행해보자 

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import os
from pathlib import Path

base_model_dir = Path("model").expanduser()

model_name = "road-segmentation-adas-0001"
model_xml_name = f'{model_name}.xml'
model_bin_name = f'{model_name}.bin'

model_xml_path = base_model_dir / model_xml_name
model_bin_path = base_model_dir / model_bin_name
import requests

def download_file(url, destination):
    response = requests.get(url)
    if response.status_code == 200:
        with open(destination, 'wb') as file:
            file.write(response.content)
            print("XML 다운완료")
    else:
        print("failed to dwonload file")

def segmentation_map_to_image(segmentation_map, colormap):
    
    print(segmentation_map.shape)
    
    height,width,_ = segmentation_map.shape
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            label = segmentation_map[y,x]
            color = colormap[label]
            output_image[y,x] = color
    return output_image
    
    
# 사진을 넣자
if not model_xml_path.exists():
    model_xml_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml"
    model_bin_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.bin"

    download_file(model_xml_url,model_xml_path)
    download_file(model_bin_url,model_bin_path)
else:
    print("이미 다운로드 되어있다.")
    



core = ov.Core()

model = core.read_model(model=model_xml_path)
compiled_model = core.compile_model(model=model, device_name='AUTO')

input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output(0)

# 비디오 파일 읽기

video_file = "./model/test.mkv"
cap = cv2.VideoCapture(video_file)

# 비디오의 높이와 너비 가져오기
image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

N, C, H, W = input_layer_ir.shape


if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (W, H))
            input_image = np.expand_dims(
            resized_image.transpose(2, 0, 1), 0
            )
            result = compiled_model([input_image])[output_layer_ir]
            segmentation_mask = np.argmax(result, axis=1)

            # Define colormap, each color represents a class.
            colormap = np.array([[0, 0, 255], [48, 103, 141], [53, 183, 120], [199, 216, 52]])

            # Define the transparency of the segmentation mask on the photo.
            alpha = 0.3

            # Use function from notebook_utils.py to transform mask to an RGB image.
            mask = segmentation_map_to_image(segmentation_mask, colormap)
            resized_mask = cv2.resize(mask, (image_w, image_h))

            # Create an image with mask.
            image_with_mask = cv2.addWeighted(resized_mask, alpha, rgb_image, 1 - alpha, 0)
            cv2.waitKey(25)
        else:
            break
else:
    print("can't open video.")
cap.release()
cv2.destoryAllWindows()








     
    
