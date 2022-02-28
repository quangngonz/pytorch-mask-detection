import torch
import numpy as np
import cv2, json

model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

image_list = ["data/valid/images/maksssksksss832_png.rf.499264dc627921e0b05ef602c0db62ae.jpg", 'data/valid/images/maksssksksss843_png.rf.06d4da26a112f2a8dbb7c3dba252a016.jpg', 'data/valid/images/maksssksksss809_png.rf.93ee79b14e28a2b7220ad834fd0a8e37.jpg']

def predict(image_path):
    image = cv2.imread(image_path)

    predictions = model(image)

    results_json = json.loads(predictions.pandas().xyxy[0].to_json(orient='records'))

    for i in results_json:
        print("xmin", i['xmin'])
        print("ymin", i['ymin'])
        print("xmax", i['xmax'])
        print("ymax", i['ymax'])
        print("confidence", i['confidence'])
        print("class", i['name'])
        print("----------------------------------")

    cv2.imshow('Predictions', np.squeeze(predictions.render()))
    cv2.waitKey(0)

for i in image_list:
    predict(i)

