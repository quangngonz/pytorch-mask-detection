import torch
import numpy as np
import cv2, json

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/best.pt', force_reload=True)

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt', force_reload=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    # Make detections 
    results = model(frame)


    # https://stackoverflow.com/questions/70523588/how-do-i-retrieve-the-resultant-image-as-a-matrixnumpy-array-from-results-give
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient='records')) # im predictions to json 
    
    for i in results_json:
        print("xmin", i['xmin'])
        print("ymin", i['ymin'])
        print("xmax", i['xmax'])
        print("ymax", i['ymax'])
        print("confidence", i['confidence'])
        print("class", i['class'])
        print("----------------------------------")
    # https://github.com/ultralytics/yolov5/issues/2703

    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
