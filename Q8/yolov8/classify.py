from ultralytics import YOLO
 
# Load a model
model = YOLO("yolov8x-cls.pt")  # load a pretrained model
 
# Use the model to detect object
model.predict(source="./grocerystore/", project='./ultralytics', name='classifications', save=True)