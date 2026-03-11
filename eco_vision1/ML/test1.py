from ultralytics import YOLO
model=YOLO("best (2).pt")
result=model.predict("/Users/dhiya/Downloads/eco 6/p1.jpeg")
result[0].show()