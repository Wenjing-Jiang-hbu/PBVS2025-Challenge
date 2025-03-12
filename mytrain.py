from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/11/yolo11s-cls.yaml")
model.load("yolo11s-cls.pt")

# Train the model
results = model.train(data="PBVS/ODdataset/", epochs=50, imgsz=128,batch=64,workers=2)
