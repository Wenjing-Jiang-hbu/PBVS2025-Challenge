from ultralytics import YOLO
import csv
import torch
import torch.nn.functional as F
import os



if __name__ == '__main__':
    model = YOLO("weights/classifier.pt")
    results = model.predict(source='Unicorn_Dataset/test',batch=128)
    results_csv = 'results_classid.csv'
    ref = {
    'sedan':0,
    'SUV':1,
    'pickup_truck':2,
    'van':3,
    'box_truck':4,
    'motorcycle':5,
    'flatbed_truck':6,
    'bus':7,
    'pickup_truck_w_trailer':8,
    'semi_w_trailer':9
    }
    with open(results_csv, mode='w', newline='') as csv_file:
        fieldnames = ['image_id', 'class_id']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(results)):
            class_id = ref[results[i].names[results[i].probs.top1]]
            writer.writerow({'image_id': os.path.splitext(os.path.basename(results[i].path))[0][6:], 'class_id': class_id})
