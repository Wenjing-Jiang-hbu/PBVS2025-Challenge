from ultralytics import YOLO
import csv
import torch
import torch.nn.functional as F
import os



if __name__ == '__main__':
    model = YOLO("weights/scorer.pt")
    results = model.predict(source='Unicorn_Dataset/test',batch=128)
    results_csv = 'results_score.csv'
    ref = {
    'id':0,
    'od':1,}
    with open(results_csv, mode='w', newline='') as csv_file:
        fieldnames = ['image_id', 'score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(results)):
            class_id = ref[results[i].names[results[i].probs.top1]]
            if class_id == 0:
                writer.writerow({'image_id': os.path.splitext(os.path.basename(results[i].path))[0][6:], 'score': results[i].probs.top1conf.item()})
            elif class_id == 1:
                writer.writerow({'image_id': os.path.splitext(os.path.basename(results[i].path))[0][6:], 'score': 1-results[i].probs.top1conf.item()})

