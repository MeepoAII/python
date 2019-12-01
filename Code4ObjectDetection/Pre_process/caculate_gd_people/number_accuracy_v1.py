# Accuracy of total people (prediction people / annotation)



import json
import os

in_path = r'./data/val.json'

out_path = r'./data/output.txt'
prediction_path = r'./data/bbox_my_val_results.json'

global a, b
a = 0
b = 0

with open(in_path, 'r') as f, open(out_path, 'w') as out:
    data = json.load(f)
    a = len(data['annotations'])
    print("Ground Truth: Total number of people is ", a)

with open(prediction_path, 'r') as f:
    data = json.load(f)
    b = len(data)
    print("Prediction: Total number of people is ", b)

print("The accuracy of the number of people is ", "{:.2f}".format((1 - abs(a-b) / a) * 100), "%", sep="")
print("test")
