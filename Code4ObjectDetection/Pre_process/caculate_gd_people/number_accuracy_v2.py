# Take the accuracy rate of each picture and take the average

import json
import datetime
from multiprocessing import Process



in_path = r'../../../data/caculate_gd_people/val.json'
prediction_path = r'../../../data/caculate_gd_people/bbox_my_val_4869_results.json'


def cal(annotation_file_path, predict_file_path):
    with open(annotation_file_path, 'r') as f, open(predict_file_path, 'r') as g:
        data_anno = json.load(f)
        data_pred = json.load(g)
        number_of_images = len(data_anno['images'])
        accuracy_list = []
        for i in range(number_of_images):
            a, b = 0, 0
            for j in range(len(data_anno['annotations'])):
                if data_anno['annotations'][j]['image_id'] == i:
                    a += 1
            for j in range(len(data_pred)):
                if data_pred[j]['image_id'] == i:
                    b += 1
            this_accuracy = 1 - abs(a - b) / a
            accuracy_list.append(this_accuracy)
        final_average_accuracy = \
            sum(accuracy_list) / len(accuracy_list) * 100
        print("Every accuracy = ", accuracy_list)
        print("Final average accuracy = {:.4f}%".format(final_average_accuracy))
        print("The length of final average accuracy is ", len(accuracy_list))
    return





if __name__ == '__main__':
    start_time = datetime.datetime.now()

    proc = Process(target=cal(in_path, prediction_path))
    proc.start()

    end_time = datetime.datetime.now()
    print("Runing time is", end_time - start_time)

