import os
import pandas as pd
import numpy as np
import os

images_path = '/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/images/stitch'
des_path = '/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/try-1/data'

df = pd.read_csv('/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/images/train_pre_label.csv')

print(type(df['label'][3]))

# for train data
for i in range(220):
    label = df['label'][i]
    filename = 'instance' + str(i) + '.jpg'
    source_pwd = os.path.join(images_path, filename)
    des_folder = str(label)
    destination_pwd = os.path.join(des_path, 'train', des_folder, filename)
    command = "cp" + " " + source_pwd + " " + destination_pwd
    os.system(command)
    print("Process: {}".format(i))


# for val data
for i in range(220, 270):
    label = df['label'][i]
    filename = 'instance' + str(i) + '.jpg'
    source_pwd = os.path.join(images_path, filename)
    des_folder = str(label)
    destination_pwd = os.path.join(des_path, 'val', des_folder, filename)
    command = "cp" + " " + source_pwd + " " + destination_pwd
    os.system(command)
    print("Process: {}".format(i))

# for test data
for i in range(270, 300):
    label = df['label'][i]
    filename = 'instance' + str(i) + '.jpg'
    source_pwd = os.path.join(images_path, filename)
    des_folder = str(label)
    destination_pwd = os.path.join(des_path, 'test', des_folder, filename)
    command = "cp" + " " + source_pwd + " " + destination_pwd
    os.system(command)
    print("Process: {}".format(i))
