import h5py
import scipy.misc
import imageio
import cv2
import os


# print([key for key in f.keys()])
# print(f['data'][:].shape)
#
# print(f['data'][0][0][0][0])

save_path = r'/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/images'

with h5py.File('/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/train/train_pre_data.h5', 'r') as file:
    images = file['data'][:]

    for i in range(len(images)):
        folder_pwd = save_path + "/instance" + str(i)
        print(folder_pwd)
        os.mkdir(folder_pwd)
        for j in range(len(images[i][0])):
            img_pwd = folder_pwd + "/" + str(j) + ".jpg"
            print(img_pwd)
            cv2.imwrite(img_pwd, images[i][0][j])
