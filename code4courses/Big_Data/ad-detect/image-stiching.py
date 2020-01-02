import cv2
import os
import glob
import numpy as np

f = r'/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/images/raw'
save_path = r'/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/meepo/data/bigdata/my-experiment/images/stitch'

for file in os.listdir(f):
    folder = os.path.join(f, file)
    print(folder)
    path = folder + "/" + "0.jpg"
    img_out = cv2.imread(path)

    for i in range(1, 10):
        path = folder + "/" + str(i) + ".jpg"
        img_tmp = cv2.imread(path)

        img_out = np.concatenate((img_out, img_tmp), axis=1)

    for i in range(1, 8):
        img_first_path = folder + "/" + str(i*10) + ".jpg"
        img_first = cv2.imread(img_first_path)
        if i != 7:
            for j in range(1, 10):
                file_path = folder + "/" + str(i*10+j) + ".jpg"
                img_tmp = cv2.imread(file_path)
                img_first = np.concatenate((img_tmp, img_first), axis=1)
        else:
            for j in range(1, 9):
                file_path = folder + "/" + str(i*10+j) + ".jpg"
                img_tmp = cv2.imread(file_path)
                img_first = np.concatenate((img_tmp, img_first), axis=1)
            temp = np.zeros([95, 79, 3], np.uint8)
            img_first = np.concatenate((img_first, temp), axis=1)

        img_out = np.concatenate((img_out, img_first), axis=0)

    # cv2.imshow("IMG", img_out)
    save = os.path.join(save_path, file) + ".jpg"
    cv2.imwrite(save, img_out)
    cv2.waitKey(0)