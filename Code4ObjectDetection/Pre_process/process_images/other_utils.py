import os
path = r"/home/lab-1/PycharmProjects/PrivateCode/postgraduate/object-detection/test-github/detectron/detectron/datasets/data/coco/my_test/my_val_img"
files = os.listdir(path)
for file in files:
    pwd_in = os.path.join(path, file)
    file_name_no_extension = file.split(".")[0]
    pwd_out = os.path.join(path, file_name_no_extension)
    command = "mv " + pwd_in + " " + pwd_out
    os.system(command)
    print("Processing: ", file)