# 以裁剪的文件名为索引去寻找原图片并保存


import os


in_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/xml"
out_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/image"
reference_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/image/image-cut-per-8-seconds"


files = os.listdir(in_path)
for file in files:
    # new_image_path = os.path.join(out_path, file.split(".")[0]+".jpg")
    name = file.split(".")[0]+".jpg"
    result_path = os.path.join(out_path, name)
    cmd = "find " + reference_path + " " + "-name " + name
    full_image_path = os.popen(cmd).read()
    new_full_image_path = full_image_path[0:-1]    # to delete \n added by os.popen

    cmd_cp = "cp " + new_full_image_path + " " + result_path
    os.system(cmd_cp)
    print("Processing: ", file)


