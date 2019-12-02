# 将整个images，xmls文件分割train validate

import os, random

def sep(image_path, xml_path="", train_path="", validate_path="", ratio=0.7):
    filelist = os.listdir(image_path)
    random.shuffle(filelist)
    train_num = int(len(filelist)*ratio)

    for file in filelist[0:train_num]:
        # for train
        name = file.split(".")[0]
        image = name + ".jpg"
        src_path = os.path.join(image_path, image)
        dst_path = os.path.join(train_path, "images", image)
        cp_image_cmd = "cp " + src_path + " " + dst_path
        os.system(cp_image_cmd)

        xml = name + ".xml"
        src_path = os.path.join(xml_path, xml)
        dst_path = os.path.join(train_path, "xmls", xml)
        cp_xml_cmd = "cp " + src_path + " " + dst_path
        os.system(cp_xml_cmd)
        print("Processing train folder: ", name)

    for file in filelist[train_num:]:
        # for validate
        name = file.split(".")[0]
        image = name + ".jpg"
        src_path = os.path.join(image_path, image)
        dst_path = os.path.join(validate_path, "images", image)
        cp_image_cmd = "cp " + src_path + " " + dst_path
        os.system(cp_image_cmd)

        xml = name + ".xml"
        src_path = os.path.join(xml_path, xml)
        dst_path = os.path.join(validate_path, "xmls", xml)
        cp_xml_cmd = "cp " + src_path + " " + dst_path
        os.system(cp_xml_cmd)
        print("Processing validate folder: ", name)




    return









def main():
    image_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/image"
    xml_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/xml"
    train_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/train"
    validate_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/validate"

    sep(image_path, xml_path, train_path, validate_path)




    return

if __name__ == '__main__':
    main()