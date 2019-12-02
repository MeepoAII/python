import xml.etree.ElementTree as ET
import os




def add_number(x, y):
    return str(int(x)+126), str(int(y)+263)

def change_coor(xml_in, xml_out):
    tree = ET.parse(xml_in)
    root = tree.getroot()

    for element in root.findall('object'):
        bndbox = element.find('bndbox')
        xmin = bndbox.find('xmin')
        ymin = bndbox.find('ymin')
        xmax = bndbox.find('xmax')
        ymax = bndbox.find('ymax')

        # change coordinate
        xmin.text, ymin.text = add_number(xmin.text, ymin.text)
        xmax.text, ymax.text = add_number(xmax.text, ymax.text)

        # change size

    size = root.find('size')
    width = size.find('width')
    height = size.find('height')
    width.text = str(1920)
    height.text = str(1080)


    tree.write(xml_out)
    return



in_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/xmls"
out_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/xmls_transfered"
files = os.listdir(in_path)
for file in files:
    xml_pwd_in = os.path.join(in_path, file)
    xml_pwd_out = os.path.join(out_path, file)
    change_coor(xml_pwd_in, xml_pwd_out)
    print("Changing the coordiante", file)


