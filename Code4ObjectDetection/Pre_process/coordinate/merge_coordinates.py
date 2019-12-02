# Combine the coordinates of the upper and lower halves
# test ok
import xml.etree.ElementTree as ET
import os

upper_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/only_upper"
lower_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/xmls"
output_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/temp_results"
#
# tree = ET.parse(upper_path)
# root = tree.getroot()
#
# element_list = []
#
# for element in root.findall('object'):
#     # print(element.tag, element.attrib, element.text, sep="\n")
#     element_list.append(element)
#
#
# root.append(element_list[0])
# root.append(element_list[1])
#
#
# for element in root.findall('object'):
#     print(element)
#
#
# tree.write("output.xml")

def merge(upper_path, lower_path, output_path):
    treeU = ET.parse(upper_path)
    treeL = ET.parse(lower_path)
    rootU = treeU.getroot()
    rootL = treeL.getroot()

    for element in rootU.findall('object'):
        rootL.append(element)

    treeL.write(output_path)

    return


# merge(upper_path, lower_path, output_path)

for file in os.listdir(upper_path):
    upperxml = os.path.join(upper_path, file)
    lowerxml = os.path.join(lower_path, file)
    result_path = os.path.join(output_path, file)
    merge(upperxml, lowerxml, result_path)
    print("Processing: ", file)



