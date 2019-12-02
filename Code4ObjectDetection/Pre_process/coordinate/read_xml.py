import xml.etree.ElementTree as ET

tree = ET.parse("./test_data/2019-03-20-08-03-09.xml")
root = tree.getroot()

for element in root.findall('object'):
    bndbox = element.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin_value = xmin.text
    new_value = int(xmin_value) + 1
    xmin.text = str(new_value)
    print(xmin.text)

tree.write('./test_data/output.xml')

