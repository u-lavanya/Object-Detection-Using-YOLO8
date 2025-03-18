absolutepath_of_directory_with_xmlfiles = r'/Users/lava/Downloads/Person and PPE detection/DAt/labels'
absolutepath_of_directory_with_imgfiles = r'/Users/lava/Downloads/Person and PPE detection/DAt/images'
absolutepath_of_directory_with_yolofiles = r'/Users/lava/Downloads/Person and PPE detection/DAt/Converted_Labels'
absolutepath_of_directory_with_classes_txt = r'/Users/lava/Downloads/Person and PPE detection/DAt/classes.txt'
absolutepath_of_directory_with_error_txt = r'/Users/lava/Downloads/Person and PPE detection/error_logs/'

##############################################################################################################################################################################################################################################################################


import os
import cv2
import logging
from lxml import etree

from xml.etree import ElementTree
from glob import glob


class GetDataFromXMLfile:
    def __init__(self, xmlfile_path):
        self.xmlfile_path = xmlfile_path
        self.xmlfile_datalists_list = []

    def get_datalists_list(self):
        self.parse_xmlfile()
        return self.xmlfile_datalists_list

    def parse_xmlfile(self):
        lxml_parser = etree.XMLParser(encoding='utf-8')
        xmltree = ElementTree.parse(self.xmlfile_path, parser=lxml_parser).getroot()

        for object in xmltree.findall('object'):
            xmlfile_datalist = []
            class_name = object.find('name').text
            xmlfile_datalist.append(class_name)
            bndbox = object.find("bndbox")
            xmlfile_datalist.append(bndbox)
            self.xmlfile_datalists_list.append(xmlfile_datalist)

        img_filename = xmltree.find('filename').text
        
        self.add_data_to_datalist(img_filename)
    
    def add_data_to_datalist(self, img_filename):
        for xmlfile_datalist in self.xmlfile_datalists_list:
            xmin = float(xmlfile_datalist[1].find('xmin').text)
            ymin = float(xmlfile_datalist[1].find('ymin').text)
            xmax = float(xmlfile_datalist[1].find('xmax').text)
            ymax = float(xmlfile_datalist[1].find('ymax').text)
            bndbox_coordinates_list = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
            xmlfile_datalist[1] = bndbox_coordinates_list
        self.xmlfile_datalists_list.append(img_filename)
        self.xmlfile_datalists_list.append(self.xmlfile_path)


class CreateYOLOfile:
    def __init__(self, xmlfile_datalists_list, classes_list):
        self.xmlfile_datalists_list = xmlfile_datalists_list
        self.xmlfile_path = self.xmlfile_datalists_list.pop()
        self.img_filename = self.xmlfile_datalists_list.pop()
        self.yolofile_path = absolutepath_of_directory_with_yolofiles + os.path.basename(self.xmlfile_path).split('.', 1)[0] + '.txt'
        self.classes_list = classes_list
        try:
            (self.img_height, self.img_width, _) = cv2.imread(absolutepath_of_directory_with_imgfiles + self.img_filename).shape
            self.create_yolofile()
        except:
            with open(absolutepath_of_directory_with_error_txt+'xmlfiles_with_no_paired.txt', 'a') as f:
                f.write(os.path.basename(self.xmlfile_path)+'\n')

    def create_yolofile(self):
        for xmlfile_datalist in self.xmlfile_datalists_list:
            yolo_datalist = self.convert_xml_to_yolo_format(xmlfile_datalist)

            logging.debug(f"Writing to YOLO file: {self.yolofile_path}")
            with open(self.yolofile_path, 'a') as f:
                f.write("%d %.06f %.06f %.06f %.06f\n" % (yolo_datalist[0], yolo_datalist[1], yolo_datalist[2], yolo_datalist[3], yolo_datalist[4]))

    def convert_xml_to_yolo_format(self, xmlfile_datalist):
        class_name = xmlfile_datalist[0]
        self.add_class_to_classeslist(class_name)
        bndbox_coordinates_list = xmlfile_datalist[1]
        coordinates_min = bndbox_coordinates_list[0]
        coordinates_max = bndbox_coordinates_list[2]

        class_id = self.classes_list.index(class_name)
        yolo_xcen = float((coordinates_min[0] + coordinates_max[0])) / 2 / self.img_width
        yolo_ycen = float((coordinates_min[1] + coordinates_max[1])) / 2 / self.img_height
        yolo_width = float((coordinates_max[0] - coordinates_min[0])) / self.img_width
        yolo_height = float((coordinates_max[1] - coordinates_min[1])) / self.img_height
        yolo_datalist = [class_id, yolo_xcen, yolo_ycen, yolo_width, yolo_height]

        return yolo_datalist
    
    def add_class_to_classeslist(self, class_name):
        if class_name not in self.classes_list:
            self.classes_list.append(class_name)


class CreateClasssesfile:
    def __init__(self, classes_list):
        self.classes_list = classes_list

    def create_classestxt(self):
        with open(absolutepath_of_directory_with_classes_txt + 'classes.txt', 'w') as f:
            for class_name in self.classes_list:
                f.write(class_name+'\n')


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug("Logging is set up.")

def main():
    setup_logging()

    xmlfiles_pathlist = glob(absolutepath_of_directory_with_xmlfiles + "/*.xml")
    logging.debug(f"Found XML files: {xmlfiles_pathlist}")
    classes_list = []
    if not os.path.exists(absolutepath_of_directory_with_error_txt):
        os.makedirs(absolutepath_of_directory_with_error_txt)
if not os.path.exists(absolutepath_of_directory_with_error_txt):
    os.makedirs(absolutepath_of_directory_with_error_txt)
    if not os.path.exists(absolutepath_of_directory_with_error_txt):
        os.makedirs(absolutepath_of_directory_with_error_txt)

    for xmlfile_path in xmlfiles_pathlist:
        logging.debug(f"Processing XML file: {xmlfile_path}")
        process_xmlfile = GetDataFromXMLfile(xmlfile_path)
        xmlfile_datalists_list = process_xmlfile.get_datalists_list()
        CreateYOLOfile(xmlfile_datalists_list, classes_list)

    process_classesfile = CreateClasssesfile(classes_list)
    process_classesfile.create_classestxt()
    
    
if __name__ == '__main__':
    main()
