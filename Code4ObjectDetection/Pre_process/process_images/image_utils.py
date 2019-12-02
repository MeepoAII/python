# -*- coding: utf-8 -*


# conda activate detectron
# python version = 2.7 ????

import cv2
import os
from os.path import *
import datetime
import math, time
from math import *
import random

import numpy as np
np.set_printoptions(threshold=np.inf)


def rename(input_path, output_path):
    files = os.listdir(input_path)
    i = 1
    for file in files:
        os.rename(input_path + "//" + file, input_path + "//" + str(i) + "gray_image.jpg")
        i = i + 1



def cut_photo(input_path, output_path):
    files = os.listdir(input_path)
    i = 1
    x, y, w, h = 1, 1, 1, 1
    for file in files:
        pwd_in = os.path.join(input_path, file)
        pwd_out = os.path.join(output_path, file)
        image = cv2.imread(pwd_in)
        im2 = cv2.resize(image, (1920,1080), interpolation=cv2.INTER_CUBIC)
        if i == 1:
            rect = cv2.selectROI("testcut", im2, True, False)
            (x, y, w, h) = rect
        x, y, w, h = 126, 263, 1609, 523
        cropImg = im2[y : y+h, x:x+w]
        cv2.imwrite(pwd_out, cropImg)
        cv2.waitKey(1)
        print(i)
        print("x, y, w, h", x, y, w, h)
        i = i+1

    return


def cut_photo_two_recurve(input_path, output_path):
    files = os.listdir(input_path)
    i = 1
    x, y, w, h = 1, 1, 1, 1
    for file in files:
        in_folder_path = os.path.join(input_path, file)
        out_folder_path = os.path.join(output_path, file)
        if os.path.exists(out_folder_path):
            pass
        else:
            os.makedirs(out_folder_path)
            for img in os.listdir(in_folder_path):
                pwd_in = os.path.join(in_folder_path, img)
                pwd_out = os.path.join(out_folder_path, img)
                image = cv2.imread(pwd_in)
                im2 = cv2.resize(image, (1920,1080), interpolation=cv2.INTER_CUBIC)
                if i == 1:
                    rect = cv2.selectROI("testcut", im2, True, False)
                (x, y, w, h) = rect
                x, y, w, h = 126, 263, 1609, 523
                cropImg = im2[y:y+h, x:x+w]
                cv2.imwrite(pwd_out, cropImg)
                cv2.waitKey(1)
                print(i)
                print("x, y, w, h", x, y, w, h)
                i = i+1

    return





def re(input_path, output_path):
    files = os.listdir(input_path)
    c = 1
    for file in files:
        pwd = input_path+"//"+file
        temp = cv2.UMat(cv2.imread(pwd, cv2.IMREAD_GRAYSCALE))
        temp_1 = cv2.bitwise_not(temp)
        pwd_out = output_path + "//" + file
        cv2.imwrite(pwd_out, temp_1)
        cv2.waitKey(1)
        print("calculating %d photo" % c)
        c += 1

    return

def rename_images(in_path, seconds):
    # Rename the images in 2019-03-20-12-34-55 format
    files = os.listdir(in_path)
    files.sort(key=lambda x:int(x[0:-4]))
    filename = basename(normpath(in_path))
    print(filename)
    time_list = filename.split("-")
    time_list = list(map(int, time_list))
    start_stamp = datetime.datetime(time_list[0], time_list[1], time_list[2],
                                    time_list[3], time_list[4], time_list[5])

    this_stamp = start_stamp

    print(start_stamp.strftime("%Y-%m-%d-%H-%M-%S"))

    for file in files:
        new_time = this_stamp.strftime("%Y-%m-%d-%H-%M-%S")+".jpg"
        print("Rename to ", new_time)
        old_img_pwd = join(in_path, file)
        new_img_pwd = join(in_path, new_time)
        os.rename(old_img_pwd, new_img_pwd)
        delta = datetime.timedelta(seconds=seconds)
        this_stamp = this_stamp+delta


    return


class CropImageUsingCoordinates:
    def rotate(self, img, pt1, pt2, pt3, pt4, out_path):
        # print(pt1, pt2, pt3, pt4)
        withRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
        heightRect = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        # print(withRect, heightRect)
        angle = acos((pt4[0] - pt1[0]) / withRect) * (180 / math.pi)  # 矩形框旋转角度
        # print(angle)

        if pt4[1] > pt1[1]:
            pass
            # print("顺时针旋转")
        else:
            # print("逆时针旋转")
            angle = -angle

        height = img.shape[0]
        width = img.shape[1]
        rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
        heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
        widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

        rotateMat[0, 2] += (widthNew - width) / 2
        rotateMat[1, 2] += (heightNew - height) / 2
        imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew), borderValue=(255, 255, 255))

        # 旋转后图像的四点坐标
        [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
        [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
        [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

        # 处理反转的情况
        if pt2[1] > pt4[1]:
            pt2[1], pt4[1] = pt4[1], pt2[1]
        if pt1[0] > pt3[0]:
            pt1[0], pt3[0] = pt3[0], pt1[0]

        imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
        cv2.imwrite(out_path, imgOut)  # 裁减得到的旋转矩形框
        return imgRotation  # rotated image


    def Start(self, coor_file_pwd, in_path, out_path):

        # Read the coordinates first
        # file format:
        # 245,846,96,725,485,249,633,370
        # Every two coordinates representing a point

        with open(coor_file_pwd) as file:
            lines = file.readlines()
            lines = lines[0]
            items = lines.split(",")
            pt1 = [int(items[0]), int(items[1])]
            pt4 = [int(items[2]), int(items[3])]
            pt3 = [int(items[4]), int(items[5])]
            pt2 = [int(items[6]), int(items[7])]

        img = cv2.imread(in_path)
        self.rotate(img, pt1, pt2, pt3, pt4, out_path)




class CropImage_Using_MinAreaRect:
    '''
    Crop the image with the minimum bounding rectangle, enter the point coordinates of the polygon.
    '''
    def __init__(self, points_array, in_path, out_path="./"):
        self.points_array = points_array
        self.in_path = in_path
        self.out_path = out_path

    def cut(self):
        # haven't cut yet
        rect = cv2.minAreaRect(self.points_array)
        print("About the rectangle:\n", rect)
        rect_coordinates = cv2.boxPoints(rect)
        print("The result rectangle information:\n", rect_coordinates)
        img = cv2.imread(self.in_path)
        # cv2.imshow("test_original", img)
        # cv2.waitKey(0)
        cv2.polylines(img, np.int32([rect_coordinates]), True, (0,0,255), thickness=2)
        cv2.imshow("test", img)
        cv2.waitKey(0)


class OtherUtilities:
    def rotate(self, in_path, out_path, angle, scale):
        img_original = cv2.imread(in_path)
        h, w = img_original.shape[:2]
        center = (w//2, h//2)
        M_1 = cv2.getRotationMatrix2D(center, angle, scale)
        img_rotate = cv2.warpAffine(img_original, M_1, (w, h))
        cv2.imwrite(out_path, img_rotate)
        return

    def adjust_image_sequence(self, in_path, background_path):
        # Add background to image sequence
        files = os.listdir(in_path)
        length = len(files)
        i = 0
        for file in files:
            src = background_path
            background_image_name = str(i)+".jpg"
            dst = os.path.join(in_path, background_image_name)
            cmd = 'copy "%s" "%s"' % (src, dst)
            os.system(cmd)

            in_pwd = os.path.join(in_path, file)
            new_file_name = str(i+1)+".jpg"
            out_pwd = os.path.join(in_path, new_file_name)
            os.rename(in_pwd, out_pwd)

            i += 2

            print("Processing: ", file)

        return

    def crop_image_margin(self, in_path, out_path):
        img = cv2.imread(in_path)
        new_img = img[1:613, 1:189]
        cv2.imwrite(out_path, new_img)
        return


    def filter(self, image_in_path, xml_out_path, image_out_path):
        '''
        Filter out useless images(ie no corresponding xml file, annotation file), and
        copy result to destination folder respectively.
        :param image_in_path: folder path
        :param xml_out_path:
        :param image_out_path:
        :return:
        '''
        files = os.listdir(image_in_path)
        files = [item.split(".")[0] for item in files]
        files = list(dict.fromkeys(files))

        for file in files:
            xml_full_name = os.path.join(image_in_path, file) + ".xml"
            image_full_name = os.path.join(image_in_path, file) + ".jpg"
            flag1 = os.path.exists(xml_full_name)
            flag2 = os.path.exists(image_full_name)
            if flag1 and flag2:
                print("Copying file:", file)
                # copy the xml and image file to destination
                commandxml = "cp " + xml_full_name + " " + os.path.join(xml_out_path, file)+".xml"
                commandjpg = "cp " + image_full_name + " " + os.path.join(image_out_path, file)+".jpg"
                os.system(commandxml)
                os.system(commandjpg)



    def renamefile(self, in_path, out_path):
        '''
        change file name to number to satisfy voc2coco.py
        :param in_path:
        :return:
        '''
        i = 0
        files = os.listdir(in_path)
        files.sort()
        final_out = []
        for item in files:
            str = "%06d" % i + ".jpg"
            i += 1
            final_out.append(str)

        for a, b in zip(files, final_out):
            pwd1 = os.path.join(in_path, a)
            pwd2 = os.path.join(out_path, b)
            command = "mv " + pwd1 + " " + pwd2
            print("Processing: ", a, b)
            os.system(command)


        return


    def get_hard_images(self, in_path, out_path):
        files = os.listdir(in_path)
        files = [item.split(".")[0] for item in files]
        files = list(dict.fromkeys(files))

        for file in files:
            xml_full_name = os.path.join(in_path, file) + ".xml"
            image_full_name = os.path.join(in_path, file) + ".jpg"
            flag1 = os.path.exists(xml_full_name)
            flag2 = os.path.exists(image_full_name)
            if flag1 is False and flag2 is True:
                # this file is not labeled i.e. hard images, cp it
                filename = file+".jpg"
                pwd1 = os.path.join(in_path, filename)
                pwd2 = os.path.join(out_path, filename)
                command = "cp " + pwd1 + " " + pwd2
                print("execute", command)
                os.system(command)



        return


    def delete_suffix(self, in_path):
        files = os.listdir(in_path)
        files.sort()
        for file in files:
            pwd_in = os.path.join(in_path, file)
            file_name_no_extension = file.split(".")[0]
            pwd_out = os.path.join(in_path, file_name_no_extension)
            command = "mv " + pwd_in + " " + pwd_out
            os.system(command)
            print("Processing: ", file)



    def merge_image(self, image_1, image_2, out_path):
        pre = cv2.imread(image_1)
        later = cv2.imread(image_2)
        cut_area = pre[0:283, 0:1920]
        later[0:283, 0:1920] = cut_area
        # cv2.imshow("result", later)
        cv2.imwrite(out_path, later)
        cv2.waitKey(0)

        return



    def get_labeled_img(self, xml_path, image_path, out_path):
        # find image according by xml, and copy to out_path
        for name in os.listdir(xml_path):
            name = name.split(".")[0]




        return










class BackgroundSubtraction:

    def absdiff(self, image_1, image_2, sThre):
        gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
        gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 3), 0)
        gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
        d_frame = cv2.absdiff(gray_image_1, gray_image_2)
        ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)
        return d_frame


class ViBe:
    '''
    classdocs
    '''
    __defaultNbSamples = 20  # 每个像素点的样本个数
    __defaultReqMatches = 2  # min指数
    __defaultRadius = 20;  # Sqthere半径
    __defaultSubsamplingFactor = 16  # 子采样概率
    __BG = 0  # 背景像素
    __FG = 255  # 前景像素
    __c_xoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0]  # x的邻居点 len=9
    __c_yoff = [-1, 0, 1, -1, 1, -1, 0, 1, 0]  # y的邻居点 len=9

    __samples = []  # 保存每个像素点的样本值,len defaultNbSamples+1
    __Height = 0
    __Width = 0

    def __init__(self, grayFrame):
        '''
        Constructor
        '''
        self.__Height = grayFrame.shape[0]
        self.__Width = grayFrame.shape[1]

        for i in range(self.__defaultNbSamples + 1):
            self.__samples.insert(i, np.zeros((grayFrame.shape[0], grayFrame.shape[1]), dtype=grayFrame.dtype));

        self.__init_params(grayFrame)

    def __init_params(self, grayFrame):
        # 记录随机生成的 行(r) 和 列(c)
        rand = 0
        r = 0
        c = 0

        # 对每个像素样本进行初始化
        for y in range(self.__Height):
            for x in range(self.__Width):
                for k in range(self.__defaultNbSamples):
                    # 随机获取像素样本值
                    rand = random.randint(0, 8)
                    r = y + self.__c_yoff[rand]
                    if r < 0:
                        r = 0
                    if r >= self.__Height:
                        r = self.__Height - 1  # 行
                    c = x + self.__c_xoff[rand]
                    if c < 0:
                        c = 0
                    if c >= self.__Width:
                        c = self.__Width - 1  # 列
                    # 存储像素样本值
                    self.__samples[k][y, x] = grayFrame[r, c]
            self.__samples[self.__defaultNbSamples][y, x] = 0

    def update(self, grayFrame):
        foreground = np.zeros((self.__Height, self.__Width), dtype=np.uint8)
        for y in range(self.__Height):  # Height
            for x in range(self.__Width):  # Width
                # 用于判断一个点是否是背景点,index记录已比较的样本个数，count表示匹配的样本个数
                count = 0;
                index = 0;
                dist = 0.0;
                while (count < self.__defaultReqMatches) and (index < self.__defaultNbSamples):
                    dist = float(grayFrame[y, x]) - float(self.__samples[index][y, x]);
                    if dist < 0: dist = -dist
                    if dist < self.__defaultRadius: count = count + 1
                    index = index + 1

                if count >= self.__defaultReqMatches:
                    # 判断为背景像素,只有背景点才能被用来传播和更新存储样本值
                    self.__samples[self.__defaultNbSamples][y, x] = 0

                    foreground[y, x] = self.__BG

                    rand = random.randint(0, self.__defaultSubsamplingFactor)
                    if rand == 0:
                        rand = random.randint(0, self.__defaultNbSamples)
                        self.__samples[rand][y, x] = grayFrame[y, x]
                    rand = random.randint(0, self.__defaultSubsamplingFactor)
                    if rand == 0:
                        rand = random.randint(0, 8)
                        yN = y + self.__c_yoff[rand]
                        if yN < 0: yN = 0
                        if yN >= self.__Height: yN = self.__Height - 1
                        rand = random.randint(0, 8)
                        xN = x + self.__c_xoff[rand]
                        if xN < 0: xN = 0
                        if xN >= self.__Width: xN = self.__Width - 1
                        rand = random.randint(0, self.__defaultNbSamples)
                        self.__samples[rand][yN, xN] = grayFrame[y, x]
                else:
                    # 判断为前景像素
                    foreground[y, x] = self.__FG;
                    self.__samples[self.__defaultNbSamples][y, x] += 1
                    if self.__samples[self.__defaultNbSamples][y, x] > 50:
                        rand = random.randint(0, self.__defaultNbSamples)
                        if rand == 0:
                            rand = random.randint(0, self.__defaultNbSamples)
                            self.__samples[rand][y, x] = grayFrame[y, x]
        return foreground


class EraseNoise:
    def gaussian(self, in_path, out_path):
        img = cv2.imread(in_path)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.blur(img, (5, 5))
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("erase", img)
        cv2.waitKey(0)



def main():


    # call to rename file name to satisfy voc2coco.py
    # in_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/start/train/images"
    # out_path = r"/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/start/train/images"
    a = OtherUtilities()
    # a.renamefile("/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/validate/images", "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/validate/images")
    a.delete_suffix("/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/train/images")
    a.delete_suffix("/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/train/xmls")
    a.delete_suffix("/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/validate/images")
    a.delete_suffix("/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-6/start/validate/xmls")


    # for file in os.listdir("/home/lab-1/Downloads/wider/WIDER_train_annotations"):
    #     new_name = file+".xml"
    #     pwd1 = os.path.join("/home/lab-1/Downloads/wider/WIDER_train_annotations", file)
    #     pwd2 = os.path.join("/home/lab-1/Downloads/wider/WIDER_train_annotations", new_name)
    #     command = "mv " + pwd1 + " " + pwd2
    #     print("Processing: ", file)
    #     os.system(command)



    return


if __name__ == '__main__':
    main()




