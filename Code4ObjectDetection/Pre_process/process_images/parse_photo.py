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


# 手动绘制矩形ROI，并将此选定区域用于该目录下所有图片，然后将截取后的ROI保存到输出目录
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
        cropImg = im2[y : y+h, x:x+w]
        cv2.imwrite(pwd_out, cropImg)
        cv2.waitKey(1)
        print(i)
        print(rect)
        i = i+1

    return

# 手动绘制多边形ROI，并将此选定区域用于该目录下所有图片，然后将截取后的ROI保存到输出目录
def draw_multi_roi():

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
    in_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/test"
    out_path = "/media/lab-1/machine/Data/wp/object-detection/my-dataset/try-4/test_results"
    cut_photo(in_path, out_path)
    # in_path = '/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-per-3-seconds/2019-03-23-09-13-49-2019-03-23-11-13-52'
    # rename_images(in_path, 3)
    # in_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-per-3-seconds/2019-03-19-14-20-24-2019-03-19-16-20-27"
    # out_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-four-region/2019-03-19-14-20-24-2019-03-19-16-20-27/left"
    # cut_photo(in_path, out_path)

    # call to get minAreaRect, get the rectangle coordinates
    # in_path = r"/home/lab-1/PycharmProjects/PrivateCode/postgraduate/seat-rec/label_program/Data/2019-03-19-14-20-24.jpg"
    # polygon_points = np.array([[417,361],[398,379],[383,394],[365,414],[348,430],[332,448],[316,465],[300,481],[274,509],[260,528],[244,545],[235,565],[224,580],[209,597],[198,612],[188,627],[169,644],[148,672],[131,692],[111,715],[97,725],[154,738],[182,743],[227,748],[276,755],[319,754],[337,734],[363,698],[380,677],[395,641],[412,618],[436,579],[463,538],[476,514],[499,488],[521,459],[548,420],[569,388],[582,365],[590,350],[594,338],[592,338],[570,335],[550,335],[519,335],[497,336],[472,338],[446,348],[431,353]], dtype=np.int32)
    # test = CropImage_Using_MinAreaRect(points_array=polygon_points, in_path=in_path)
    # test.cut()

    # Read coordinates from a file and use them to create a rectangle, rotate according to the angle, and finally crop
    # a = CropImageUsingCoordinates()
    # in_directory = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-per-3-seconds/2019-03-19-16-20-27-2019-03-19-18-17-47"
    # out_directory = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-four-region/2019-03-19-16-20-27-2019-03-19-18-17-47/R1"
    #
    # files = os.listdir(in_directory)
    # for file in files:
    #     print("Processing: ", file)
    #     in_pwd = os.path.join(in_directory, file)
    #     out_pwd = os.path.join(out_directory, file)
    #     coordinates_file_pwd = r"/home/lab-1/PycharmProjects/PrivateCode/postgraduate/seat-rec/label_program/get_coordinates/coordinates.txt"
    #     a.Start(coordinates_file_pwd, in_pwd, out_pwd)


    # rotate the reverse image
    # in_directory = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-four-region/2019-03-19-16-20-27-2019-03-19-18-17-47/R1"
    # out_directory = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/cut-four-region/2019-03-19-16-20-27-2019-03-19-18-17-47_rotate/R1"
    # files = os.listdir(in_directory)
    # for file in files:
    #     in_path = os.path.join(in_directory, file)
    #     out_path = os.path.join(out_directory, file)
    #     rotate = OtherUtilities()
    #     rotate.rotate(in_path, out_path, 180, 1)
    #     print("Rotating: ", file)


    # from windows call, add the background to sequence
    # in_path = r"D:\Data\wp\test-1-2019-08-06\test-subtraction\R1"
    # out_path = r"D:\Data\wp\test-1-2019-08-06\test-subtraction\R1"
    # background_path = r"D:\Data\wp\test-1-2019-08-06\test-subtraction\background\0-1.jpg"
    #
    # a = OtherUtilities()
    # a.adjust_image_sequence(in_path, background_path)


    # Frame difference
    # in_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/R1"
    # background_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/background/1.jpg"
    # out_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/R1-sub"
    # files = os.listdir(in_path)
    # a = BackgroundSubtraction()
    # i = 0
    # for file in files:
    #     in_pwd = os.path.join(in_path, file)
    #     image_1 = cv2.imread(background_path)
    #     image_2 = cv2.imread(in_pwd)
    #     d_frame = a.absdiff(image_1, image_2, 60)
    #     print("Processing: ", file)
    #     # cv2.imshow("foreground", d_frame)
    #     # cv2.waitKey(0)
    #     pwd = os.path.join(out_path, file)
    #     cv2.imwrite(pwd, d_frame)
    #     print("sucess")
    #     i += 1



    # Test vibe, bad
    # in_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/R1"
    # files = os.listdir(in_path)
    # image = cv2.cvtColor(cv2.imread(os.path.join(in_path, files[0])), cv2.COLOR_BGR2GRAY)
    # vibe = ViBe(image)
    # i = 0
    # for file in files:
    #     if i == 20:
    #
    #         in_pwd = os.path.join(in_path, file)
    #         in_image = cv2.imread(in_pwd)
    #         image_gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    #         foreground = vibe.update(image_gray)
    #         cv2.imshow("foreground", foreground)
    #         cv2.waitKey(0)
    #     i += 1


    # test erase noise
    # a = EraseNoise()
    # in_path = r"C:\Users\meepo\Downloads\bgslibrary2_qtgui_opencv320_x64\output\fg\2.png"
    # a.gaussian(in_path, "")

    # Other Utilities: crop image
    # a = OtherUtilities()
    # in_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/R1-sub"
    # out_path = r"/media/lab-1/machine/Data/wp/test-1-2019-08-06/test-subtraction/R1-sub-crop"
    # files = os.listdir(in_path)
    # for file in files:
    #     in_pwd = os.path.join(in_path, file)
    #     out_pwd = os.path.join(out_path, file)
    #     a.crop_image_margin(in_pwd, out_pwd)




    return


if __name__ == '__main__':
    main()




