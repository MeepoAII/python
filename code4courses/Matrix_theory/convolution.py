#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import datetime
import pylab
import sys



# 定义卷积操作的函数
def conv(image, weight):
    height, width = image.shape
    h, w = weight.shape
    # 经滑动卷积操作后得到的新的图像的尺寸
    new_h = height - h + 1
    new_w = width - w + 1
    new_image = np.zeros((new_h, new_w), dtype=np.float)
    # 进行卷积操作,实则是对应的窗口覆盖下的矩阵对应元素值相乘,卷积操作
    for i in range(new_w):
        for j in range(new_h):
            new_image[i, j] = np.sum(image[i:i + h, j:j + w] * weight)
    # 去掉矩阵乘法后的小于0的和大于255的原值,重置为0和255
    new_image = new_image.clip(0, 255)
    new_image = np.rint(new_image).astype('uint8')
    return new_image


def conv2(img, conv_filter):


    img_h, img_w, img_ch = img.shape
    filter_num, filter_h, filter_w, img_ch = conv_filter.shape
    feature_h = img_h - filter_h + 1
    feature_w = img_w - filter_w + 1

    # 初始化输出的特征图片，由于没有使用零填充，图片尺寸会减小
    img_out = np.zeros((feature_h, feature_w, filter_num))
    img_matrix = np.zeros((feature_h * feature_w, filter_h * filter_w * img_ch))
    filter_matrix = np.zeros((filter_h * filter_w * img_ch, filter_num))

    # 将输入图片张量转换成矩阵形式
    for i in range(feature_h * feature_w):
        for j in range(img_ch):
            img_matrix[i, j * filter_h * filter_w:(j + 1) * filter_h * filter_w] = \
                img[np.uint16(i / feature_w):np.uint16(i / feature_w + filter_h),
                np.uint16(i % feature_w):np.uint16(i % feature_w + filter_w), j].reshape(filter_h * filter_w)

    # 将卷积核张量转换成矩阵形式
    for i in range(filter_num):
        filter_matrix[:, i] = conv_filter[i, :].reshape(filter_w * filter_h * img_ch)

    feature_matrix = np.dot(img_matrix, filter_matrix)

    for i in range(filter_num):
        img_out[:, :, i] = feature_matrix[:, i].reshape(feature_h, feature_w)

    return img_out



if __name__ == "__main__":

    start = datetime.datetime.now()

    # 读取图像数据并且转换为对应的numpy下的数组
    A = Image.open("/home/lab-1/Downloads/meepo.jpg", 'r')
    output_path = "./outputPic2/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    a = np.array(A)
    # 定义的三种类型的卷积核  3+3+2  输出8张图片
    sobel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    sobel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    sobel = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))

    prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))

    laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
    laplacian_2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))

    my_test = np.zeros((1, 3, 3, 3))
    w_shape = (1, 3, 3, 3)

    weights = np.ndarray(shape=(1,3,3,3))

    # weight_list = ("weights")
    # weight_list = ("weights", "sobel_x", "sobel_y", "sobel", "prewitt_x", "prewitt_y", "prewitt", "laplacian", "laplacian_2")



    print("Gridient detection\n")
    for i in range(8):
        test = conv2(a, weights)
        #
        # print("Convolution on R")
        # R = conv(a[:, :, 0], eval(w))
        # print("Convolution on G")
        # G = conv(a[:, :, 1], eval(w))
        # print("Convolution on B")
        # B = conv(a[:, :, 2], eval(w))
        #
        #
        # I = np.stack((R, G, B), axis=2)
    #Image.fromarray(test).save("%s//%s.jpg" % (output_path, test))

    end = datetime.datetime.now()
    print("Time used: ", end-start)





