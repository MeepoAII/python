# -*- coding: UTF-8 -*-
# python2 or python3
import os
import cv2
import threading
import queue
# import chardet

def convert_from_hikvision(ffmpeg_path, input_path, output_path):
    # 用ffmpeg命令处理海康威视视频，使其能播放，方便剪辑
    files = os.listdir(input_path)
    for file in files:
        input_pwd = os.path.join(input_path, file)
        output_pwd = os.path.join(output_path, file)
        command = ffmpeg_path + " -i " + input_pwd + " -vcodec " + "copy " + output_pwd
        result = os.popen(command)
        print(result.read())

    return


def take_picture_from_video_per_x_seconds(in_path, out_path, seconds):
    # 按秒把视频切为图片
    files = os.listdir(in_path)
    files.sort()
    for file in files:
        c = 0
        print("Processing: ", file)
        video_pwd = os.path.join(in_path, file)
        new_out_path = os.path.join(out_path, file.split(".")[0])
        os.makedirs(new_out_path)
        try:
            video = cv2.VideoCapture(video_pwd)
            fps = int(round(video.get(cv2.CAP_PROP_FPS)))
            print("FPS: ", fps)
            if video.isOpened():
                rval, frame = video.read()
                print("Video reading sucess")
            else:
                rval = False

            while rval:
                rval, frame = video.read()

                # according to the fps of this video, calculate the how many frame should i skip
                steps = int(fps*seconds)
                if (c%steps == 0):
                    img_name = str(c)+".jpg"
                    img_pwd = os.path.join(new_out_path, img_name)
                    cv2.imwrite(img_pwd, frame)
                    cv2.waitKey(1)
                    print("Saving {}.jpg".format(c))
                c += 1
            video.release()
        except:
            print("Something wrong!")

    return


def main():
    # call examples
    # path = r"D:\Data\wp\get-background\video-avaliable\0514"
    # output_path = r"D:\Data\wp\get-background\background"
    # parse_video(path, output_path, 500)
    # ffmpeg_path = r"C:\Users\meepo\Downloads\ffmpeg-20190703-93a73df-win64-static\bin\ffmpeg.exe"
    # input_path = r"H:\postgraduate\1\data\vide-original\0326"
    # output_path = r"H:\postgraduate\1\data\video-convert\0326"
    # convert_from_hikvision(ffmpeg_path, input_path, output_path)

    # Called on August 07, 2019 at 18:59:07
    # use to process test data
    # first cut video per 3 seconds
    in_path = r'/media/lab-1/BOBOU/视频剪辑'
    out_path = r'/media/lab-1/BOBOU/视频剪辑/cut-per-2-seconds'
    take_picture_from_video_per_x_seconds(in_path, out_path, 1)



    return



def cut_video(input_path, output_path, start, end):
    return


if __name__ == '__main__':
    main()
