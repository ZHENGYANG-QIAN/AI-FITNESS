import os
import shutil

import cv2

video_path = 'open.mp4'  # 视频地址
EXTRACT_FOLDER = 'open'  # 存放帧图片的位置
EXTRACT_FREQUENCY = 1  # 帧提取频率


def extract_frames(video_path, dst_folder, index):
    # 实例化视频对象
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    # 循环遍历视频中的所有帧
    while True:
        # 逐帧读取
        _, frame = video.read()
        if frame is None:
            break
        # 按照设置的频率保存图片
        if frame_count % EXTRACT_FREQUENCY == 0:
            # 设置保存文件名
            save_path = "{}/{:>03d}.jpg".format(dst_folder, index)
            # 保存图片
            cv2.imwrite(save_path, frame)
            # 保存图片数＋1
            index += 1
        # 读取视频帧数＋1
        frame_count += 1
    # 视频总帧数
    print(f'the number of frames: {frame_count}')
    # 打印出所提取图片的总数
    print("Totally save {:d} imgs".format(index - 1))

    # 计算FPS
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')  # Find OpenCV version
    # (major_ver, minor_ver, subminor_ver) = (4, 5, 4)
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)  # 获取当前版本opencv的FPS
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video.get(cv2.CAP_PROP_FPS)  # 获取当前版本opencv的FPS
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    # 最后释放掉实例化的视频
    video.release()


def main():
    # 递归删除之前存放帧图片的文件夹，并新建一个
    try:
        shutil.rmtree(EXTRACT_FOLDER)
    except OSError:
        pass
    os.mkdir(EXTRACT_FOLDER)
    # 抽取帧图片，并保存到指定路径
    extract_frames(video_path, EXTRACT_FOLDER, 1)


if __name__ == '__main__':
    main()
