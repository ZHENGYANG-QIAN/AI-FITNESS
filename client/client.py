import multiprocessing
import sys

import cv2
import mediapipe as mp
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap, QImage, QCloseEvent
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QMessageBox, QLabel, QVBoxLayout, \
    QDialog
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

import ui
from EMADictSmoothing import EMADictSmoothing
from FullBodyPoseEmbedder import FullBodyPoseEmbedder
from PoseClassificationVisualizer import PoseClassificationVisualizer
from PoseClassifier import PoseClassifier
from RepetitionCounter import RepetitionCounter


class MyDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('选择动作类型')
        self.setFixedSize(600, 400)

        label = QLabel('请选择动作类型：', self)
        label.setPixmap(QPixmap("cxk.jpg"))
        label.setScaledContents(True)
        squat_button = QPushButton('深蹲', self)
        squat_button.clicked.connect(self.do_operation1)

        jump_button = QPushButton('开合跳', self)
        jump_button.clicked.connect(self.do_operation2)

        pushup_button = QPushButton('俯卧撑', self)
        pushup_button.clicked.connect(self.do_operation3)

        situp_button = QPushButton('仰卧起坐', self)
        situp_button.clicked.connect(self.do_operation4)

        squat_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        jump_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        pushup_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        situp_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')
        vbox = QVBoxLayout(self)
        vbox.addWidget(label)
        vbox.addWidget(squat_button)
        vbox.addWidget(jump_button)
        vbox.addWidget(pushup_button)
        vbox.addWidget(situp_button)

    def do_operation1(self):
        self.done(1)

    def do_operation2(self):
        self.done(2)

    def do_operation3(self):
        self.done(3)

    def do_operation4(self):
        self.done(4)

    def closeEvent(self, event: QCloseEvent):
        event.ignore()
        response = QMessageBox.question(self, '提示', '是否要退出？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class MyDialog2(QDialog):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('选择检测类型')
        self.setFixedSize(600, 400)

        label = QLabel('请选择检测类型：', self)
        label.setPixmap(QPixmap("cxk.jpg"))
        label.setScaledContents(True)
        pic_button = QPushButton('图片', self)
        pic_button.clicked.connect(self.do_operation1)

        video_button = QPushButton('视频', self)
        video_button.clicked.connect(self.do_operation2)

        realtime_button = QPushButton('实时检测', self)
        realtime_button.clicked.connect(self.do_operation3)

        pic_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        video_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        realtime_button.setStyleSheet(
            '''QPushButton{background:rgb(255, 255, 255, 60);border-radius:5px;font-family:华文行楷;color:black;font-size:30px;}QPushButton:hover\n
            {background:rgb(255, 255, 255, 250);}''')

        vbox = QVBoxLayout(self)
        vbox.addWidget(label)
        vbox.addWidget(pic_button)
        vbox.addWidget(video_button)
        vbox.addWidget(realtime_button)

    def do_operation1(self):
        self.done(1)

    def do_operation2(self):
        self.done(2)

    def do_operation3(self):
        self.done(3)

    def closeEvent(self, event: QCloseEvent):
        event.ignore()
        response = QMessageBox.question(self, '提示', '是否要退出？', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if response == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


import numpy as np

class ClientPoseCount():
    # 健身动作计数核心处理函数
    @classmethod
    def PoseCount(cls, pose_samples_folder, class_name):
        # initialize tracker
        pose_tracker = mp_pose.Pose()

        # initialize embedder
        pose_embedder = FullBodyPoseEmbedder()

        # initialize classifier
        pose_classifier = PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)

        # initialize EMA smoothing
        pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)

        # two thresholds for the specified action
        repetition_counter = RepetitionCounter(class_name=class_name, enter_threshold=6, exit_threshold=4)

        # initialize renderer
        pose_classification_visualizer = PoseClassificationVisualizer(class_name=class_name)
        infor = "计数："
        frame_idx = 0
        output_frame = None
        video_cap = cv2.VideoCapture(1)
        video_cap.open(0)
        global i
        i = 0
        while video_cap.isOpened() & i == 0:
            # get next frame of the video
            success, input_frame = video_cap.read()
            if not success:
                break
            if input_frame is not None:
                # Run pose tracker
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                result = pose_tracker.process(image=input_frame)
                pose_landmarks = result.pose_landmarks
                # draw pose prediction
                output_frame = input_frame.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=output_frame,
                        landmark_list=pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)

                if pose_landmarks is not None:
                    # get landmarks
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array(
                        [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in
                         pose_landmarks.landmark],
                        dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                    # classify the pose on the current frame
                    pose_classification = pose_classifier(pose_landmarks)

                    # smooth classification using EMA
                    pose_classification_filtered = pose_classification_filter(pose_classification)

                    # count repetitions
                    repetitions_count = repetition_counter(pose_classification_filtered)
                else:
                    # No pose -> no classfication on current frame
                    pose_classification = None

                    # smoothing for future frames
                    pose_classification_filtered = pose_classification_filter(dict())
                    pose_classification_filtered = None

                    # take the latest repetitions count
                    repetitions_count = repetition_counter.n_repeats
                infor1 = "当前计数：" + str(repetitions_count) + "个"
                if infor1 != infor:
                    infor = infor1
                    ui.printstr(infor)
                frame_idx += 1
                output_frame = pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=pose_classification_filtered,
                    repetitions_count=repetitions_count)
                output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_BGR2RGB)
                output_frame = cv2.resize(output_frame, dsize=(1600, 900), dst=None, fx=2, fy=2,
                                          interpolation=cv2.INTER_NEAREST)
                h, w, ch = output_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(output_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                q_pixmap = QPixmap.fromImage(q_image)
                ui.labelcamera.setPixmap(q_pixmap)

                # 按‘q’或‘esc’退出
                if cv2.waitKey(1) in [ord('q'), 27]:
                    break
        # 关闭摄像头
        video_cap.release()
        # 关闭窗口
        cv2.destroyAllWindows()


    # 健身计数主函数
    @classmethod
    def PoseCountNumber(cls):
        ui.labelcamera.setGeometry(QRect(340, 355, 1500, 620))
        # building classifier to output CSVs
        dialog = MyDialog()
        response = dialog.exec_()
        if response == 1:
            pose_samples_folder = 'squat_csvs_out'
            class_name = 'down'
            cls.PoseCount(pose_samples_folder, class_name)
        elif response == 2:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'close'
            cls.PoseCount(pose_samples_folder, class_name)
        elif response == 3:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'close'
            cls.PoseCount(pose_samples_folder, class_name)
        elif response == 4:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'close'
            cls.PoseCount(pose_samples_folder, class_name)

class ClientPointProcess():
    # 图片关键点检测函数
    @classmethod
    def PicPointProcess(cls):
        pose = mp.solutions.pose.Pose(static_image_mode=True,
                                      smooth_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getOpenFileName(ui, "选择图片文件", "",
                                                   "Image Files (*.png *.jpg *.bmp *.gif);;All Files (*)", options=options)
        if file_path:
            ui.printstr("已选择的图片文件路径为:" + file_path)
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
            q_pixmap = QPixmap.fromImage(q_image)
            ui.labelcamera.setPixmap(q_pixmap)


    # 视频关键点检测函数
    @classmethod
    def VideoPointProcess(cls):
        global i
        i = 0
        pose = mp_pose.Pose(static_image_mode=False,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        file_path1, _ = QFileDialog.getOpenFileName(ui, "选择视频文件", "",
                                                    "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
                                                    options=options1)
        ui.printstr("已选择的视频文件路径为:" + file_path1)
        video_cap = cv2.VideoCapture(file_path1)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_percent = 620 / height

        # 计算缩放后的新高度和新宽度
        new_height = int(height * scale_percent)
        new_width = int(width * scale_percent)

        # 缩放图片

        ui.labelcamera.setGeometry(QRect(340, 355, new_width, new_height))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filehead = file_path1.split('/')[-1]
        # output_path = "Point-out-" + filehead + '.mp4'
        output_path = "point-out-" + filehead
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
        while video_cap.isOpened() & i == 0:
            success, img = video_cap.read()
            if not success:
                break
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                out.write(img)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
                q_pixmap = QPixmap.fromImage(q_image)
                ui.labelcamera.setPixmap(q_pixmap)
                if cv2.waitKey(1) in [ord('q'), 27]:
                    break
        video_cap.release()
        out.release()
        cv2.destroyAllWindows()
        ui.printstr('输出视频已保存')


    # 实时关键点检测函数
    @classmethod
    def RealTimePointProcess(cls):
        global i
        i = 0
        pose = mp_pose.Pose(static_image_mode=False,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        video_cap = cv2.VideoCapture(0)
        video_cap.open(0)
        while video_cap.isOpened() & i == 0:
            success, img = video_cap.read()
            if not success:
                break
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                q_image = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
                q_pixmap = QPixmap.fromImage(q_image)
                ui.labelcamera.setPixmap(q_pixmap)
                if cv2.waitKey(1) in [ord('q'), 27]:
                    break


    # 关键点检测主函数
    @classmethod
    def PointProcess(cls):
        dialog = MyDialog2()
        response = dialog.exec_()
        if response == 1:
            cls.PicPointProcess()
        elif response == 2:
            cls.VideoPointProcess()
        elif response == 3:
            cls.RealTimePointProcess()

class ClientVideoNumberProcess():
    # 视频计数核心处理函数
    @classmethod
    def VideoNumberProcess(cls, pose_samples_folder, class_name):
        global i
        i = 0
        pose = mp_pose.Pose(static_image_mode=False,
                            smooth_landmarks=True,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        options1 = QFileDialog.Options()
        options1 |= QFileDialog.DontUseNativeDialog
        file_path1, _ = QFileDialog.getOpenFileName(ui, "选择视频文件", "",
                                                    "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
                                                    options=options1)

        ui.printstr("已选择的视频文件路径为:" + file_path1)

        # building classifier to output CSVs
        # initialize tracker
        pose_tracker = mp_pose.Pose()

        # initialize embedder
        pose_embedder = FullBodyPoseEmbedder()

        # initialize classifier
        pose_classifier = PoseClassifier(
            pose_samples_folder=pose_samples_folder,
            pose_embedder=pose_embedder,
            top_n_by_max_distance=30,
            top_n_by_mean_distance=10)

        # initialize EMA smoothing
        pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)

        # two thresholds for the specified action
        repetition_counter = RepetitionCounter(class_name=class_name, enter_threshold=6, exit_threshold=4)

        # initialize renderer
        pose_classification_visualizer = PoseClassificationVisualizer(class_name=class_name)
        infor = "计数："
        frame_idx = 0
        output_frame = None
        video_cap = cv2.VideoCapture(file_path1)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_percent = 620 / height

        # 计算缩放后的新高度和新宽度
        new_height = int(height * scale_percent)
        new_width = int(width * scale_percent)

        # 缩放图片

        ui.labelcamera.setGeometry(QRect(340, 355, new_width, new_height))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filehead = file_path1.split('/')[-1]
        # output_path = "Count-out-" + filehead + '.mp4'
        output_path = "count-out-" + filehead
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height), isColor=True)
        while video_cap.isOpened() & i == 0:
            # get next frame of the video
            success, input_frame = video_cap.read()
            if not success:
                break
            if input_frame is not None:
                # Run pose tracker
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                resized_img = cv2.resize(input_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                result = pose_tracker.process(image=resized_img)
                pose_landmarks = result.pose_landmarks
                # draw pose prediction
                output_frame = resized_img.copy()
                if pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image=output_frame,
                        landmark_list=pose_landmarks,
                        connections=mp_pose.POSE_CONNECTIONS)

                if pose_landmarks is not None:
                    # get landmarks
                    frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                    pose_landmarks = np.array(
                        [[lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width] for lmk in
                         pose_landmarks.landmark],
                        dtype=np.float32)
                    assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                    # classify the pose on the current frame
                    pose_classification = pose_classifier(pose_landmarks)

                    # smooth classification using EMA
                    pose_classification_filtered = pose_classification_filter(pose_classification)

                    # count repetitions
                    repetitions_count = repetition_counter(pose_classification_filtered)
                else:
                    # No pose -> no classfication on current frame
                    pose_classification = None

                    # smoothing for future frames
                    pose_classification_filtered = pose_classification_filter(dict())
                    pose_classification_filtered = None

                    # take the latest repetitions count
                    repetitions_count = repetition_counter.n_repeats
                infor1 = "当前计数：" + str(repetitions_count) + "个"
                if infor1 != infor:
                    infor = infor1
                    ui.printstr(infor)

                frame_idx += 1
                output_frame = pose_classification_visualizer(
                    frame=output_frame,
                    pose_classification=pose_classification,
                    pose_classification_filtered=pose_classification_filtered,
                    repetitions_count=repetitions_count)
                output_frame = cv2.cvtColor(np.array(output_frame), cv2.COLOR_BGR2RGB)
                output_frame = cv2.resize(output_frame, dsize=(new_width, new_height), dst=None, fx=2, fy=2,
                                          interpolation=cv2.INTER_NEAREST)
                out.write(output_frame)
                h, w, ch = output_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(output_frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
                q_pixmap = QPixmap.fromImage(q_image)
                ui.labelcamera.setPixmap(q_pixmap)

                # 按‘q’或‘esc’退出
                if cv2.waitKey(1) in [ord('q'), 27]:
                    break
        video_cap.release()
        out.release()
        cv2.destroyAllWindows()
        ui.printstr('输出视频已保存')


    # 视频动作计数主函数
    @classmethod
    def VideoNumber(cls):
        dialog = MyDialog()
        response = dialog.exec_()
        if response == 1:
            pose_samples_folder = 'squat_csvs_out'
            class_name = 'down'
            cls.VideoNumberProcess(pose_samples_folder, class_name)
        elif response == 2:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'open'
            cls.VideoNumberProcess(pose_samples_folder, class_name)
        elif response == 3:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'open'
            cls.VideoNumberProcess(pose_samples_folder, class_name)
        elif response == 4:
            pose_samples_folder = 'jump_csvs_out'
            class_name = 'open'
            cls.VideoNumberProcess(pose_samples_folder, class_name)

class ClientCleanText():
    # 清空输出
    @classmethod
    def CleanText(cls):
        ui.textBrowser.clear()
        ui.labelcamera.setGeometry(QRect(340, 355, 1500, 620))
        q_pixmap = QPixmap('white.jpg')
        ui.labelcamera.setPixmap(q_pixmap)

class ClientReadme():
    # 使用说明
    @classmethod
    def Readme(cls):
        # 打开matters.txt文件
        with open("matter.txt", "r", encoding="utf-8") as file:
            result = file.read()
            ui.printstr(result)

class ClientRecognitionEnd():
    # 停止检测
    @classmethod
    def RecognitionEnd(cls):
        global i
        i = 1


if __name__ == '__main__':
    multiprocessing.Process()
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    # 调用Hand.py绘制gui构建桌面应用并指定每个控件的作用
    MainWindow = QMainWindow()
    ui = ui.Ui_MainWindow()
    ui.setupUi(MainWindow)
    ui.pushButton_clean.clicked.connect(ClientCleanText.CleanText)
    ui.pushButton_readme.clicked.connect(ClientReadme.Readme)
    ui.pushButton_stop.clicked.connect(ClientRecognitionEnd.RecognitionEnd)
    ui.pushButton_number.clicked.connect(ClientPoseCount.PoseCountNumber)
    ui.pushButton_video.clicked.connect(ClientVideoNumberProcess.VideoNumber)
    ui.pushButton_point.clicked.connect(ClientPointProcess.PointProcess)
    MainWindow.show()
    sys.exit(app.exec_())
