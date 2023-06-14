import sys
import os
import time

import cv2
from PyQt5.QtCore import QTimer

import ui
import multiprocessing
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


import csv
import numpy as np


class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(self,
                 pose_samples_folder,
                 pose_embedder,
                 file_extension='csv',
                 file_separator=',',
                 n_landmarks=33,
                 n_dimensions=3,
                 top_n_by_max_distance=30,
                 top_n_by_mean_distance=10,
                 axes_weights=(1., 1., 0.2)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)

    def _load_pose_samples(self,
                           pose_samples_folder,
                           file_extension,
                           file_separator,
                           n_landmarks,
                           n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.

        Required folder structure:
          neutral_standing.csv
          pushups_down.csv
          pushups_up.csv
          squats_down.csv
          ...

        Required CSV structure:
          sample_00001,x1,y1,z1,x2,y2,z2,....
          sample_00002,x1,y1,z1,x2,y2,z2,....
          ...
        """
        # Each file in the folder represents one pose class.
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                    ))

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # Find outliers in target poses
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name, count in pose_classification.items() if
                           count == max(pose_classification.values())]

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(
            pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([-1, 1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        return result


class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


class RepetitionCounter(object):
    """Counts number of repetitions of given target pose class."""

    def __init__(self, class_name, enter_threshold=6, exit_threshold=4):
        self._class_name = class_name

        # If pose counter passes given threshold, then we enter the pose.
        self._enter_threshold = enter_threshold
        self._exit_threshold = exit_threshold

        # Either we are in given pose or not.
        self._pose_entered = False

        # Number of times we exited the pose.
        self._n_repeats = 0

    @property
    def n_repeats(self):
        return self._n_repeats

    def __call__(self, pose_classification):
        """Counts number of repetitions happend until given frame.

        We use two thresholds. First you need to go above the higher one to enter
        the pose, and then you need to go below the lower one to exit it. Difference
        between the thresholds makes it stable to prediction jittering (which will
        cause wrong counts in case of having only one threshold).

        Args:
          pose_classification: Pose classification dictionary on current frame.
            Sample:
              {
                'pushups_down': 8.3,
                'pushups_up': 1.7,
              }

        Returns:
          Integer counter of repetitions.
        """
        # Get pose confidence.
        pose_confidence = 0.0
        if self._class_name in pose_classification:
            pose_confidence = pose_classification[self._class_name]

        # On the very first frame or if we were out of the pose, just check if we
        # entered it on this frame and update the state.
        if not self._pose_entered:
            self._pose_entered = pose_confidence > self._enter_threshold
            return self._n_repeats

        # If we were in the pose and are exiting it, then increase the counter and
        # update the state.
        if pose_confidence < self._exit_threshold:
            self._n_repeats += 1
            self._pose_entered = False

        return self._n_repeats


import io
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


class PoseClassificationVisualizer(object):
    """Keeps track of classifcations for every frame and renders them."""

    def __init__(self,
                 class_name,
                 plot_location_x=0,
                 plot_location_y=0,
                 plot_max_width=0.6,
                 plot_max_height=0.6,
                 plot_figsize=(10, 6),
                 plot_x_max=None,
                 plot_y_max=None,
                 counter_location_x=0.85,
                 counter_location_y=0.05,
                 counter_font_path='https://github.com/googlefonts/roboto/blob/main/src/hinted/Roboto-Regular.ttf?raw=true',
                 counter_font_color='Black',
                 counter_font_size=0.5):
        self._class_name = class_name
        self._plot_location_x = plot_location_x
        self._plot_location_y = plot_location_y
        self._plot_max_width = plot_max_width
        self._plot_max_height = plot_max_height
        self._plot_figsize = plot_figsize
        self._plot_x_max = plot_x_max
        self._plot_y_max = plot_y_max
        self._counter_location_x = counter_location_x
        self._counter_location_y = counter_location_y
        self._counter_font_path = counter_font_path
        self._counter_font_color = counter_font_color
        self._counter_font_size = counter_font_size

        self._counter_font = None

        self._pose_classification_history = []
        self._pose_classification_filtered_history = []

    def __call__(self,
                 frame,
                 pose_classification,
                 pose_classification_filtered,
                 repetitions_count):
        """Renders pose classifcation and counter until given frame."""
        # Extend classification history.
        self._pose_classification_history.append(pose_classification)
        self._pose_classification_filtered_history.append(pose_classification_filtered)

        # Output frame with classification plot and counter.
        output_img = Image.fromarray(frame)

        output_width = output_img.size[0]
        output_height = output_img.size[1]

        # Draw the plot.
        img = self._plot_classification_history(output_width, output_height)
        img.thumbnail((int(output_width * self._plot_max_width),
                       int(output_height * self._plot_max_height)),
                      Image.ANTIALIAS)
        output_img.paste(img,
                         (int(output_width * self._plot_location_x),
                          int(output_height * self._plot_location_y)))

        # Draw the count.
        output_img_draw = ImageDraw.Draw(output_img)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 20
        font = ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc', 25)
        output_img_draw.text((output_width * self._counter_location_x,
                              output_height * self._counter_location_y),
                             '计数:' + str(repetitions_count),
                             font=font,
                             fill=self._counter_font_color)

        return output_img

    def _plot_classification_history(self, output_width, output_height):
        fig = plt.figure(figsize=self._plot_figsize)

        for classification_history in [self._pose_classification_history,
                                       self._pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif self._class_name in classification:
                    y.append(classification[self._class_name])
                else:
                    y.append(0)
            plt.plot(y, linewidth=7)

        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('骨架')
        plt.ylabel('置信度')
        plt.title('骨架变化状态监测 '.format(self._class_name))
        plt.legend(loc='upper right')

        if self._plot_y_max is not None:
            plt.ylim(top=self._plot_y_max)
        if self._plot_x_max is not None:
            plt.xlim(right=self._plot_x_max)

        # Convert plot to image.
        buf = io.BytesIO()
        dpi = min(
            output_width * self._plot_max_width / float(self._plot_figsize[0]),
            output_height * self._plot_max_height / float(self._plot_figsize[1]))
        fig.savefig(buf, dpi=dpi)
        buf.seek(0)
        img = Image.open(buf)
        plt.close()

        return img


def squat():
    class_name = 'down'
    # building classifier to output CSVs
    pose_samples_folder = 'squat_csvs_out'

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


def jump():
    class_name = 'close'
    # building classifier to output CSVs
    pose_samples_folder = 'jump_csvs_out'

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


def crul():
    class_name = 'down'
    # building classifier to output CSVs
    pose_samples_folder = 'squat_csvs_out'

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


def pullup():
    class_name = 'down'
    # building classifier to output CSVs
    pose_samples_folder = 'squat_csvs_out'

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


def pictureprocess():
    pose = mp.solutions.pose.Pose(static_image_mode=True,
                                  smooth_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    file_path, _ = QFileDialog.getOpenFileName(ui,"选择图片文件", "","Image Files (*.png *.jpg *.bmp *.gif);;All Files (*)", options=options)
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

def videoprocess():
    global i
    i = 0
    pose = mp_pose.Pose(static_image_mode=False,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    ui.timer_camera = QTimer()
    options1 = QFileDialog.Options()
    options1 |= QFileDialog.DontUseNativeDialog
    file_path1, _ = QFileDialog.getOpenFileName(ui, "选择视频文件", "", "Video Files (*.mp4 *.avi *.mkv);;All Files (*)",
                                               options=options1)
    ui.printstr("已选择的视频文件路径为:" + file_path1)
    cap = cv2.VideoCapture(file_path1)
    while i == 0:
        ret, frame = cap.read()  # 读取视频帧
        if not ret:
            break
        # 转换BGR视频帧为RGB格式，因为Qt中使用的是RGB格式
        # 将视频帧转换为QImage格式
        results = pose.process(frame)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        q_pixmap = QPixmap.fromImage(q_image)
        ui.labelcamera.setPixmap(q_pixmap)
        if cv2.waitKey(1) == ord('q'):  # 按下q键退出循环
            break


# 清空输出
def cleantext():
    ui.textBrowser.clear()
    q_pixmap = QPixmap('white.jpg')
    ui.labelcamera.setPixmap(q_pixmap)


# 使用说明
def readme():
    # 打开matters.txt文件
    with open("matter.txt", "r", encoding="utf-8") as file:
        result = file.read()
        ui.printstr(result)


# 停止检测
def Recognitionend():
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
    ui.pushButton_squat.clicked.connect(squat)
    ui.pushButton_clean.clicked.connect(cleantext)
    ui.pushButton_readme.clicked.connect(readme)
    ui.pushButton_stop.clicked.connect(Recognitionend)
    ui.pushButton_jump.clicked.connect(jump)
    ui.pushButton_curl.clicked.connect(crul)
    ui.pushButton_pullup.clicked.connect(pullup)
    ui.pushButton_picture.clicked.connect(pictureprocess)
    ui.pushButton_video.clicked.connect(videoprocess)
    MainWindow.show()
    sys.exit(app.exec_())
