/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.mlkit.vision.demo.java.posedetector;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.Task;
import com.google.android.odml.image.MlImage;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.demo.GraphicOverlay;
import com.google.mlkit.vision.demo.java.VisionProcessorBase;
import com.google.mlkit.vision.demo.java.posedetector.classification.PoseClassifierProcessor;
import com.google.mlkit.vision.demo.preference.PreferenceUtils;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseDetection;
import com.google.mlkit.vision.pose.PoseDetector;
import com.google.mlkit.vision.pose.PoseDetectorOptionsBase;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

/**
 * A processor to run pose detector.
 */
public class PoseDetectorProcessor
        extends VisionProcessorBase<PoseDetectorProcessor.PoseWithClassification> {
    private static final String TAG = "PoseDetectorProcessor";

    // 构造函数中初始化的pose检测器。
    private final PoseDetector detector;
    private final boolean showInFrameLikelihood;
    private final boolean visualizeZ;
    private final boolean rescaleZForVisualization;
    // 控制是否在检测到姿势时运行分类器。
    private final boolean runClassification;
    // 控制是否在流模式下运行分类器。
    private final boolean isStreamMode;
    // 应用程序上下文，由构造函数设置。
    private final Context context;
    // 单个线程执行器，用于运行分类器。
    private final Executor classificationExecutor;

    // 一个姿势分类器处理器，用于在检测到姿势时运行分类器并获取分类结果。
    private PoseClassifierProcessor poseClassifierProcessor;

    private String poseSamplesFile;

    private String[] poseClasses;

    /**
     * Internal class to hold Pose and classification results.
     */
    protected static class PoseWithClassification {
        private final Pose pose;
        private final List<String> classificationResult;

        public PoseWithClassification(Pose pose, List<String> classificationResult) {
            this.pose = pose;
            this.classificationResult = classificationResult;
        }

        public Pose getPose() {
            return pose;
        }

        public List<String> getClassificationResult() {
            return classificationResult;
        }
    }

    public PoseDetectorProcessor(
            Context context,
            boolean runClassification,
            boolean isStreamMode) {
        super(context);
        PoseDetectorOptionsBase options =
                PreferenceUtils.getPoseDetectorOptionsForLivePreview(context);
        Log.i(TAG, "Using Pose Detector with options " + options);
        this.showInFrameLikelihood =
                PreferenceUtils.shouldShowPoseDetectionInFrameLikelihoodLivePreview(context);
        this.visualizeZ = PreferenceUtils.shouldPoseDetectionVisualizeZ(context);
        this.rescaleZForVisualization = PreferenceUtils.shouldPoseDetectionRescaleZForVisualization(context);
        detector = PoseDetection.getClient(options);
        this.runClassification = runClassification;
        this.isStreamMode = isStreamMode;
        this.context = context;
        classificationExecutor = Executors.newSingleThreadExecutor();
    }

    public PoseDetectorProcessor(
            Context context,
            boolean runClassification,
            boolean isStreamMode,
            String poseSamplesFile,
            String[] poseClasses) {
        this(context, runClassification, isStreamMode);
        this.poseSamplesFile = poseSamplesFile;
        this.poseClasses = poseClasses;
    }

    public PoseDetectorProcessor(
            Context context,
            boolean runClassification,
            boolean isStreamMode,
            String poseSamplesFile,
            String poseClass) {
        this(context, runClassification, isStreamMode);
        this.poseSamplesFile = poseSamplesFile;
        this.poseClasses = new String[]{poseClass};
    }

    @Override
    public void stop() {
        super.stop();
        detector.close();
    }

    // 这是 PoseDetectorProcessor 类中一个用于检测输入图像中姿态检测和分类的方法。
    // 它接受 InputImage 对象并返回一个 Task<PoseWithClassification> 对象，该对象包含了 pose 和 classification 结果。
    // 在这个方法中，使用 detector.process(image)方法将输入图片传递给 pose 检测器并获取 Pose 结果。
    // 然后，如果 runClassification 为 true，则创建一个新的 PoseClassifierProcessor 对象，将其传递给 pose 姿态检测器，并获取分类结果。
    // 最后，将结果封装在 PoseWithClassification 对象中返回。continueWith() 方法在异步任务完成后执行，因此在此处等待任务完成并返回结果。
    @Override
    protected Task<PoseWithClassification> detectInImage(InputImage image) {
        return detector
                .process(image)
                .continueWith(
                        classificationExecutor,
                        task -> {
                            Pose pose = task.getResult();
                            List<String> classificationResult = new ArrayList<>();
                            if (runClassification) {
                                if (poseClassifierProcessor == null) {
                                    poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode, poseSamplesFile, poseClasses);
                                }
                                classificationResult = poseClassifierProcessor.getPoseResult(pose);
                            }
                            return new PoseWithClassification(pose, classificationResult);
                        });
    }

    // 这是 PoseDetectorProcessor 类中一个用于检测MLImage对象中姿态检测和分类的方法。
    @Override
    protected Task<PoseWithClassification> detectInImage(MlImage image) {
        return detector
                .process(image)
                .continueWith(
                        classificationExecutor,
                        task -> {
                            Pose pose = task.getResult();
                            List<String> classificationResult = new ArrayList<>();
                            if (runClassification) {
                                if (poseClassifierProcessor == null) {
                                    poseClassifierProcessor = new PoseClassifierProcessor(context, isStreamMode, poseSamplesFile, poseClasses);
                                }
                                classificationResult = poseClassifierProcessor.getPoseResult(pose);
                            }
                            return new PoseWithClassification(pose, classificationResult);
                        });
    }

    // 这是 PoseDetectorProcessor 类中的一个方法，用于在成功完成检测和分类后，在图形叠加层上添加 PoseGraphic 对象来可视化姿势并显示分类结果。
    // 该方法接受两个参数：
    // PoseWithClassification 对象和图形重叠层对象 GraphicOverlay。
    // 然后在 graphicOverlay 上添加一个新的 PoseGraphic 对象，该对象用于可视化检测到的姿势，并在需要时显示分类结果。
    // PoseGraphic 对象接受多个参数，包括图形重叠层、检测到的姿势、是否显示框架可能性等，以及分类结果列表。
    // 在PoseGraphic内部，它将利用所有给定参数来绘制出识别到的人体姿势，并显示分类结果。
    // 此方法在成功完成检测和分类后被调用。
    @Override
    protected void onSuccess(
            @NonNull PoseWithClassification poseWithClassification,
            @NonNull GraphicOverlay graphicOverlay) {
        graphicOverlay.add(
                new PoseGraphic(
                        graphicOverlay,
                        poseWithClassification.pose,
                        showInFrameLikelihood,
                        visualizeZ,
                        rescaleZForVisualization,
                        poseWithClassification.classificationResult));
    }

    @Override
    protected void onFailure(@NonNull Exception e) {
        Log.e(TAG, "Pose detection failed!", e);
    }

    @Override
    protected boolean isMlImageEnabled(Context context) {
        // Use MlImage in Pose Detection by default, change it to OFF to switch to InputImage.
        return true;
    }
}
