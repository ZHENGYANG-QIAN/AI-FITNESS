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

package com.google.mlkit.vision.demo.java.posedetector.classification;

import android.content.Context;
import android.media.AudioManager;
import android.media.ToneGenerator;
import android.os.Looper;
import android.util.Log;

import androidx.annotation.WorkerThread;

import com.google.common.base.Preconditions;
import com.google.mlkit.vision.pose.Pose;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Accepts a stream of {@link Pose} for classification and Rep counting.
 */
// 这是一个姿势分类器处理器，用于接收姿势流并进行分类和重复次数计数。
// 它可以接收姿势数据流，对其进行分类，并返回格式化的结果，如“姿势类别：X次”和“姿势类别：[0.0-1.0]置信度”。
// 其中，在流模式下，它还可以使用指数移动平均值平滑处理来平滑输入数据，并使用重复计数器来跟踪特定动作（比如卧推和深蹲）的重复次数，并在每次更新时播放提示音。
// 这个处理器是根据指定的POSE_SAMPLES_FILE对姿势进行分类，其中包含一组示例姿势的CSV文件。
public class PoseClassifierProcessor {
    private static final String TAG = "PoseClassifierProcessor";
    private static final String POSE_SAMPLES_FILE = "pose/fitness_pose_samples.csv";

    // Specify classes for which we want rep counting.
    // These are the labels in the given {@code POSE_SAMPLES_FILE}. You can set your own class labels
    // for your pose samples.
    private static final String PUSHUPS_CLASS = "pushups_down";
    private static final String SQUATS_CLASS = "squats_down";
    private static final String[] POSE_CLASSES = {
            PUSHUPS_CLASS, SQUATS_CLASS
    };

    private final boolean isStreamMode;

    // 交互移动平均值平滑处理器
    private EMASmoothing emaSmoothing;
    // 重复计数器
    private List<RepetitionCounter> repCounters;
    // 用于对输入的姿势进行分类
    private PoseClassifier poseClassifier;
    // 保存上一次重复计数结果的字符串表示
    private String lastRepResult;

    @WorkerThread
    public PoseClassifierProcessor(Context context, boolean isStreamMode) {
        // 首先使用Preconditions.checkState（）方法检查当前线程不是UI线程
        Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
        this.isStreamMode = isStreamMode;
        if (isStreamMode) {
            emaSmoothing = new EMASmoothing();
            repCounters = new ArrayList<>();
            lastRepResult = "";
        }
        loadPoseSamples(context);
    }

    private void loadPoseSamples(Context context) {
        List<PoseSample> poseSamples = new ArrayList<>();
        try {
            // 使用BufferedReader从文件中逐行读取csv文件，并将每一行转换为PoseSample对象
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(context.getAssets().open(POSE_SAMPLES_FILE)));
            String csvLine = reader.readLine();
            while (csvLine != null) {
                // If line is not a valid {@link PoseSample}, we'll get null and skip adding to the list.
                PoseSample poseSample = PoseSample.getPoseSample(csvLine, ",");
                if (poseSample != null) {
                    poseSamples.add(poseSample);
                }
                csvLine = reader.readLine();
            }
        } catch (IOException e) {
            Log.e(TAG, "Error when loading pose samples.\n" + e);
        }
        // 使用poseSamples初始化PoseClassifier对象。
        poseClassifier = new PoseClassifier(poseSamples);
        // 如果isStreamMode为true，则为POSE_CLASSES数组中的每个类别创建一个重复计数器（RepetitionCounter）并将其添加到repCounters数组中。
        if (isStreamMode) {
            for (String className : POSE_CLASSES) {
                repCounters.add(new RepetitionCounter(className));
            }
        }
    }

    /**
     * Given a new {@link Pose} input, returns a list of formatted {@link String}s with Pose
     * classification results.
     *
     * <p>Currently it returns up to 2 strings as following:
     * 0: PoseClass : X reps
     * 1: PoseClass : [0.0-1.0] confidence
     */
    // 用于对姿势进行分类并获取分类结果。
    // 输入一个姿势pose。
    // 返回一个字符串列表，其中包含表示每个类别置信度的字符串和更新重复次数计数器的结果。
    @WorkerThread
    public List<String> getPoseResult(Pose pose) {
        // 使用Preconditions.checkState（）方法检查当前线程不是UI线程。
        Preconditions.checkState(Looper.myLooper() != Looper.getMainLooper());
        // 定义一个空列表result用于存储所有结果。
        List<String> result = new ArrayList<>();
        // 使用poseClassifier.classify（）方法将姿势pose传递给PoseClassifier分析器，获取其分类结果classification。
        ClassificationResult classification = poseClassifier.classify(pose);

        // Update {@link RepetitionCounter}s if {@code isStreamMode}.
        // 如果isStreamMode为true，则执行平滑处理，并使用传入的分类结果更新重复计数器，同时播放"Beep"声音通知用户重复次数已经更新。
        // 最后，将最新的结果添加到result列表中。
        if (isStreamMode) {
            // Feed pose to smoothing even if no pose found.
            classification = emaSmoothing.getSmoothedResult(classification);

            // Return early without updating repCounter if no pose found.
            if (pose.getAllPoseLandmarks().isEmpty()) {
                result.add(lastRepResult);
                return result;
            }

            for (RepetitionCounter repCounter : repCounters) {
                int repsBefore = repCounter.getNumRepeats();
                int repsAfter = repCounter.addClassificationResult(classification);
                if (repsAfter > repsBefore) {
                    // Play a fun beep when rep counter updates.
                    ToneGenerator tg = new ToneGenerator(AudioManager.STREAM_NOTIFICATION, 100);
                    tg.startTone(ToneGenerator.TONE_PROP_BEEP);
                    lastRepResult = String.format(
                            Locale.US, "%s : %d reps", repCounter.getClassName(), repsAfter);
                    break;
                }
            }
            result.add(lastRepResult);
        }

        // Add maxConfidence class of current frame to result if pose is found.
        // 如果pose不为空（即找到了正确的姿势），则将最大置信度的类添加到结果列表中（使用getClassConfidence()方法和getMaxConfidenceClass()方法）。
        // 最后，返回所有结果列表。
        if (!pose.getAllPoseLandmarks().isEmpty()) {
            String maxConfidenceClass = classification.getMaxConfidenceClass();
            String maxConfidenceClassResult = String.format(
                    Locale.US,
                    "%s : %.2f confidence",
                    maxConfidenceClass,
                    classification.getClassConfidence(maxConfidenceClass)
                            / poseClassifier.confidenceRange());
            result.add(maxConfidenceClassResult);
        }

        return result;
    }

}
