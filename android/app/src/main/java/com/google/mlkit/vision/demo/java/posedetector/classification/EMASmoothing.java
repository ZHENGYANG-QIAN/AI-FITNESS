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

import android.os.SystemClock;

import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * Runs EMA smoothing over a window with given stream of pose classification results.
 */
// 用于对给定的姿势分类结果进行指数移动平均（EMA）平滑。该算法通过维护一个窗口并针对该窗口上的一系列值进行平滑化。
public class EMASmoothing {
    // 默认窗口大小，初值设为10。
    private static final int DEFAULT_WINDOW_SIZE = 10;
    // 指数平滑系数 alpha 的默认值，初值设为 0.2。
    private static final float DEFAULT_ALPHA = 0.2f;
    // 时间阈值，如果两个输入之间的时间间隔超过这个阈值就会清除窗口，初值设为 100ms。

    private static final long RESET_THRESHOLD_MS = 100;

    // 窗口的大小，初始化时取默认值 DEFAULT_WINDOW_SIZE。
    private final int windowSize;
    // 指数平滑系数，初始化时取默认值 DEFAULT_ALPHA。
    private final float alpha;
    // This is a window of {@link ClassificationResult}s as outputted by the {@link PoseClassifier}.
    // We run smoothing over this window of size {@link windowSize}.
    // 用于存放 ClassificationResult 对象的 window，类型为 Deque<ClassificationResult>。
    private final Deque<ClassificationResult> window;

    // 记录上次输入的时间戳，用于判断输入时间间隔是否超过 RESET_THRESHOLD_MS。
    private long lastInputMs;

    public EMASmoothing() {
        this(DEFAULT_WINDOW_SIZE, DEFAULT_ALPHA);
    }

    public EMASmoothing(int windowSize, float alpha) {
        this.windowSize = windowSize;
        this.alpha = alpha;
        this.window = new LinkedBlockingDeque<>(windowSize);
    }

    // 接受一个 ClassificationResult 对象
    // 返回一个经过 EMA 平滑过的 ClassificationResult 对象。
    public ClassificationResult getSmoothedResult(ClassificationResult classificationResult) {
        // 如果两个输入之间的时间间隔超过阈值 RESET_THRESHOLD_MS，则清除窗口。
        // Resets memory if the input is too far away from the previous one in time.
        long nowMs = SystemClock.elapsedRealtime();
        if (nowMs - lastInputMs > RESET_THRESHOLD_MS) {
            window.clear();
        }
        lastInputMs = nowMs;

        // 向窗口前面添加新的 classificationResult 对象，并移除窗口后面的老对象，保持窗口大小 windowSize 一致。
        // If we are at window size, remove the last (oldest) result.
        if (window.size() == windowSize) {
            window.pollLast();
        }
        // Insert at the beginning of the window.
        window.addFirst(classificationResult);

        Set<String> allClasses = new HashSet<>();
        for (ClassificationResult result : window) {
            allClasses.addAll(result.getAllClasses());
        }

        ClassificationResult smoothedResult = new ClassificationResult();

        // 对于窗口中的每个分类名称，针对该分类的所有 windowSize 个分类结果，计算该分类的指数移动平均值，存储在输出 result 对象中。
        for (String className : allClasses) {
            // 指数平滑的系数 factor
            float factor = 1;
            // 系数加权的置信度的和 topSum
            float topSum = 0;
            // 系数总和 bottomSum
            float bottomSum = 0;
            for (ClassificationResult result : window) {
                float value = result.getClassConfidence(className);

                topSum += factor * value;
                bottomSum += factor;

                factor = (float) (factor * (1.0 - alpha));
            }
            // 计算当前分类名称的平滑结果 topSum/bottomSum，并将其存储在输出变量 smoothedResult 中。
            smoothedResult.putClassConfidence(className, topSum / bottomSum);
        }

        return smoothedResult;
    }
}
