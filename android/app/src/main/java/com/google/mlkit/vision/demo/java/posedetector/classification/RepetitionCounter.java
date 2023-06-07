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

/**
 * Counts reps for the give class.
 */
// 一个动作重复次数计数器的 Java 类，用于统计在给定动作类别下的重复次数。
// 主要功能是通过计算当前动作识别结果的置信度，判断是否满足动作进入和退出的阈值，并记录每个动作实例重复的次数。
public class RepetitionCounter {
    // These thresholds can be tuned in conjunction with the Top K values in {@link PoseClassifier}.
    // The default Top K value is 10 so the range here is [0-10].
    // 默认的进入阈值和退出阈值，分别为 6 和 4。这些阈值可以与 PoseClassifier 中设置的 Top K 值一起调整来获得更好的结果。
    private static final float DEFAULT_ENTER_THRESHOLD = 6f;
    private static final float DEFAULT_EXIT_THRESHOLD = 4f;

    private final String className;
    private final float enterThreshold;
    private final float exitThreshold;

    // 该动作实例的重复次数。
    private int numRepeats;
    // 动作是否进入。
    private boolean poseEntered;

    // 默认阈值的构造函数
    public RepetitionCounter(String className) {
        this(className, DEFAULT_ENTER_THRESHOLD, DEFAULT_EXIT_THRESHOLD);
    }

    // 自定义阈值的构造函数
    public RepetitionCounter(String className, float enterThreshold, float exitThreshold) {
        this.className = className;
        this.enterThreshold = enterThreshold;
        this.exitThreshold = exitThreshold;
        numRepeats = 0;
        poseEntered = false;
    }

    /**
     * Adds a new Pose classification result and updates reps for given class.
     *
     * @param classificationResult {link ClassificationResult} of class to confidence values.
     * @return number of reps.
     */
    // 添加新的姿势分类结果并更新给定类别的重复次数。
    public int addClassificationResult(ClassificationResult classificationResult) {
        // 获取目标分类类别的置信度，并存储在 poseConfidence 变量中。
        float poseConfidence = classificationResult.getClassConfidence(className);

        // 如果动作还没有进入，即 poseEntered 为 false：
        // 则检查该分类器的结果是否满足进入阈值：
        // 如果满足，则将 poseEntered 设为 true，并返回当前 numRepeats 的值。
        if (!poseEntered) {
            poseEntered = poseConfidence > enterThreshold;
            return numRepeats;
        }

        // 如果动作已经进入（poseEntered 为 true）：
        // 则检查该分类器的结果是否满足退出阈值：
        // 如果满足，则将 numRepeats 加一，并将 poseEntered 设为 false。
        if (poseConfidence < exitThreshold) {
            numRepeats++;
            poseEntered = false;
        }
        // 返回当前 numRepeats 的值，表示该类别的动作已经被重复执行了多少次。
        return numRepeats;
    }

    public String getClassName() {
        return className;
    }

    public int getNumRepeats() {
        return numRepeats;
    }
}
