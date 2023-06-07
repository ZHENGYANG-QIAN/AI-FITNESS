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

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static java.util.Collections.max;

/**
 * Represents Pose classification result as outputted by {@link PoseClassifier}. Can be manipulated.
 */
public class ClassificationResult {
    // 一个映射表，用于存储分类名称和对应的置信度值。
    // 名为classConfidences的Map，用于存储一个PoseSample被分类为某个类别的置信度（或可信度）得分。
    // 该Map的键是类别名称，值是此类别在经过前K个最相似样本过滤后，与当前姿势样本匹配的数目，值域为[0,K]。
    // 值可以是浮点数，使用EMA平滑后取整，并表示给定姿势样本属于此类别的可能性大小。
    // 该分数越大，则表明该样本可能属于此类别的可能性就越高。
    // 在算法分类结束后，我们可以通过查看classConfidences中每个类别的得分情况，来确定当前姿势样本最有可能所属的类别。如果一个类别的得分远高于其他类别，则我们可以判断当前样本属于该类别。
    // For an entry in this map, the key is the class name, and the value is how many times this class
    // appears in the top K nearest neighbors. The value is in range [0, K] and could be a float after
    // EMA smoothing. We use this number to represent the confidence of a pose being in this class.
    private final Map<String, Float> classConfidences;

    public ClassificationResult() {
        classConfidences = new HashMap<>();
    }

    // 获取所有分类名称，返回一个 Set 集合。
    public Set<String> getAllClasses() {
        return classConfidences.keySet();
    }

    // 根据给定的分类名称获取对应的置信度值，如果没有该分类名称则返回 0。
    public float getClassConfidence(String className) {
        return classConfidences.containsKey(className) ? classConfidences.get(className) : 0;
    }

    // 获取置信度值最大的分类名称。
    public String getMaxConfidenceClass() {
        return max(
                classConfidences.entrySet(),
                (entry1, entry2) -> (int) (entry1.getValue() - entry2.getValue()))
                .getKey();
    }

    // 将给定分类名称的置信度值加一，如果不存在该分类名称则新建。
    public void incrementClassConfidence(String className) {
        classConfidences.put(className,
                classConfidences.containsKey(className) ? classConfidences.get(className) + 1 : 1);
    }

    // 设置给定分类名称的置信度值为指定的值。
    public void putClassConfidence(String className, float confidence) {
        classConfidences.put(className, confidence);
    }
}
