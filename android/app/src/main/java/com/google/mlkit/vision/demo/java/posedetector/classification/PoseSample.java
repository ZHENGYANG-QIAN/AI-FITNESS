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

import android.util.Log;

import com.google.common.base.Splitter;
import com.google.mlkit.vision.common.PointF3D;

import java.util.ArrayList;
import java.util.List;

/**
 * Reads Pose samples from a csv file.
 */
public class PoseSample {
    private static final String TAG = "PoseSample";
    private static final int NUM_LANDMARKS = 33;
    private static final int NUM_DIMS = 3;

    // 图片源文件名，第1列
    private final String name;
    // 动作类名，第2列
    private final String className;
    // 包含关键点（landmarks）的列表
    private final List<PointF3D> embedding;

    public PoseSample(String name, String className, List<PointF3D> landmarks) {
        this.name = name;
        this.className = className;
        this.embedding = PoseEmbedding.getPoseEmbedding(landmarks);
    }

    public String getName() {
        return name;
    }

    public String getClassName() {
        return className;
    }

    public List<PointF3D> getEmbedding() {
        return embedding;
    }

    // 用于将csv格式的字符串转换成PoseSample(name, className, landmarks)对象。
    // 该方法接受两个参数：csvLine和separator（分隔符）。
    // csvLine是表示样本信息的字符串，separator是字符串中用于分隔不同元素的字符。
    public static PoseSample getPoseSample(String csvLine, String separator) {
        // 该方法首先使用Splitter.onPattern()方法将csvLine字符串转换为tokens列表。
        List<String> tokens = Splitter.onPattern(separator).splitToList(csvLine);
        // Format is expected to be Name,Class,X1,Y1,Z1,X2,Y2,Z2...
        // + 2 is for Name & Class.
        // tokens的第一个元素是名称，第二个元素是类别名称，接下来的元素是关键点的x、y、z坐标，每个点有三个独立的坐标轴。
        // 如果tokens数目不等于（NUM_LANDMARKS * NUM_DIMS）+2，则会记录日志并返回空值。
        if (tokens.size() != (NUM_LANDMARKS * NUM_DIMS) + 2) {
            Log.e(TAG, "Invalid number of tokens for PoseSample");
            return null;
        }
        String name = tokens.get(0);
        String className = tokens.get(1);
        List<PointF3D> landmarks = new ArrayList<>();
        // Read from the third token, first 2 tokens are name and class.
        // 接着，使用PointF3D.from()方法将每个坐标解析成浮点数，并将其添加到landmarks列表中。
        for (int i = 2; i < tokens.size(); i += NUM_DIMS) {
            try {
                landmarks.add(
                        PointF3D.from(
                                Float.parseFloat(tokens.get(i)),
                                Float.parseFloat(tokens.get(i + 1)),
                                Float.parseFloat(tokens.get(i + 2))));
            } catch (NullPointerException | NumberFormatException e) {
                Log.e(TAG, "Invalid value " + tokens.get(i) + " for landmark position.");
                return null;
            }
        }
        // 最后，使用名称、类别名称和landmarks列表创建一个新的PoseSample对象。
        return new PoseSample(name, className, landmarks);
    }
}
