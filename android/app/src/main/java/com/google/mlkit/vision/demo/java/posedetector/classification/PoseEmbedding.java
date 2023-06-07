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

import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.average;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.l2Norm2D;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiplyAll;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtract;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtractAll;

import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.PoseLandmark;

import java.util.ArrayList;
import java.util.List;

/**
 * Generates embedding for given list of Pose landmarks.
 */
public class PoseEmbedding {
    // Multiplier to apply to the torso to get minimal body size. Picked this by experimentation.
    // 这是一个常量，用于将躯干长度乘以一个系数，得到最小的身体尺寸。经过实验，选取了系数为 2.5f。
    private static final float TORSO_MULTIPLIER = 2.5f;

    // 接受一个 List<PointF3D> 类型的 landmarks 参数，将其normalize后传递给 getEmbedding 方法，最终返回获取到的嵌入向量列表。
    public static List<PointF3D> getPoseEmbedding(List<PointF3D> landmarks) {
        List<PointF3D> normalizedLandmarks = normalize(landmarks);
        return getEmbedding(normalizedLandmarks);
    }

    // 将传入的姿势关键点列表进行标准化。
    // 首先复制输入列表 landmarks 的所有元素到一个新列表 normalizedLandmarks；
    // 然后对其进行平移和缩放，以将其放置在同一大小的空间中。
    // 平移和缩放过程通过取两髋关节的中心作为原点，并对关键点进行相应的变换来完成。
    private static List<PointF3D> normalize(List<PointF3D> landmarks) {
        List<PointF3D> normalizedLandmarks = new ArrayList<>(landmarks);
        // Normalize translation.
        PointF3D center = average(
                landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));
        subtractAll(center, normalizedLandmarks);

        // Normalize scale.
        multiplyAll(normalizedLandmarks, 1 / getPoseSize(normalizedLandmarks));
        // Multiplication by 100 is not required, but makes it easier to debug.
        multiplyAll(normalizedLandmarks, 100);
        return normalizedLandmarks;
    }

    // Translation normalization should've been done prior to calling this method.
    // 用于计算姿势的大小。
    // 该方法根据传入的关键点列表 landmarks 来计算并返回一个浮点数值，该值表示测试人员的身体大小。
    // 它通过使用髋关节和肩膀之间的距离来计算躯干长度，并将其乘以一个常量系数 TORSO_MULTIPLIER 来得到一个基本的最小身体尺寸。
    // 然后遍历所有的关键点，找出距离 hipsCenter（髋关节中心）最远的那个关键点，并返回代表实际身体尺寸的 maxDistance。
    // 还需要注意的是，该方法只使用二维关键点计算姿势大小，因为作者在实验中发现，使用 z 坐标并没有帮助，您可以根据您的实际情况进行调整。
    private static float getPoseSize(List<PointF3D> landmarks) {
        // Note: This approach uses only 2D landmarks to compute pose size as using Z wasn't helpful
        // in our experimentation but you're welcome to tweak.
        PointF3D hipsCenter = average(
                landmarks.get(PoseLandmark.LEFT_HIP), landmarks.get(PoseLandmark.RIGHT_HIP));

        PointF3D shouldersCenter = average(
                landmarks.get(PoseLandmark.LEFT_SHOULDER),
                landmarks.get(PoseLandmark.RIGHT_SHOULDER));

        float torsoSize = l2Norm2D(subtract(hipsCenter, shouldersCenter));

        float maxDistance = torsoSize * TORSO_MULTIPLIER;
        // torsoSize * TORSO_MULTIPLIER is the floor we want based on experimentation but actual size
        // can be bigger for a given pose depending on extension of limbs etc so we calculate that.
        for (PointF3D landmark : landmarks) {
            float distance = l2Norm2D(subtract(hipsCenter, landmark));
            if (distance > maxDistance) {
                maxDistance = distance;
            }
        }
        return maxDistance;
    }

    // 用于从三维关键点列表 lm 中提取姿势嵌入特征。
    // 该方法遍历多个已经选择好的关键点对之间的距离，使用欧几里得距离计算每个点对之间的距离，并将所有距离组成一个特征向量 embedding。
    // 1.将关键点对按照连线的节点数进行分类，分别是：单节点、两个节点、四个节点和五个节点。
    // 2.对于每个关键点对，通过计算二者之间的欧几里得距离，获取它们之间的距离差量；
    // 3.将所有的距离差量存储在一个新的列表 embedding 中；
    // 4.返回 embedding 列表，其中包含每个关键点对之间的特征向量。
    private static List<PointF3D> getEmbedding(List<PointF3D> lm) {
        List<PointF3D> embedding = new ArrayList<>();

        // We use several pairwise 3D distances to form pose embedding. These were selected
        // based on experimentation for best results with our default pose classes as captued in the
        // pose samples csv. Feel free to play with this and add or remove for your use-cases.

        // We group our distances by number of joints between the pairs.
        // One joint.
        embedding.add(subtract(
                average(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.RIGHT_HIP)),
                average(lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.RIGHT_SHOULDER))
        ));

        embedding.add(subtract(
                lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_ELBOW)));
        embedding.add(subtract(
                lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_ELBOW)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_ELBOW), lm.get(PoseLandmark.LEFT_WRIST)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_ELBOW), lm.get(PoseLandmark.RIGHT_WRIST)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_KNEE)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_KNEE)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_KNEE), lm.get(PoseLandmark.LEFT_ANKLE)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_KNEE), lm.get(PoseLandmark.RIGHT_ANKLE)));

        // Two joints.
        embedding.add(subtract(
                lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_WRIST)));
        embedding.add(subtract(
                lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_WRIST)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_ANKLE)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_ANKLE)));

        // Four joints.
        embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_WRIST)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_WRIST)));

        // Five joints.
        embedding.add(subtract(
                lm.get(PoseLandmark.LEFT_SHOULDER), lm.get(PoseLandmark.LEFT_ANKLE)));
        embedding.add(subtract(
                lm.get(PoseLandmark.RIGHT_SHOULDER), lm.get(PoseLandmark.RIGHT_ANKLE)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_HIP), lm.get(PoseLandmark.LEFT_WRIST)));
        embedding.add(subtract(lm.get(PoseLandmark.RIGHT_HIP), lm.get(PoseLandmark.RIGHT_WRIST)));

        // Cross body.
        embedding.add(subtract(lm.get(PoseLandmark.LEFT_ELBOW), lm.get(PoseLandmark.RIGHT_ELBOW)));
        embedding.add(subtract(lm.get(PoseLandmark.LEFT_KNEE), lm.get(PoseLandmark.RIGHT_KNEE)));

        embedding.add(subtract(lm.get(PoseLandmark.LEFT_WRIST), lm.get(PoseLandmark.RIGHT_WRIST)));
        embedding.add(subtract(lm.get(PoseLandmark.LEFT_ANKLE), lm.get(PoseLandmark.RIGHT_ANKLE)));

        return embedding;
    }

    private PoseEmbedding() {
    }
}
