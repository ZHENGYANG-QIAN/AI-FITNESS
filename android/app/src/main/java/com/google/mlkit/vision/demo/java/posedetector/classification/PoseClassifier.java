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

import static com.google.mlkit.vision.demo.java.posedetector.classification.PoseEmbedding.getPoseEmbedding;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.maxAbs;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiply;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.multiplyAll;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.subtract;
import static com.google.mlkit.vision.demo.java.posedetector.classification.Utils.sumAbs;
import static java.lang.Math.max;
import static java.lang.Math.min;

import android.util.Pair;

import com.google.mlkit.vision.common.PointF3D;
import com.google.mlkit.vision.pose.Pose;
import com.google.mlkit.vision.pose.PoseLandmark;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
 * Classifies {link Pose} based on given {@link PoseSample}s.
 *
 * <p>Inspired by K-Nearest Neighbors Algorithm with outlier filtering.
 * https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
 */
// 一个基于KNN算法的姿势分类器，用于将给定的姿势分类为不同的活动类别。
// 它使用姿势嵌入向量作为特征，并计算给定姿势与每个已知姿势样本的相似度。
// 在相似度计算中，该算法忽略了一些偏离度较高的样本，并使用加权的X，Y和Z轴来更好地区分不同的姿势类型。
public class PoseClassifier {
    private static final String TAG = "PoseClassifier";
    private static final int MAX_DISTANCE_TOP_K = 30;
    private static final int MEAN_DISTANCE_TOP_K = 10;
    // Note Z has a lower weight as it is generally less accurate than X & Y.
    private static final PointF3D AXES_WEIGHTS = PointF3D.from(1, 1, 0.2f);

    // poseSamples是一个包含多个PoseSample的列表，每个PoseSample代表一个已知的姿势样本
    private final List<PoseSample> poseSamples;
    // maxDistanceTopK和meanDistanceTopK分别是用于分类的两个K值，它们限制了在计算相似度时保留的最大样本数量。
    // 在相似度计算中：
    // 先选出距离待分类姿势最近的MAX_DISTANCE_TOP_K个姿势样本，
    // 然后再从这些样本中选出平均距离最近的MEAN_DISTANCE_TOP_K个样本，
    // 从而获得较准确的分类结果。
    private final int maxDistanceTopK;
    private final int meanDistanceTopK;
    // 三维向量，用于加权不同轴上的姿势特征，其中z轴的权重较小，因为它相比x和y轴通常不准确。
    private final PointF3D axesWeights;

    public PoseClassifier(List<PoseSample> poseSamples) {
        this(poseSamples, MAX_DISTANCE_TOP_K, MEAN_DISTANCE_TOP_K, AXES_WEIGHTS);
    }

    public PoseClassifier(List<PoseSample> poseSamples, int maxDistanceTopK,
                          int meanDistanceTopK, PointF3D axesWeights) {
        this.poseSamples = poseSamples;
        this.maxDistanceTopK = maxDistanceTopK;
        this.meanDistanceTopK = meanDistanceTopK;
        this.axesWeights = axesWeights;
    }

    /**
     * Returns the max range of confidence values.
     *
     * <p><Since we calculate confidence by counting {@link PoseSample}s that survived
     * outlier-filtering by maxDistanceTopK and meanDistanceTopK, this range is the minimum of two.
     */
    // 返回分类器算法能够提供的最大置信度范围。
    // 在该算法中，每个已知的姿势样本都被赋予了一个类别标签，
    // 然后当待分类姿势与样本进行相似度比较时，算法会统计距离（相似度）最近的MAX_DISTANCE_TOP_K个样本和平均距离最近的MEAN_DISTANCE_TOP_K个样本，
    // 针对这些样本进行加权投票，从而确定待分类姿势所属的类别。
    // 因此，该算法能提供的最大置信度范围为MAX_DISTANCE_TOP_K与MEAN_DISTANCE_TOP_K中的较小值。
    public int confidenceRange() {
        return min(maxDistanceTopK, meanDistanceTopK);
    }

    // 从一个姿势(Pose)对象中提取所有关键点的三维坐标信息。
    // 该方法将遍历Pose对象中的所有PoseLandmark对象，每个PoseLandmark对象代表一个姿势关键点，然后将该关键点的三维坐标信息(Position3D)添加到一个列表中。
    // 最终，该方法会返回包含所有姿势关键点的三维坐标信息的列表(List<PointF3D>)。这里假设传入的Pose参数已经检测到了所有的关键点。
    private static List<PointF3D> extractPoseLandmarks(Pose pose) {
        List<PointF3D> landmarks = new ArrayList<>();
        for (PoseLandmark poseLandmark : pose.getAllPoseLandmarks()) {
            landmarks.add(poseLandmark.getPosition3D());
        }
        return landmarks;
    }

    public ClassificationResult classify(Pose pose) {
        return classify(extractPoseLandmarks(pose));
    }

    // 这是一个姿势分类器的Java代码，它可以使用给定的姿势关键点的列表对其进行分类。该算法分为两个阶段：
    // 1.首先，通过最大距离选择前K个样本，以去除与给定姿势几乎相同但可能有几个关节弯曲方向不同的样本。
    // 2.然后，通过平均距离选择前K个样本，以选取平均距离最近的样本。在去除异常值后，我们选择最接近平均值的样本。
    // 在代码中，我们使用两个优先队列（maxDistances和meanDistances）来保存最大距离和平均距离。每个队列都包含Pair对象，该对象存储着一个PoseSample对象和浮点型的距离值。PoseSample对象是一个拥有嵌入向量和类名的类，表示一个已知的姿势样本。
    // 在第一个阶段中，我们计算给定姿势与所有已知样本之间的最大距离，并将其添加到maxDistances队列中。我们只保留前maxDistanceTopK个最小的距离值，因为它们是最相似的样本。
    // 在第二个阶段中，我们计算这些最相似的样本与给定姿势之间的平均距离，然后将它们添加到meanDistances队列中。我们只保留前meanDistanceTopK个平均距离最小的样本，因为它们是最适合的样本，我们使用它们来确定分类结果。
    // 最后，我们对每个样本计算一个类别置信度，然后返回ClassificationResult对象，该对象存储了每个类别的置信度分数。
    public ClassificationResult classify(List<PointF3D> landmarks) {
        ClassificationResult result = new ClassificationResult();
        // Return early if no landmarks detected.
        if (landmarks.isEmpty()) {
            return result;
        }

        // 将关键点坐标进行翻转，以使得算法在水平镜像情况下保持不变。
        // 这个翻转关键点的过程可以使算法具有更好的鲁棒性和泛化能力，因为它增加了算法对不同人体侧面的适应性。
        // We do flipping on X-axis so we are horizontal (mirror) invariant.
        List<PointF3D> flippedLandmarks = new ArrayList<>(landmarks);
        // 将X轴坐标取反（乘以-1），同时保持Y轴和Z轴坐标不变，得到新的翻转坐标。
        multiplyAll(flippedLandmarks, PointF3D.from(-1, 1, 1));

        // 使用原始关键点列表和翻转后的关键点列表来计算它们的嵌入向量（embedding）。
        List<PointF3D> embedding = getPoseEmbedding(landmarks);
        List<PointF3D> flippedEmbedding = getPoseEmbedding(flippedLandmarks);


        // Classification is done in two stages:
        //  * First we pick top-K samples by MAX distance. It allows to remove samples that are almost
        //    the same as given pose, but maybe has few joints bent in the other direction.
        //  * Then we pick top-K samples by MEAN distance. After outliers are removed, we pick samples
        //    that are closest by average.

        // 通过计算样本与数据集中每个参考姿势嵌入向量的欧氏距离，选择最接近的前k个姿势。
        // 其中，k是我们在初始化阶段定义的一个变量maxDistanceTopK。

        // 首先，使用一个优先队列maxDistances来保存最接近的top K个姿势，
        // 容器中的每个元素是一个PoseSample和对应的距离值，即Pair<PoseSample, Float>。
        // 容器中距离值从大到小排列，使得排在队列前面的元素具有更大的距离值，表示它们与当前样本的距离较远。
        // Keeps max distance on top so we can pop it when top_k size is reached.
        PriorityQueue<Pair<PoseSample, Float>> maxDistances = new PriorityQueue<>(
                maxDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
        // 接下来，我们遍历数据集样本，并计算其与当前姿势样本的距离。
        // maxDistances队列排序当前所有的结果，该队列的大小限制为maxDistanceTopK，即只保留前K个距离最小的样本。
        // Retrieve top K poseSamples by least distance to remove outliers.
        for (PoseSample poseSample : poseSamples) {
            List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

            float originalMax = 0;
            float flippedMax = 0;
            for (int i = 0; i < embedding.size(); i++) {
                originalMax =
                        max(
                                originalMax,
                                maxAbs(multiply(subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights)));
                flippedMax =
                        max(
                                flippedMax,
                                maxAbs(
                                        multiply(
                                                subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights)));
            }
            // 将原始图像和翻转图像的样本距离两者中的最大值设为max distance
            // Set the max distance as min of original and flipped max distance.
            maxDistances.add(new Pair<>(poseSample, min(originalMax, flippedMax)));
            // 当新的样本距离比最大距离还大时，我们会将其从队列中弹出，从而保证队列中始终存储着距离最小的K个样本。
            // We only want to retain top n so pop the highest distance.
            if (maxDistances.size() > maxDistanceTopK) {
                maxDistances.poll();
            }
        }

        // 在已经计算出的距离最小的k个参考姿势中，进一步筛选出平均距离最小的前n个姿势样本，并将它们用于分类当前姿势。

        // 首先，定义一个优先队列meanDistances，用于存储根据平均距离排序后的前n个参考姿势样本。
        // 由于我们要将平均距离最大的样本排在队列的前面，因此我们需要使用负的比较器对象（-Float.compare(o1.second, o2.second)）对队列进行初始化。
        // Keeps higher mean distances on top so we can pop it when top_k size is reached.
        PriorityQueue<Pair<PoseSample, Float>> meanDistances = new PriorityQueue<>(
                meanDistanceTopK, (o1, o2) -> -Float.compare(o1.second, o2.second));
        // 然后，对于maxDistances队列中的每个元素，我们根据其包含的参考姿势poseSample和当前姿势样本的嵌入向量embedding以及flippedEmbedding，分别计算它们之间的平均距离。
        // 具体来说，我们通过遍历embedding中的每个元素，计算它们与poseSample.getEmbedding()中对应元素之间的距离，并使用axesWeights进行多维加权。
        // 其中，originalSum和flippedSum分别记录了原始样本和反转后样本的总距离。
        // Retrive top K poseSamples by least mean distance to remove outliers.
        for (Pair<PoseSample, Float> sampleDistances : maxDistances) {
            PoseSample poseSample = sampleDistances.first;
            List<PointF3D> sampleEmbedding = poseSample.getEmbedding();

            float originalSum = 0;
            float flippedSum = 0;
            for (int i = 0; i < embedding.size(); i++) {
                originalSum += sumAbs(multiply(
                        subtract(embedding.get(i), sampleEmbedding.get(i)), axesWeights));
                flippedSum += sumAbs(
                        multiply(subtract(flippedEmbedding.get(i), sampleEmbedding.get(i)), axesWeights));
            }
            // Set the mean distance as min of original and flipped mean distances.
            float meanDistance = min(originalSum, flippedSum) / (embedding.size() * 2);
            // 只要计算出了平均距离，我们就可以将当前参考姿势poseSample与平均距离meanDistance组成的pair添加到meanDistances队列中。
            // 与maxDistances队列相同，我们同样需要保证队列中存储的元素数量不超过预设的值meanDistanceTopK，
            // 因此在队列元素数量超过meanDistanceTopK后，就会把平均距离最大的元素弹出队列，以确保队列中始终只保存前n个平均距离最小的姿势样本。
            meanDistances.add(new Pair<>(poseSample, meanDistance));
            // We only want to retain top k so pop the highest mean distance.
            if (meanDistances.size() > meanDistanceTopK) {
                meanDistances.poll();
            }
        }

        // 增加姿势识别结果result中某一类别的置信度（confidence）。
        for (Pair<PoseSample, Float> sampleDistances : meanDistances) {
            String className = sampleDistances.first.getClassName();
            result.incrementClassConfidence(className);
        }

        return result;
    }
}
