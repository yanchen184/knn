package com.yc.kmeans.kmeans;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 加權KNN的輔助工具類，實現評估和計算相關功能
 */
@Slf4j
public class WeightedKNNUtils {

    /**
     * 為WeightedKNNClassifier執行評估操作
     * 
     * @param classifier 待評估的分類器
     * @param folds 交叉驗證的折數
     * @param maxTestSamplesPerFold 每折最大測試樣本數
     * @return 評估結果
     */
    public static EvaluationResult evaluateModel(WeightedKNNClassifier classifier, 
                                                int folds, 
                                                int maxTestSamplesPerFold) {
        if (!classifier.isTrained()) {
            throw new IllegalStateException("分類器尚未訓練");
        }
        
        // 獲取訓練數據和標籤
        List<LabeledPoint> trainingData = new ArrayList<>(classifier.getTrainingData());
        Map<String, List<LabeledPoint>> labelToPointsMap = classifier.getLabelToPointsMap();
        
        log.info("準備評估模型，訓練數據大小 = {}", trainingData.size());
        if (trainingData.size() < folds) {
            throw new IllegalStateException("訓練數據不足以進行指定折數的交叉驗證");
        }

        // 初始化評估結果
        EvaluationResult result = new EvaluationResult();

        // 計算每個類別的樣本數
        Map<String, Integer> classCounts = new HashMap<>();
        for (Map.Entry<String, List<LabeledPoint>> entry : labelToPointsMap.entrySet()) {
            classCounts.put(entry.getKey(), entry.getValue().size());
        }
        result.setClassCounts(classCounts);

        // 初始化混淆矩陣
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        for (String actualLabel : labelToPointsMap.keySet()) {
            confusionMatrix.put(actualLabel, new HashMap<>());
            for (String predictedLabel : labelToPointsMap.keySet()) {
                confusionMatrix.get(actualLabel).put(predictedLabel, 0);
            }
        }

        // 打亂訓練數據
        log.info("打亂數據進行交叉驗證");
        List<LabeledPoint> shuffledData = new ArrayList<>(trainingData);
        Collections.shuffle(shuffledData);

        // 計算每折的大小
        int foldSize = trainingData.size() / folds;

        int totalCorrect = 0;
        int totalSamples = 0;

        // 相關係數計算準備
        double sumActualY = 0;
        double sumPredictedY = 0;
        double sumActualYSquared = 0;
        double sumPredictedYSquared = 0;
        double sumActualPredictedY = 0;

        // 創建數字與標籤的映射（用於R2計算）
        Map<String, Integer> labelToNumber = new HashMap<>();
        int labelNumber = 0;
        for (String label : labelToPointsMap.keySet()) {
            labelToNumber.put(label, labelNumber++);
        }

        // 交叉驗證
        for (int i = 0; i < folds; i++) {
            int startIdx = i * foldSize;
            int endIdx = (i == folds - 1) ? shuffledData.size() : (i + 1) * foldSize;

            // 提取測試集 - 限制測試集大小
            List<LabeledPoint> fullTestFold = new ArrayList<>(shuffledData.subList(startIdx, endIdx));
            List<LabeledPoint> testFold;

            if (fullTestFold.size() > maxTestSamplesPerFold) {
                // 從完整測試集中隨機抽樣
                Collections.shuffle(fullTestFold);
                testFold = new ArrayList<>(fullTestFold.subList(0, maxTestSamplesPerFold));
                log.info("縮減測試集大小從 {} 到 {}", fullTestFold.size(), testFold.size());
            } else {
                testFold = fullTestFold;
            }

            // 提取訓練集（排除測試集）
            List<LabeledPoint> trainFold = new ArrayList<>();
            for (int j = 0; j < shuffledData.size(); j++) {
                if (j < startIdx || j >= endIdx) {
                    trainFold.add(shuffledData.get(j));
                }
            }

            // 創建並訓練臨時分類器
            WeightedKNNClassifier tempClassifier = new WeightedKNNClassifier(classifier.getK());
            tempClassifier.setUseClassWeights(classifier.isUseClassWeights());
            tempClassifier.setEpsilon(classifier.getEpsilon());
            tempClassifier.train(trainFold);

            // 在測試集上評估
            int foldCorrect = 0;
            for (LabeledPoint testPoint : testFold) {
                foldCorrect++;
                if(foldCorrect % 1000 == 0){
                    log.info("已處理測試樣本數 = {}", foldCorrect);
                }
                String actualLabel = testPoint.getLabel();
                String predictedLabel = tempClassifier.predict(testPoint.getFeatures());

                // 更新混淆矩陣
                confusionMatrix.get(actualLabel).put(
                        predictedLabel,
                        confusionMatrix.get(actualLabel).get(predictedLabel) + 1
                );

                // 計算正確預測數
                if (predictedLabel.equals(actualLabel)) {
                    totalCorrect++;
                }

                // 為R2計算收集數據
                int actualY = labelToNumber.get(actualLabel);
                int predictedY = labelToNumber.get(predictedLabel);

                sumActualY += actualY;
                sumPredictedY += predictedY;
                sumActualYSquared += actualY * actualY;
                sumPredictedYSquared += predictedY * predictedY;
                sumActualPredictedY += actualY * predictedY;

                totalSamples++;
            }
        }

        // 計算準確率
        double accuracy = (double) totalCorrect / totalSamples;
        result.setAccuracy(accuracy);

        // 計算精確率、召回率和F1分數（多類別的宏平均）
        double totalPrecision = 0;
        double totalRecall = 0;
        int classCount = labelToPointsMap.size();

        for (String label : labelToPointsMap.keySet()) {
            int truePositives = confusionMatrix.get(label).get(label);

            // 計算假陽性（其他類被預測為此類）
            int falsePositives = 0;
            for (String otherLabel : labelToPointsMap.keySet()) {
                if (!otherLabel.equals(label)) {
                    falsePositives += confusionMatrix.get(otherLabel).get(label);
                }
            }

            // 計算假陰性（此類被預測為其他類）
            int falseNegatives = 0;
            for (String otherLabel : labelToPointsMap.keySet()) {
                if (!otherLabel.equals(label)) {
                    falseNegatives += confusionMatrix.get(label).get(otherLabel);
                }
            }

            // 計算此類的精確率和召回率
            double precision = (truePositives + falsePositives) > 0 ?
                    (double) truePositives / (truePositives + falsePositives) : 0;
            double recall = (truePositives + falseNegatives) > 0 ?
                    (double) truePositives / (truePositives + falseNegatives) : 0;

            totalPrecision += precision;
            totalRecall += recall;
        }

        // 計算宏平均
        double avgPrecision = totalPrecision / classCount;
        double avgRecall = totalRecall / classCount;

        result.setPrecision(avgPrecision);
        result.setRecall(avgRecall);

        // 計算F1分數
        double f1Score = (avgPrecision + avgRecall) > 0 ?
                2 * avgPrecision * avgRecall / (avgPrecision + avgRecall) : 0;
        result.setF1Score(f1Score);

        // 計算R2分數（決定係數）
        double numerator = totalSamples * sumActualPredictedY - sumActualY * sumPredictedY;
        double denomPart1 = totalSamples * sumActualYSquared - sumActualY * sumActualY;
        double denomPart2 = totalSamples * sumPredictedYSquared - sumPredictedY * sumPredictedY;

        double r = denomPart1 > 0 && denomPart2 > 0 ?
                numerator / Math.sqrt(denomPart1 * denomPart2) : 0;
        double r2 = r * r;

        result.setR2Score(r2);
        result.setConfusionMatrix(confusionMatrix);

        return result;
    }
    
    /**
     * 計算歐氏距離
     *
     * @param a 第一個點的特徵向量
     * @param b 第二個點的特徵向量
     * @return 兩點之間的歐氏距離
     */
    public static double calculateDistance(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("特徵維度不匹配");
        }

        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}
