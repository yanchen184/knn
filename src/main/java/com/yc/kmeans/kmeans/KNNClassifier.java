package com.yc.kmeans.kmeans;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * K最近鄰（KNN）分類器實現
 * 基於歐氏距離計算最近鄰，並使用多數投票進行分類
 */
@Slf4j
public class KNNClassifier implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    private final List<LabeledPoint> trainingData;
    /**
     * -- GETTER --
     *  獲取k值
     *
     *
     * -- SETTER --
     *  設置k值
     *
     @return k值
      * @param k 新的k值
     */
    @Setter
    @Getter
    private int k;  // 最近鄰居數量
    private boolean isTrained = false;
    /**
     * -- GETTER --
     *  獲取最後的評估結果
     *
     * @return 評估結果
     */
    @Getter
    private EvaluationResult lastEvaluation;
    private final Set<String> uniqueLabels; // 存儲所有唯一標籤
    private final Map<String, List<LabeledPoint>> labelToPointsMap; // 按標籤存儲訓練數據的映射

    /**
     * 構造函數
     *
     * @param k 最近鄰居數量
     */
    public KNNClassifier(int k) {
        this.k = k;
        this.trainingData = new ArrayList<>();
        this.uniqueLabels = new HashSet<>();
        this.labelToPointsMap = new HashMap<>();
    }

    /**
     * 訓練分類器
     *
     * @param labeledPoints 帶標籤的數據點列表
     */
    public void train(List<LabeledPoint> labeledPoints) {
        if (labeledPoints == null || labeledPoints.isEmpty()) {
            throw new IllegalArgumentException("訓練數據不能為空");
        }

        trainingData.clear();
        uniqueLabels.clear();
        labelToPointsMap.clear();

        // 存儲所有訓練數據點並收集唯一標籤
        for (LabeledPoint point : labeledPoints) {
            trainingData.add(point);
            String label = point.getLabel();
            uniqueLabels.add(label);

            // 為每個標籤建立數據點列表
            labelToPointsMap.computeIfAbsent(label, _ -> new ArrayList<>()).add(point);
        }

        isTrained = true;
        log.info("已完成訓練，共有 {} 個數據點、{} 個類別", trainingData.size(), uniqueLabels.size());
    }

    /**
     * 預測新點的標籤
     *
     * @param x 第一個特徵值
     * @param y 第二個特徵值
     * @return 預測的標籤
     */
    public String predict(double x, double y) {
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }

        double[] features = {x, y};
        return predict(features);
    }

    /**
     * 預測新點的標籤
     *
     * @param features 特徵數組
     * @return 預測的標籤
     */
    public String predict(double[] features) {
        Map<String, Integer> labelCounts = getLabelCounts(features);
        return getMostFrequentLabel(labelCounts);
    }

    /**
     * 從標籤計數中獲取最頻繁的標籤
     * 
     * @param labelCounts 標籤計數映射
     * @return 出現頻率最高的標籤
     */
    private static String getMostFrequentLabel(Map<String, Integer> labelCounts) {
        // 使用 Java 8 Stream 找出最頻繁的標籤
        return labelCounts.entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    /**
     * 獲取特徵向量k個最近鄰中各標籤的出現次數
     *
     * @param features 特徵向量
     * @return 標籤計數映射
     */
    private Map<String, Integer> getLabelCounts(double[] features) {
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }

        // 計算到所有訓練點的距離
        List<DistanceResult> distances = new ArrayList<>();
        for (LabeledPoint point : trainingData) {
            double dist = distance(features, point.getFeatures());
            distances.add(new DistanceResult(dist, point.getLabel()));
        }

        // 根據距離排序
        Collections.sort(distances);

        // 取前k個點，統計最頻繁的標籤
        Map<String, Integer> labelCounts = new HashMap<>();
        int count = Math.min(k, distances.size());

        for (int i = 0; i < count; i++) {
            String label = distances.get(i).getLabel();
            labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
        }
        return labelCounts;
    }

    /**
     * 根據標籤獲取所有該標籤的數據點
     *
     * @param label 標籤
     * @return 具有該標籤的所有數據點列表
     */
    public List<LabeledPoint> getPointsByLabel(String label) {
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }

        if (!uniqueLabels.contains(label)) {
            return Collections.emptyList();
        }

        return new ArrayList<>(labelToPointsMap.get(label));
    }

    /**
     * 獲取標籤到數據點的映射
     *
     * @return 標籤到數據點的映射
     */
    public Map<String, List<LabeledPoint>> getLabelToPointsMap() {
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }

        // 返回一個副本，防止外部修改
        Map<String, List<LabeledPoint>> copy = new HashMap<>();
        for (Map.Entry<String, List<LabeledPoint>> entry : labelToPointsMap.entrySet()) {
            copy.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        return copy;
    }

    /**
     * 評估模型性能
     * 使用交叉驗證
     *
     * @param folds 交叉驗證的折數
     * @param maxTestSamplesPerFold 每折最大測試樣本數
     * @return 評估結果
     */
    public EvaluationResult evaluateModel(int folds, int maxTestSamplesPerFold) {
        log.info("準備評估模型，訓練數據大小 = {}", trainingData.size());
        if (!isTrained || trainingData.size() < folds) {
            throw new IllegalStateException("分類器未訓練或訓練數據不足");
        }

        // 初始化評估結果
        EvaluationResult result = new EvaluationResult();

        // 計算每個類別的樣本數
        Map<String, Integer> classCounts = new HashMap<>();
        for (LabeledPoint point : trainingData) {
            String label = point.getLabel();
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        result.setClassCounts(classCounts);

        // 初始化混淆矩陣
        Map<String, Map<String, Integer>> confusionMatrix = new HashMap<>();
        for (String actualLabel : uniqueLabels) {
            confusionMatrix.put(actualLabel, new HashMap<>());
            for (String predictedLabel : uniqueLabels) {
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
        for (String label : uniqueLabels) {
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
            KNNClassifier tempClassifier = new KNNClassifier(k);
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
        int classCount = uniqueLabels.size();

        for (String label : uniqueLabels) {
            int truePositives = confusionMatrix.get(label).get(label);

            // 計算假陽性（其他類被預測為此類）
            int falsePositives = 0;
            for (String otherLabel : uniqueLabels) {
                if (!otherLabel.equals(label)) {
                    falsePositives += confusionMatrix.get(otherLabel).get(label);
                }
            }

            // 計算假陰性（此類被預測為其他類）
            int falseNegatives = 0;
            for (String otherLabel : uniqueLabels) {
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

        this.lastEvaluation = result;
        return result;
    }

    /**
     * 計算歐氏距離
     *
     * @param a 第一個點的特徵向量
     * @param b 第二個點的特徵向量
     * @return 兩點之間的歐氏距離
     */
    private double distance(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("特徵維度不匹配");
        }

        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * 保存模型到文件
     *
     * @param filePath 文件路徑
     * @throws IOException 如果保存失敗
     */
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
            oos.writeObject(this);
            log.info("模型已保存到: {}", filePath);
        }
    }

    /**
     * 從文件加載模型
     *
     * @param filePath 文件路徑
     * @return KNNClassifier模型實例
     * @throws IOException 如果加載失敗
     * @throws ClassNotFoundException 如果找不到類
     */
    public static KNNClassifier loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            KNNClassifier model = (KNNClassifier) ois.readObject();
            log.info("已從 {} 加載模型，訓練數據大小: {}", filePath, model.getTrainingDataSize());
            return model;
        }
    }

    /**
     * 獲取模型是否已訓練
     *
     * @return 是否已訓練
     */
    public boolean isTrained() {
        return isTrained;
    }

    /**
     * 獲取訓練數據大小
     *
     * @return 訓練數據點數量
     */
    public int getTrainingDataSize() {
        return trainingData.size();
    }

    /**
     * 存儲距離計算結果的不可變類
     * 用於計算和排序距離結果
     */
    private static final class DistanceResult implements Comparable<DistanceResult>, Serializable {
        @Serial
        private static final long serialVersionUID = 1L;
        private final double distance; // 計算得出的距離
        private final String label;   // 數據點的標籤

        /**
         * 建構一個距離計算結果
         *
         * @param distance 計算得出的距離
         * @param label 數據點的標籤
         */
        public DistanceResult(double distance, String label) {
            this.distance = distance;
            this.label = label;
        }

        /**
         * 獲取距離值
         *
         * @return 距離值
         */
        public double getDistance() {
            return distance;
        }

        /**
         * 獲取標籤
         *
         * @return 標籤
         */
        public String getLabel() {
            return label;
        }

        @Override
        public int compareTo(DistanceResult other) {
            return Double.compare(this.distance, other.distance);
        }

        @Override
        public String toString() {
            return "DistanceResult{distance=" + distance + ", label='" + label + "'}"; 
        }
    }
}