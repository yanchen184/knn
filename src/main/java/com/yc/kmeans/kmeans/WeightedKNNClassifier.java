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
 * 加權K最近鄰（KNN）分類器實現
 * 基於歐氏距離計算最近鄰，並使用加權投票進行分類
 * 權重基於距離的反比：1/(distance+epsilon)
 */
@Slf4j
public class WeightedKNNClassifier implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    
    @Getter
    private final List<LabeledPoint> trainingData;
    
    @Setter
    @Getter
    private int k;  // 最近鄰居數量
    
    @Setter
    @Getter
    private double epsilon = 0.00001; // 防止除零錯誤的小值
    
    @Setter
    @Getter
    private boolean useClassWeights = true; // 是否使用類別權重來平衡類別
    
    @Setter
    @Getter
    private double maxClassWeight = 50.0; // 類別權重的最大值
    
    @Setter
    @Getter
    private double distanceWeightFactor = 2.0; // 距離權重因子，增加距離權重的影響
    
    private boolean isTrained = false;
    
    @Getter
    private EvaluationResult lastEvaluation;
    
    private final Set<String> uniqueLabels; // 存儲所有唯一標籤
    private final Map<String, List<LabeledPoint>> labelToPointsMap; // 按標籤存儲訓練數據的映射
    private Map<String, Double> classWeights; // 類別權重，用於處理類別不平衡

    /**
     * 構造函數
     *
     * @param k 最近鄰居數量
     */
    public WeightedKNNClassifier(int k) {
        this.k = k;
        this.trainingData = new ArrayList<>();
        this.uniqueLabels = new HashSet<>();
        this.labelToPointsMap = new HashMap<>();
        this.classWeights = new HashMap<>();
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
        classWeights.clear();

        // 存儲所有訓練數據點並收集唯一標籤
        for (LabeledPoint point : labeledPoints) {
            trainingData.add(point);
            String label = point.getLabel();
            uniqueLabels.add(label);

            // 為每個標籤建立數據點列表
            labelToPointsMap.computeIfAbsent(label, _ -> new ArrayList<>()).add(point);
        }
        
        // 計算類別權重（逆比於類別頻率）
        if (useClassWeights) {
            calculateClassWeights();
        }

        isTrained = true;
        log.info("已完成訓練，共有 {} 個數據點、{} 個類別", trainingData.size(), uniqueLabels.size());
        
        // 輸出每個類別的樣本數和權重
        for (String label : uniqueLabels) {
            int sampleCount = labelToPointsMap.get(label).size();
            double weight = useClassWeights ? classWeights.get(label) : 1.0;
            log.info("類別 '{}': {} 個樣本, 權重: {}", label, sampleCount, weight);
            
            // 對於單樣本類別，提供額外信息
            if (sampleCount == 1 && useClassWeights) {
                log.info("注意: 類別 '{}' 只有1個樣本，將主要依賴距離權重進行預測", label);
            }
        }
    }
    
    /**
     * 計算類別權重，處理類別不平衡問題
     * 權重與類別樣本數成反比：maxCount/count
     */
    private void calculateClassWeights() {
        // 獲取最大類別的樣本數
        int maxCount = 0;
        for (List<LabeledPoint> points : labelToPointsMap.values()) {
            maxCount = Math.max(maxCount, points.size());
        }
        
        // 計算每個類別的權重
        for (String label : uniqueLabels) {
            int count = labelToPointsMap.get(label).size();
            // 對於樣本數極少的類別(如只有1個)，給予更高權重
            // 使用對數函數來平滑極端值，防止單樣本類別獲得過高權重
            double rawWeight = (double) maxCount / count;
            // 使用對數平滑並設定上限
            double weight = Math.min(Math.log10(rawWeight * 10), maxClassWeight);
            classWeights.put(label, weight);
            log.debug("類別 '{}' 原始權重: {}, 平滑後權重: {}", label, rawWeight, weight);
        }
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
        Map<String, Double> labelWeights = getWeightedVotes(features);
        return getMostWeightedLabel(labelWeights);
    }

    /**
     * 從標籤權重中獲取權重最高的標籤
     * 
     * @param labelWeights 標籤權重映射
     * @return 權重最高的標籤
     */
    private static String getMostWeightedLabel(Map<String, Double> labelWeights) {
        // 使用 Java 8 Stream 找出最權重最高的標籤
        return labelWeights.entrySet()
                .stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse(null);
    }

    /**
     * 獲取特徵向量k個最近鄰中各標籤的加權投票
     *
     * @param features 特徵向量
     * @return 標籤權重映射
     */
    private Map<String, Double> getWeightedVotes(double[] features) {
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }

        // 計算到所有訓練點的距離
        List<DistanceResult> distances = new ArrayList<>();
        for (LabeledPoint point : trainingData) {
            double dist = WeightedKNNUtils.calculateDistance(features, point.getFeatures());
            distances.add(new DistanceResult(dist, point.getLabel()));
        }

        // 根據距離排序
        Collections.sort(distances);

        // 取前k個點，計算加權投票
        Map<String, Double> labelWeights = new HashMap<>();
        int count = Math.min(k, distances.size());

        for (int i = 0; i < count; i++) {
            DistanceResult result = distances.get(i);
            String label = result.getLabel();
            double distance = result.getDistance();
            
            // 計算距離的權重: 1/(distance+epsilon)^distanceWeightFactor
            // 增加距離權重因子可以放大近距離的優勢
            double distanceWeight = Math.pow(1.0 / (distance + epsilon), distanceWeightFactor);
            
            // 如果使用類別權重，則結合距離權重和類別權重
            double weight = distanceWeight;
            if (useClassWeights) {
                weight *= classWeights.getOrDefault(label, 1.0);
            }
            
            // 累加該標籤的權重
            labelWeights.put(label, labelWeights.getOrDefault(label, 0.0) + weight);
        }
        
        return labelWeights;
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
        if (!isTrained) {
            throw new IllegalStateException("分類器尚未訓練");
        }
        
        this.lastEvaluation = WeightedKNNUtils.evaluateModel(this, folds, maxTestSamplesPerFold);
        return this.lastEvaluation;
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
     * @return WeightedKNNClassifier模型實例
     * @throws IOException 如果加載失敗
     * @throws ClassNotFoundException 如果找不到類
     */
    public static WeightedKNNClassifier loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            WeightedKNNClassifier model = (WeightedKNNClassifier) ois.readObject();
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
