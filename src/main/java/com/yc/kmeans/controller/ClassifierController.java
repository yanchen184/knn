package com.yc.kmeans.controller;

import com.yc.kmeans.kmeans.EvaluationResult;
import com.yc.kmeans.kmeans.WeightedKNNClassifier;
import com.yc.kmeans.kmeans.LabeledPoint;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static com.yc.kmeans.utils.ReadExcel.readExcelData;

@RestController
@Slf4j
public class ClassifierController {
    @Value("${classifier.xlsx-file-path:C:\\Users\\yanchen\\workspace\\ars\\Address-20250220103453.xlsx}")
    private String xlsxFilePath;
    
    @Value("${classifier.model-path:weighted_knn_classifier.ser}")
    private String modelFilePath;
    
    @Value("${classifier.need-train:true}")
    private boolean isNeedTrain;
    
    @Value("${classifier.k:10}")
    private int k;
    private WeightedKNNClassifier classifier;

    /**
     * start.
     */
    @PostConstruct
    public void init() {
        if (isNeedTrain) {
            createAndTrainNewModel();
        } else {
            File modelFile = new File(modelFilePath);
            if (modelFile.exists()) {
                try {
                    classifier = WeightedKNNClassifier.loadModel(modelFilePath);
                    log.info("成功載入已訓練的加權KNN分類器");
                } catch (Exception e) {
                    log.warn("載入模型失敗，將創建新模型: {}", e.getMessage());
                    createAndTrainNewModel();
                }
            } else {
                log.info("未找到已訓練的模型，將創建新模型");
                createAndTrainNewModel();
            }
        }
    }

    /**
     * create and train new model.
     */
    private void createAndTrainNewModel() {
        List<LabeledPoint> trainingData = new ArrayList<>();
        try {
            trainingData = readExcelData(xlsxFilePath);
            // show the first 5 data points
            int count = Math.min(5, trainingData.size());
            log.info("成功讀取 {} 個資料點", trainingData.size());
            log.info("前 {} 個資料點示例：", count);
            for (int i = 0; i < count; i++) {
                log.info(trainingData.get(i).toString());
            }

        } catch (IOException e) {
            log.info("讀取Excel檔案時發生錯誤: {}", e.getMessage());
            e.printStackTrace();
        }

        // 如果未配置 k 值，則定義為訓練數據的開平方根
        if (k <= 0) {
            k = (int) Math.sqrt(trainingData.size());
        }
        log.info("k值: {}", k);
        classifier = new WeightedKNNClassifier(k);
        // 啟用類別權重，對樣本少的類別給予更高權重
        classifier.setUseClassWeights(true);
        // 設置類別權重上限，避免單樣本類別權重過高
        classifier.setMaxClassWeight(50.0);
        // 增大距離權重因子，強調距離對分類的影響(距離近的樣本權重有極大提升)
        classifier.setDistanceWeightFactor(2.0);

        // train
        classifier.train(trainingData);

        // save model
        try {
            classifier.saveModel(modelFilePath);
            log.info("成功訓練並保存新的加權KNN分類器");
        } catch (IOException e) {
            log.warn("保存模型失敗: {}", e.getMessage());
        }
    }

    @GetMapping("/evaluate")
    public Map<String, Object> evaluateModel(@RequestParam(defaultValue = "3") int folds,
                                             @RequestParam(defaultValue = "100") int maxTestSamplesPerFold) {
        EvaluationResult result = classifier.evaluateModel(folds, maxTestSamplesPerFold);

        Map<String, Object> response = new HashMap<>();
        response.put("accuracy", result.getAccuracy());
        response.put("precision", result.getPrecision());
        response.put("recall", result.getRecall());
        response.put("f1Score", result.getF1Score());
        response.put("r2Score", result.getR2Score());
//        response.put("confusionMatrix", result.getConfusionMatrix());
        response.put("classCounts", result.getClassCounts());

        return response;
    }


    /**
     * 分類API端點 - 根據座標值預測類別
     *
     * @param latitude  第一個座標值
     * @param longitude 第二個座標值
     * @return 預測的類別
     */
    @GetMapping("/classifier")
    public String classify(@RequestParam double latitude, @RequestParam double longitude) {
        log.info("收到分類請求: latitude={}, longitude={}", latitude, longitude);
        return classifier.predict(latitude, longitude);
    }

    /**
     * 用於檢查模型狀態的端點
     *
     * @return 模型信息
     */
    @GetMapping("/classifier/info")
    public Map<String, Object> getModelInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("isTrained", classifier.isTrained());
        info.put("k", classifier.getK());
        info.put("trainingDataSize", classifier.getTrainingDataSize());
        info.put("useClassWeights", classifier.isUseClassWeights());
        info.put("maxClassWeight", classifier.getMaxClassWeight());
        info.put("distanceWeightFactor", classifier.getDistanceWeightFactor());
        return info;
    }
    
    /**
     * 調整分類器參數
     * 
     * @param useClassWeights 是否使用類別權重
     * @param maxClassWeight 類別權重上限
     * @param distanceWeightFactor 距離權重因子
     * @return 更新後的模型信息
     */
    @GetMapping("/classifier/adjust")
    public Map<String, Object> adjustClassifier(
            @RequestParam(required = false) Boolean useClassWeights,
            @RequestParam(required = false) Double maxClassWeight,
            @RequestParam(required = false) Double distanceWeightFactor) {
        
        if (useClassWeights != null) {
            classifier.setUseClassWeights(useClassWeights);
            log.info("已設置類別權重使用狀態: {}", useClassWeights);
        }
        
        if (maxClassWeight != null && maxClassWeight > 0) {
            classifier.setMaxClassWeight(maxClassWeight);
            log.info("已設置類別權重上限: {}", maxClassWeight);
        }
        
        if (distanceWeightFactor != null && distanceWeightFactor > 0) {
            classifier.setDistanceWeightFactor(distanceWeightFactor);
            log.info("已設置距離權重因子: {}", distanceWeightFactor);
        }
        
        // 如果參數有變化，嘗試保存模型
        if (useClassWeights != null || maxClassWeight != null || distanceWeightFactor != null) {
            try {
                classifier.saveModel(modelFilePath);
                log.info("已保存更新後的模型參數");
            } catch (IOException e) {
                log.warn("保存模型失敗: {}", e.getMessage());
            }
        }
        
        return getModelInfo();
    }
}