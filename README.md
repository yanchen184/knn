# 加權KNN分類器專案

這是一個基於加權KNN (Weighted K-Nearest Neighbors) 算法的分類器應用程式，用於根據經緯度座標預測配送區域代碼。特別針對樣本數量不平衡的問題進行了優化，適合處理某些類別樣本極少的情況。專案使用 Spring Boot 框架搭建，提供 RESTful API 接口。

## 功能特點

- **加權KNN分類算法**：
  - 基於距離的權重計算，距離越近權重越高
  - 類別權重平衡，對樣本數量少的類別給予更高權重
  - 特別優化單樣本類別的處理
- **模型序列化**：支援將訓練好的模型保存到檔案並從檔案載入
- **Excel 數據處理**：從 Excel 檔案讀取訓練數據
- **模型評估**：提供準確率、精確率、召回率、F1分數等評估指標
- **API 接口**：提供基於 HTTP 的分類服務及參數調整功能

## 系統需求

- Java 23 或更高版本
- Maven 3.6 或更高版本
- Spring Boot 3.4.3

## 專案結構

```
kmeans/
├── src/main/java/com/yc/kmeans/
│   ├── controller/
│   │   └── ClassifierController.java      # API控制器
│   ├── kmeans/
│   │   ├── EvaluationResult.java          # 評估結果類
│   │   ├── KNNClassifier.java             # 基礎KNN分類器實現
│   │   ├── WeightedKNNClassifier.java     # 加權KNN分類器實現
│   │   ├── WeightedKNNUtils.java          # 加權KNN工具類
│   │   ├── LabeledPoint.java              # 帶標籤的數據點
│   │   └── Point.java                     # 基礎數據點
│   ├── utils/
│   │   └── ReadExcel.java                 # Excel讀取工具
│   └── KmeansApplication.java             # 應用入口
├── src/main/resources/
│   └── application.properties             # 應用配置
└── pom.xml                                # Maven配置
```

## 加權KNN參數說明

本專案實現的加權KNN算法有以下可調整參數：

### 主要參數

| 參數名稱 | 說明 | 默認值 | 建議範圍 | 影響 |
|---------|------|-------|---------|------|
| `k` | K值（考慮的最近鄰居數量） | 10 | 3-20 | 較小的值更靈活但可能過擬合，較大的值更穩定但可能欠擬合 |
| `useClassWeights` | 是否啟用類別權重平衡 | true | true/false | 設為true以處理類別不平衡，false則純依賴距離 |
| `maxClassWeight` | 類別權重上限值 | 50.0 | 10.0-100.0 | 較小的值減弱樣本少的類別權重，較大的值增強其權重 |
| `distanceWeightFactor` | 距離權重因子 | 2.0 | 1.0-5.0 | 較大的值使近距離樣本影響更顯著 |
| `epsilon` | 防止除零的小值 | 0.00001 | 0.00001-0.001 | 通常不需調整 |

### 參數調整建議

1. **樣本極度不平衡問題**：
   - 增大`distanceWeightFactor`到3.0-4.0
   - 將`maxClassWeight`設置在30.0-50.0之間
   - 保持`useClassWeights=true`

2. **單樣本類別問題**：
   - 若單樣本類別預測不準確，嘗試降低`maxClassWeight`至20.0
   - 同時增大`distanceWeightFactor`至4.0-5.0
   - 極端情況下可設置`useClassWeights=false`

3. **樣本較為平衡情況**：
   - 使用較小的`distanceWeightFactor`(1.5-2.0)
   - 使用較小的`maxClassWeight`(10.0-30.0)
   - 或直接設置`useClassWeights=false`

## 配置方法

### 1. 應用配置文件

在 `application.properties` 檔案中設置基本配置：

```properties
# 加權KNN分類器配置
classifier.k=10                                     # K值 (鄰居數量)
classifier.model-path=weighted_knn_classifier.ser   # 模型保存路徑
classifier.need-train=true                          # 是否需要重新訓練
classifier.xlsx-file-path=您的Excel檔案路徑            # 訓練數據路徑
```

### 2. 程式碼配置

可以在`ClassifierController.java`的`createAndTrainNewModel`方法中修改默認參數：

```java
// 創建分類器並設置參數
classifier = new WeightedKNNClassifier(k);
// 啟用或禁用類別權重平衡
classifier.setUseClassWeights(true); 
// 設置類別權重上限
classifier.setMaxClassWeight(50.0);
// 設置距離權重因子
classifier.setDistanceWeightFactor(2.0);
// 設置防止除零的小值
// classifier.setEpsilon(0.00001);
```

### 3. 運行時調整

通過API接口動態調整參數，無需重啟應用：

```
GET /classifier/adjust?useClassWeights=true&maxClassWeight=40&distanceWeightFactor=3.0
```

## 快速開始

### 1. 編譯和運行

```bash
# 使用Maven編譯
mvn clean package

# 運行應用程式
java -jar target/kmeans-0.0.1-SNAPSHOT.jar
```

### 2. API 使用

#### 分類 API

將座標點分類為配送區域：

```
GET /classifier?latitude=22.123&longitude=114.456
```

回傳示例：
```
HK-CWB-01
```

#### 模型資訊 API

獲取模型訓練狀態和參數：

```
GET /classifier/info
```

回傳示例：
```json
{
  "isTrained": true,
  "k": 10,
  "trainingDataSize": 5000,
  "useClassWeights": true,
  "maxClassWeight": 50.0,
  "distanceWeightFactor": 2.0
}
```

#### 參數調整 API

動態調整模型參數，無需重新訓練：

```
GET /classifier/adjust?useClassWeights=true&maxClassWeight=40&distanceWeightFactor=3.0
```

#### 模型評估 API

評估模型性能：

```
GET /evaluate?folds=3&maxTestSamplesPerFold=100
```

回傳示例：
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.90,
  "f1Score": 0.905,
  "r2Score": 0.85,
  "classCounts": {
    "HK-CWB-01": 150,
    "HK-WCH-02": 120
  }
}
```

## 處理特殊情況

### 樣本極度不平衡

當您的數據集中某些類別的樣本數量極少（例如只有1-2個樣本）時：

1. 確保類別權重已啟用（API查看info確認`useClassWeights=true`）
2. 如果單樣本類別被錯誤分類，調整參數：
   ```
   GET /classifier/adjust?maxClassWeight=30&distanceWeightFactor=4.0
   ```
3. 如果仍不理想，您可以嘗試完全禁用類別權重，僅依賴於距離：
   ```
   GET /classifier/adjust?useClassWeights=false&distanceWeightFactor=5.0
   ```

### 生產環境優化

對於大規模生產環境部署，建議：

1. 在配置中設置`classifier.need-train=false`以避免每次啟動都重新訓練
2. 使用單獨的訓練流程生成模型文件，然後在生產環境中載入
3. 定期使用新數據重新訓練並評估模型

## 開發者

此專案由 yanchen 開發。