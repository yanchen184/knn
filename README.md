# KNN 分類器專案

這是一個基於 KNN (K-Nearest Neighbors) 算法的分類器應用程式，用於根據經緯度座標預測配送區域代碼。專案使用 Spring Boot 框架搭建，提供 RESTful API 接口。

## 功能特點

- **KNN 分類算法**：根據訓練數據中的最近鄰居進行分類預測
- **模型序列化**：支援將訓練好的模型保存到檔案並從檔案載入
- **Excel 數據處理**：從 Excel 檔案讀取訓練數據
- **模型評估**：提供準確率、精確率、召回率、F1分數等評估指標
- **API 接口**：提供基於 HTTP 的分類服務

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
│   │   ├── KNNClassifier.java             # KNN分類器實現
│   │   ├── LabeledPoint.java              # 帶標籤的數據點
│   │   └── Point.java                     # 基礎數據點
│   ├── utils/
│   │   └── ReadExcel.java                 # Excel讀取工具
│   └── KmeansApplication.java             # 應用入口
├── src/main/resources/
│   └── application.properties             # 應用配置
└── pom.xml                                # Maven配置
```

## 快速開始

### 1. 配置

在 `application.properties` 檔案中設置以下配置：

```properties
# KNN分類器配置
classifier.k=10                    # K值 (鄰居數量)
classifier.model-path=knn_classifier.ser  # 模型保存路徑
classifier.need-train=true         # 是否需要重新訓練
classifier.xlsx-file-path=您的Excel檔案路徑  # 訓練數據路徑
```

Excel 檔案應該包含以下列：
- `LATITUDE`：緯度座標
- `LONGITUDE`：經度座標
- `DELIVERY ZONE CODE`：配送區域代碼（作為分類標籤）

### 2. 編譯和運行

```bash
# 使用Maven編譯
mvn clean package

# 運行應用程式
java -jar target/kmeans-0.0.1-SNAPSHOT.jar
```

### 3. API 使用

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
  "trainingDataSize": 5000
}
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

## 實現細節

- 專案使用 KNN 算法進行地理位置分類
- 支持自動從 Excel 檔案載入訓練數據
- 模型可序列化，支持保存到檔案和從檔案載入
- 提供交叉驗證評估模型的性能

## 開發者

此專案由 yanchen 開發。