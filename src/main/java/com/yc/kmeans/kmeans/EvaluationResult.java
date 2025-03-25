package com.yc.kmeans.kmeans;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class EvaluationResult implements Serializable {
    private static final long serialVersionUID = 1L;
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;
    private double r2Score;
    private Map<String, Integer> classCounts;
    private Map<String, Map<String, Integer>> confusionMatrix;

    public EvaluationResult() {
        this.classCounts = new HashMap<>();
        this.confusionMatrix = new HashMap<>();
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public void setPrecision(double precision) {
        this.precision = precision;
    }

    public double getPrecision() {
        return precision;
    }

    public void setRecall(double recall) {
        this.recall = recall;
    }

    public double getRecall() {
        return recall;
    }

    public void setF1Score(double f1Score) {
        this.f1Score = f1Score;
    }

    public double getF1Score() {
        return f1Score;
    }

    public void setR2Score(double r2Score) {
        this.r2Score = r2Score;
    }

    public double getR2Score() {
        return r2Score;
    }

    public void setClassCounts(Map<String, Integer> classCounts) {
        this.classCounts = classCounts;
    }

    public Map<String, Integer> getClassCounts() {
        return classCounts;
    }

    public void setConfusionMatrix(Map<String, Map<String, Integer>> confusionMatrix) {
        this.confusionMatrix = confusionMatrix;
    }

    public Map<String, Map<String, Integer>> getConfusionMatrix() {
        return confusionMatrix;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Evaluation Results:\n");
        sb.append("Accuracy: ").append(String.format("%.4f", accuracy)).append("\n");
        sb.append("Precision: ").append(String.format("%.4f", precision)).append("\n");
        sb.append("Recall: ").append(String.format("%.4f", recall)).append("\n");
        sb.append("F1 Score: ").append(String.format("%.4f", f1Score)).append("\n");
        sb.append("R² Score: ").append(String.format("%.4f", r2Score)).append("\n");

        sb.append("\nClass Distribution:\n");
        for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
            sb.append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
        }

        sb.append("\nConfusion Matrix:\n");
        // 打印表頭
        sb.append("Actual\\Predicted\t");
        for (String predictedClass : confusionMatrix.keySet()) {
            sb.append(predictedClass).append("\t");
        }
        sb.append("\n");

        // 打印每行
        for (String actualClass : confusionMatrix.keySet()) {
            sb.append(actualClass).append("\t\t");
            Map<String, Integer> row = confusionMatrix.get(actualClass);
            for (String predictedClass : confusionMatrix.keySet()) {
                sb.append(row.getOrDefault(predictedClass, 0)).append("\t");
            }
            sb.append("\n");
        }

        return sb.toString();
    }
}