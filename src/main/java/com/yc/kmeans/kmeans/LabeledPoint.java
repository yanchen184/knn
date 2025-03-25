package com.yc.kmeans.kmeans;

import lombok.Getter;

import java.io.Serial;
import java.util.Arrays;

/**
 * 帶標籤的數據點，繼承自基礎數據點
 */
@Getter
public class LabeledPoint extends Point {
    @Serial
    private static final long serialVersionUID = 1L;
    private final String label;

    public LabeledPoint(double[] features, String label) {
        super(features);
        this.label = label;
    }

    @Override
    public String toString() {
        return "LabeledPoint{" + super.toString() + ", label=" + label + '}';
    }
}
