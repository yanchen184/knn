package com.yc.kmeans.kmeans;

import lombok.Getter;

import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * 基礎數據點
 */
@Getter
public class Point implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
    private final double[] features;

    public Point(double[] features) {
        this.features = features;
    }

    @Override
    public String toString() {
        return "Point{features=" + Arrays.toString(features) + "}";
    }
}
