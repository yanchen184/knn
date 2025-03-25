package com.yc.kmeans.utils;

import com.yc.kmeans.kmeans.LabeledPoint;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellType;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
public class ReadExcel {

    private ReadExcel() {
        throw new IllegalStateException("Utility class");
    }

    /**
     * 從Excel檔案讀取資料，轉換為LabeledPoint列表
     *
     * @param filePath Excel檔案路徑
     * @return LabeledPoint列表
     * @throws IOException 如果讀取檔案出錯
     */
    public static List<LabeledPoint> readExcelData(String filePath) throws IOException {
        List<LabeledPoint> points = new ArrayList<>();
        try (FileInputStream excelFile = new FileInputStream(filePath);
             Workbook workbook = new XSSFWorkbook(excelFile)) {

            for (String sheetName : List.of("ESTATE", "STREET", "STREET_NUMBER")) {
                processSheet(workbook, sheetName, points);
            }
        }
        return points;
    }

    private static void processSheet(Workbook workbook, String sheetName, List<LabeledPoint> points) {
        Sheet sheet = workbook.getSheet(sheetName);
        if (sheet == null) {
            log.info("未找到工作表: {}", sheetName);
            return;
        }

        Map<String, Integer> columnIndexes = getColumnIndexes(sheet.getRow(0));
        if (!columnIndexes.keySet().containsAll(List.of("LATITUDE", "LONGITUDE", "DELIVERY ZONE CODE"))) {
            log.info("{}工作表缺少必要的欄位", sheetName);
            return;
        }

        int latIndex = columnIndexes.get("LATITUDE");
        int lngIndex = columnIndexes.get("LONGITUDE");
        int zoneIndex = columnIndexes.get("DELIVERY ZONE CODE");

        for (int rowIndex = 1; rowIndex <= sheet.getLastRowNum(); rowIndex++) {
            processRow(sheet.getRow(rowIndex), latIndex, lngIndex, zoneIndex, points);
        }

        log.info("從 {} 工作表讀取了 {} 個資料點", sheetName, points.size());
    }

    private static void processRow(Row dataRow, int latIndex, int lngIndex, int zoneIndex, List<LabeledPoint> points) {
        if (dataRow == null) return;

        Double latitude = getCellNumericValue(dataRow.getCell(latIndex));
        Double longitude = getCellNumericValue(dataRow.getCell(lngIndex));
        String zoneCode = getCellStringValue(dataRow.getCell(zoneIndex));

        if (isValidData(latitude, longitude, zoneCode)) {
            points.add(new LabeledPoint(new double[]{latitude, longitude}, zoneCode));
        }
    }


    /**
     * 取得標題列中每個欄位名稱對應的索引
     */
    private static Map<String, Integer> getColumnIndexes(Row headerRow) {
        Map<String, Integer> columnIndexes = new HashMap<>();
        if (headerRow != null) {
            for (int i = 0; i < headerRow.getLastCellNum(); i++) {
                Cell cell = headerRow.getCell(i);
                if (cell != null) {
                    columnIndexes.put(cell.getStringCellValue().trim().toUpperCase(), i);
                }
            }
        }
        return columnIndexes;
    }

    /**
     * 檢查資料是否有效
     */
    private static boolean isValidData(Double latitude, Double longitude, String zoneCode) {
        return latitude != null && longitude != null && zoneCode != null && !zoneCode.isEmpty() && zoneCode.contains("-");
    }


    /**
     * 安全地從儲存格讀取數值
     */
    private static Double getCellNumericValue(Cell cell) {
        if (cell == null) {
            return null;
        }

        try {
            if (cell.getCellType() == CellType.NUMERIC) {
                return cell.getNumericCellValue();
            } else if (cell.getCellType() == CellType.STRING) {
                return parseStringToDouble(cell.getStringCellValue().trim());
            }
        } catch (Exception e) {
            log.error("讀取數值時發生錯誤: {}", e.getMessage());
        }

        return null;
    }

    /**
     * 將字串安全轉換為數值
     */
    private static Double parseStringToDouble(String value) {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    /**
     * 安全地從儲存格讀取字串
     */
    private static String getCellStringValue(Cell cell) {
        if (cell == null) {
            return null;
        }

        try {
            if (cell.getCellType() == CellType.STRING) {
                return cell.getStringCellValue().trim();
            } else if (cell.getCellType() == CellType.NUMERIC) {
                // 有些區域代碼可能被Excel視為數字
                return String.valueOf((int) cell.getNumericCellValue());
            }
        } catch (Exception e) {
            log.error("讀取字串時發生錯誤: {}", e.getMessage());
        }

        return null;
    }
}