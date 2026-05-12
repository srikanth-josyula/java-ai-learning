package com.ai.learning.ml;

import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;

import java.util.Arrays;

public class SparkLogML {

    public static void main(String[] args) {

        long start = System.currentTimeMillis();

        SparkSession spark = SparkSession.builder()
                .appName("LogML")
                .master("local[*]")
                .getOrCreate();

        Dataset<String> logs = spark.read().textFile("log.txt");

        Dataset<Row> df = logs.toDF("log");

        Dataset<Row> scored = df.withColumn("isValid",
                when(col("log").contains("???"), 0)
                .when(col("log").contains("bad-date"), 0)
                .when(col("log").contains("GET").or(col("log").contains("POST")), 1)
                .otherwise(0)
        );

        Dataset<Row> valid = scored.filter(col("isValid").equalTo(1));

        long total = df.count();
        long validCount = valid.count();

        valid.select("log")
                .collectAsList()
                .forEach(r -> System.out.println(r.getString(0) + " => VALID"));

        long end = System.currentTimeMillis();

        System.out.println(validCount + "/" + total + " are only valid");
        System.out.println("TOTAL TIME (ms): " + (end - start));

        spark.stop();
    }
}