package com.ai.learning.ml;

import ml.dmlc.xgboost4j.java.*;

import java.util.*;

public class XGBoostLog {

    public static void main(String[] args) throws Exception {

        long start = System.currentTimeMillis();

        float[][] data = {
                {120, 1, 1, 0, 1}, // valid
                {15, 0, 0, 1, 0},  // invalid
                {140, 1, 1, 0, 1}, // valid
                {10, 0, 0, 1, 0}   // invalid
        };

        float[] label = {1, 0, 1, 0};

        DMatrix train = new DMatrix(data, label);

        Map<String, Object> params = new HashMap<>();
        params.put("objective", "binary:logistic");
        params.put("eta", 0.3);
        params.put("max_depth", 3);

        Booster model = XGBoost.train(train, params, 5, new HashMap<>(), null, null);

        float[][] test = {
                {130, 1, 1, 0, 1}
        };

        DMatrix testMat = new DMatrix(test, 1, test.length);

        float[][] pred = model.predict(testMat);

        System.out.println("VALID PROBABILITY: " + pred[0][0]);

        long end = System.currentTimeMillis();

        System.out.println("TIME(ms): " + (end - start));
    }
}