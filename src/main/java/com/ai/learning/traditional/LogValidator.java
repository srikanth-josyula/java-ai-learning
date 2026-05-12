package com.ai.learning.traditional;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

public class LogValidator {

    public static void main(String[] args) throws Exception {

        long start = System.currentTimeMillis();

        List<String> logs = Files.readAllLines(Paths.get("log.txt"));

        int total = logs.size();
        int valid = 0;

        for (String log : logs) {
            if (isValid(log)) {
                valid++;
                System.out.println(log + " => VALID");
            }
        }

        long end = System.currentTimeMillis();

        System.out.println(valid + "/" + total + " are only valid");
        System.out.println("TOTAL TIME (ms): " + (end - start));
    }

    static boolean isValid(String log) {

        if (log == null || log.length() < 20) return false;

        if (!log.contains("GET") && !log.contains("POST")) return false;

        if (log.contains("???")) return false;

        if (log.contains("bad-date")) return false;

        if (!log.matches(".*\\d+\\.\\d+\\.\\d+\\.\\d+.*")) return false;

        return true;
    }
}