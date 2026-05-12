package com.ai.learning;

import java.io.File;
import java.util.Scanner;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DL4JOddEvenTrainer {

	private MultiLayerNetwork model;
	private static final File MODEL_FILE = new File("models/odd-even-model.zip");

	public static void main(String[] args) {

		DL4JOddEvenTrainer app = new DL4JOddEvenTrainer();
		Scanner scanner = new Scanner(System.in);

		try {

			// ----------------------------------------
			// STEP 1: FIRST INPUT
			// ----------------------------------------
			System.out.print("Enter a number: ");
			String input1 = scanner.nextLine().trim();
			long number1 = Long.parseLong(input1);

			// Try prediction first
			if (!app.loadModelIfExists()) {

				System.out.println("\nNo trained model found in memory or disk.");
				System.out.println("Using input: " + number1);
				System.out.println("Choosing to train a new model...\n");

				app.createModel();
				app.trainModel();

				ModelSerializer.writeModel(app.model, MODEL_FILE, true);

				System.out.println("Model trained and saved at:");
				System.out.println(MODEL_FILE.getAbsolutePath());
			}

			// ----------------------------------------
			// STEP 2: SECOND INPUT AFTER TRAINING
			// ----------------------------------------
			System.out.println("\n=== MODEL READY ===");

			System.out.print("Enter number again: ");
			String input2 = scanner.nextLine().trim();
			long number2 = Long.parseLong(input2);

			app.predict(number2);

		} catch (Exception e) {
			e.printStackTrace();
		}

		scanner.close();
	}

	// ----------------------------------------
	// LOAD MODEL IF EXISTS
	// ----------------------------------------
	private boolean loadModelIfExists() {

		try {
			if (MODEL_FILE.exists()) {

				model = ModelSerializer.restoreMultiLayerNetwork(MODEL_FILE);

				System.out.println("Loaded trained model from disk.");
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		return false;
	}

	// ----------------------------------------
	// CREATE MODEL
	// ----------------------------------------
	public void createModel() {

		MultiLayerConfiguration config = new NeuralNetConfiguration.Builder().seed(42).updater(new Adam(0.01)).list()
				.layer(new DenseLayer.Builder().nIn(8).nOut(16).activation(Activation.RELU).build())
				.layer(new DenseLayer.Builder().nIn(16).nOut(8).activation(Activation.RELU).build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(8).nOut(1)
						.activation(Activation.SIGMOID).build())
				.build();

		model = new MultiLayerNetwork(config);
		model.init();
	}

	// ----------------------------------------
	// TRAIN MODEL
	// ----------------------------------------
	public void trainModel() {

		System.out.println("Training model...\n");

		int samples = 100;

		double[][] x = new double[samples][8];
		double[][] y = new double[samples][1];

		for (int i = 0; i < samples; i++) {
			x[i] = toBits(i);
			y[i][0] = (i % 2 == 0) ? 0 : 1;
		}

		DataSet ds = new DataSet(Nd4j.create(x), Nd4j.create(y));

		for (int i = 0; i < 2000; i++) {
			model.fit(ds);
		}

		System.out.println("Model training completed.\n");
	}

	// ----------------------------------------
	// PREDICT
	// ----------------------------------------
	public void predict(long number) {

		INDArray input = Nd4j.create(new double[][] { toBits(number) });

		double score = model.output(input).getDouble(0);

		System.out.println("\n=== PREDICTION ===");
		System.out.println("Number: " + number);
		System.out.println("Result: " + (score > 0.5 ? "ODD" : "EVEN"));
	}

	// ----------------------------------------
	// CONVERT TO BINARY FEATURES
	// ----------------------------------------
	private double[] toBits(long number) {

		long n = Math.abs(number);
		double[] bits = new double[8];

		for (int i = 0; i < 8; i++) {
			bits[7 - i] = (n >> i) & 1;
		}

		return bits;
	}
}