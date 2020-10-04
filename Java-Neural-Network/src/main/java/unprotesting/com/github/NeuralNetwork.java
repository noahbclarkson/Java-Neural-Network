package unprotesting.com.github;

import java.util.Arrays;
import java.util.Random;
import java.io.FileWriter;
import java.io.IOException;

public class NeuralNetwork {
	// Variable Declaration

	// Layers
	static Layer[] layers;
	static Layer[] cachedLayers;
	static float cachedLayerSuccessRate = 0f;

	// Training data
	static TrainingData[] tData1, tDataFull, ittData, initData;

	//	Config Options
	//	Adjust the success rate at which the neural network will stop training, it will stop after a period time if it doesn't reach this
	public static float successRateAimValue = 99f;
	//  How many samples in the final check for success rate
	public final static int finalSuccessRateCheckAmount = 10000;

	// Boolean for Async Threads
	public static boolean isComplete = true;

	// Asnyc references to progress
	public static int time = 0;
	public static int i_stat = 0;
	public static float currentChange = 0f;
	public static float public_error = 0;
	public static float totalerror = 0;
	public static float errorchecks = 0;
	public static int algorithmImprovmentFunction = 10;

	// CSV data writer
	public static FileWriter csvWriter;

	public static boolean debugMode = false;

	// Main Method
	public static void main(String[] args) throws InterruptedException, IOException {
		// Set the Min and Max weight value for all Neurons
		Neuron.setRangeWeight(-1, 1);

		if (debugMode == true) {
			csvWriter = new FileWriter("NeuralData.csv");
			csvWriter.append("\n");
		}

		// Create the layers
		layers = new Layer[5];
		layers[0] = null; // Input Layer 0,10
		layers[1] = new Layer(10, 125); // Hidden Layer 10, 125
		layers[2] = new Layer(125, 750); // Hidden Layer 125, 750
		layers[3] = new Layer(750, 500); // Hidden Layer 750, 500
		layers[4] = new Layer(500, 2); // Output Layer 500, 2

		Thread.sleep(5);

		// For debug purposes
		Thread asyncErrorCheckThread = new Thread(() -> {
			while (!isComplete) {
				try {
					Thread.sleep(1);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				// Check error function
				if (public_error > 0) {
					totalerror = +public_error;
					errorchecks = +1;
				}
				if (public_error < 0) {
					totalerror = +-public_error;
					errorchecks = +1;
				}
			}
		});

		// For debug purposes
		Thread asyncErrorThread = new Thread(() -> {
			while (!isComplete) {
				try {
					Thread.sleep(50);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				// Check error function
				try {
					csvWriter.append(Float.toString((totalerror / errorchecks)));
					csvWriter.append("\n");
				} catch (IOException e) {
					e.printStackTrace();
				}
				totalerror = 0;
				errorchecks = 0;
			}
		});

		//	Increase the amount of times the Neural Network can change per 'reset'
		Thread algorithmImprovemntThread = new Thread(() -> {
			while(!isComplete){
				try {
					Thread.sleep(3000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				algorithmImprovmentFunction += 1;
			}
		});

		//	Start debug async function
		isComplete = false;
		if (debugMode){
			asyncErrorCheckThread.start();
			asyncErrorThread.start();
		}

		//	Train data under mutiple configurations
		//	Decreasing training iterations and learning rate simultaniously 
		//	Train intially to reach around 90% accuracy.
		successRateAimValue -= 0.0001f;
		tData1 = loadInputs(finalSuccessRateCheckAmount);
		ittData = loadInputs(250);
		cachedLayers = layers;
		System.out.println("Starting Initialization..");
		int init_iterations = 150;
		int iterations = 10;
		for (int i = 1; i < init_iterations; i++){
			System.out.println("Completed Training " + i + "/" + init_iterations);
			initialTrain(1, (0.00025f*i), loadInputs(200));
		}
		System.out.println("Initialization Complete");
		algorithmImprovemntThread.start();
		//  Only allow the network to improve to allow for greater accuracy
		for (int i = 1; i < iterations; i++){
			System.out.println("Starting Training: " + i + "/" + iterations);
			train(5, 0.2f, loadInputs((250/i)+10));
			if (debugMode){
				csvWriter.flush();
			}
		}

		//	Stop async task
		isComplete = true;

		System.out.println("Checking data Success.. ");

		// Create testing data

		finalTest(tData1);

		// Close writer
		if (debugMode == true) {
			csvWriter.flush();
			csvWriter.close();
		}

	}

	public static void finalTest(TrainingData[] tData1) throws InterruptedException, IOException {
		// Test network
		System.out.println("============");
        System.out.println("Output after testing");
		System.out.println("============");
        for(int i = 0; i < 25; i++) {
			forward(tData1[i].data);
			System.out.println("For the data: " + Arrays.toString(tData1[i].data));
			System.out.println((layers[4].neurons[0].value)+ " - "+ (layers[4].neurons[1].value) + " =(this should equal)= " + Arrays.toString(tData1[i].expectedOutput));
			Thread.sleep(100);
		}
		Thread.sleep(50);

		// Check Rate of Success on training data
		System.out.println("Calculating total Success Rate For \'" + finalSuccessRateCheckAmount + "\' Samples..");
		System.out.println("Success Rate: %" + checkSuccessRate(tData1));
	}
   
	public static void forward(float[] inputs) {
		// First bring the inputs into the input layer layers[0]
		layers[0] = new Layer(inputs);

		for (int i = 1; i < layers.length; i++) {
			for (int j = 0; j < layers[i].neurons.length; j++) {
				float sum = 0;
				for (int k = 0; k < layers[i - 1].neurons.length; k++) {
					sum += layers[i - 1].neurons[k].value * layers[i].neurons[j].weights[k];
				}
				sum += layers[i].neurons[j].bias;
				layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
			}
		}
	}

	public static float[] asyncForward(Layer[] inputs, float[] inputValues){

		Layer[] asyncLayers = inputs;

		asyncLayers[0] = new Layer(inputValues);

		for (int i = 1; i < asyncLayers.length; i++) {
			for (int j = 0; j < asyncLayers[i].neurons.length; j++) {
				float sum = 0;
				for (int k = 0; k < asyncLayers[i - 1].neurons.length; k++) {
					sum += asyncLayers[i - 1].neurons[k].value * asyncLayers[i].neurons[j].weights[k];
				}
				sum += layers[i].neurons[j].bias;
				asyncLayers[i].neurons[j].value = StatUtil.Sigmoid(sum);
			}
		}
		float[] val = {asyncLayers[4].neurons[0].value, asyncLayers[4].neurons[1].value};
		return val;
	}

	//	Create training data by loading float arrays as soprted ascendingly, descendingly or not at all
	public static TrainingData[] loadInputs(int inputs) {
		TrainingData[] outputTrainingData = new TrainingData[inputs];
		int i = 0;
		for (;i<inputs;){
			float temp = loadRandomFloat();
			if (temp > 7000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				float[] sortedInput = ascendingBubbleSortFloatArray(z);
				outputTrainingData[i] = new TrainingData(sortedInput, new float[]{1, 0});
				i++;
			}
			if (temp <= 3000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				float[] sortedInput = descendingBubbleSortFloatArray(z);
				outputTrainingData[i] = new TrainingData(sortedInput, new float[]{0, 0});
				i++;
			}
			if (7000 > temp && temp > 3000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				boolean isasorted = isAscendingSorted(z);
				boolean isdsorted = isDecsendingSorted(z);
				if (isdsorted == true){
					outputTrainingData[i] = new TrainingData(z, new float[]{0, 0});
					i++;
				}
				else if (isasorted == true){
					outputTrainingData[i] = new TrainingData(z, new float[]{1, 0});
					i++;
				}
				else {
					outputTrainingData[i] = new TrainingData(z, new float[]{0, 1});
					i++;
				}
			}
		}
		return outputTrainingData;

	}

	// Test if an array is sorted ascendingly
	public static boolean isAscendingSorted(float[] a) {
		for (int i = 0; i < a.length - 1; i++) {
			if (a[i] > a[i + 1]) {
				return false; // It is proven that the array is not sorted.
			}
		}
	
		return true; // If this part has been reached, the array must be sorted.
	}

	// Test if an array is sorted decendingly
	public static boolean isDecsendingSorted(float[] a) {
		for (int i = 0; i < a.length - 1; i++) {
			if (a[i] < a[i + 1]) {
				return false; // It is proven that the array is not sorted.
			}
		}
	
		return true; // If this part has been reached, the array must be sorted.
	}

	// Function to load a random float from 1-9999
	public static float loadRandomFloat() {
		int leftLimit = 1;
		int rightLimit = 9999;
		int generatedInteger = leftLimit + (int) (new Random().nextFloat() * (rightLimit - leftLimit));
		return(float)generatedInteger;
	}

	// Ascending sort function
	public static float[] ascendingBubbleSortFloatArray(float[] a) {
		boolean sorted = false;
		float temp;
		while(!sorted) {
			sorted = true;
			for (int i = 0; i < a.length - 1; i++) {
				if (a[i] > a[i+1]) {
					temp = a[i];
					a[i] = a[i+1];
					a[i+1] = temp;
					sorted = false;
				}
			}
		}
		return a;
	}

	// Descending sort function
	public static float[] descendingBubbleSortFloatArray(float[] a) {
		boolean sorted = false;
		float temp;
		while(!sorted) {
			sorted = true;
			for (int i = 0; i < a.length - 1; i++) {
				if (a[i] < a[i+1]) {
					temp = a[i];
					a[i] = a[i+1];
					a[i+1] = temp;
					sorted = false;
				}
			}
		}
		return a;
	}
    
    // The idea is that you calculate a gradient and cache the updated weights in the neurons.
    // When ALL the neurons new weight have been calculated we refresh the neurons.
    // Meaning we do the following:
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
    public static void backward(float learning_rate,TrainingData tData) throws IOException {
    	
    	int number_layers = layers.length;
		int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output
    	for(int i = 0; i < layers[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = layers[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
			float derivative = output-target;
			float delta = derivative*(output*(1-output));
			public_error = delta;
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
				float error = delta*previous_output;
				layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
				layers[out_index].neurons[i].cache_bias = layers[out_index].neurons[i].bias - learning_rate*error;
			}
		}

    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-2; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < layers[i].neurons.length; j++) {
    			float output = layers[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1);
    			float delta = (gradient_sum)*(output*(1-output));
    			layers[i].neurons[j].gradient = delta;
    			// And for all their weights
    			for(int k = 0; k < layers[i].neurons[j].weights.length; k++) {
    				float previous_output = layers[i-1].neurons[k].value;
    				float error = delta*previous_output;
					layers[i].neurons[j].cache_weights[k] = layers[i].neurons[j].weights[k] - learning_rate*error;
					layers[i].neurons[j].cache_bias = layers[i].neurons[j].bias - learning_rate*error;
				}
			}
    	}
    	
    	// Here we do another pass where we update all the weights
    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
				layers[i].neurons[j].update_weight();
				layers[i].neurons[j].update_biases();
    		}
		}
    	
    }
    
    // This function sums up all the gradient connecting a given neuron in a given layer
    public static float sumGradient(int n_index,int l_index) {
    	float gradient_sum = 0;
    	Layer current_layer = layers[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
	}
	
	//	Check progress of training
	public static void checkProgress(){
		System.out.println("Total percentage complete: %" + NeuralNetwork.time + ". Current training: No." + NeuralNetwork.i_stat + ". ");
	}

	//	Check success rate by loading new test data
	public static float checkSuccessRate(TrainingData[] testData1000) throws IOException {
		int checks = 0;
		float diff = 0;
		float totalDif = 0;
		for (int i = 0; i < testData1000.length; i++){
			if (testData1000[i] != null){
				float[] val = asyncForward(layers, testData1000[i].data);
				float[] optimalVal = testData1000[i].expectedOutput;
				for (int x = 0; x < 2; x++){
					if (Math.round(val[x]) == optimalVal[x]){
						diff = diff + optimalVal[x] - Math.round(val[x]);
					}
					if ((Math.round(val[x])) != optimalVal[x]){
						diff = diff + 1;
					}
					if (val == optimalVal){
						diff = diff + 0;
					}
				}
				diff = diff*50;
				totalDif = totalDif + diff;
				diff = 0;
				checks++;
			}
		}
		totalDif = 100-(totalDif/checks);
		return totalDif;
	}
    
	//	This function is used to train data by pushing it forward and backward
	//	It then checks the success rate and resets the layers to the highest success rate weight settings if the success rate is lower than its record
    public static void train(int training_iterations,float learning_rate, TrainingData[] traningData)
			throws IOException, InterruptedException {
    	for(int i = 0; i < training_iterations; i++) {
			time = (i * 100 / training_iterations);
			i_stat = i;
    		for(int j = 0; j < traningData.length; j++) {
    			forward(traningData[j].data);
				backward(learning_rate,traningData[j]);
				if (j%algorithmImprovmentFunction == 0 && j != 0){
					float currentSuccessRate = checkSuccessRate(ittData);
					if (debugMode){
						csvWriter.append(Float.toString(currentSuccessRate));
						csvWriter.append("\n");
					}
					if (currentSuccessRate > successRateAimValue){
						System.out.println("Reached "+ (successRateAimValue+0.0001f) + "% success rate");
						cachedLayers = layers;
						cachedLayerSuccessRate = currentSuccessRate;
						finalTest(tData1);
						if (debugMode){
							csvWriter.flush();
							csvWriter.close();
						}
						System.exit(-1);
					}
					else if (currentSuccessRate < cachedLayerSuccessRate) {
						layers = cachedLayers;
					}
					else if (currentSuccessRate > cachedLayerSuccessRate){
						System.out.println("New Record: %" + currentSuccessRate);
						if (debugMode){
							csvWriter.flush();
						}
						cachedLayers = layers;
						cachedLayerSuccessRate = currentSuccessRate;
						algorithmImprovmentFunction *= 1.05;
					}
					else{
						cachedLayers = layers;
						cachedLayerSuccessRate = currentSuccessRate;
					}
				}
			}
		}
		layers = cachedLayers;
	}
	
	//	Train without checking success rate
	public static void initialTrain(int training_iterations, float learning_rate, TrainingData[] traningData)
	throws IOException, InterruptedException {
		for(int i = 0; i < training_iterations; i++) {
			time = (i * 100 / training_iterations);
			i_stat = i;
			for(int j = 0; j < traningData.length; j++) {
				forward(traningData[j].data);
				backward(learning_rate,traningData[j]);
			}
		}	
	}
}