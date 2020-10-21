package unprotesting.com.github;

import java.util.Arrays;
import java.util.Random;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

public class NeuralNetwork {
	// Variable Declaration

	public static Layer[] ganLayersCache;

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

	public static File trData;

	public static BufferedReader reader, reader2;

	// Main Method
	public static void main(String[] args) throws InterruptedException, IOException {
		// Set the Min and Max weight value for all Neurons

		Neuron.setRangeWeight(-1f, 1f);

		InputStream is = NeuralNetwork.class.getResourceAsStream("/unprotesting/com/github/trainingData.txt");
		reader = new BufferedReader(new InputStreamReader(is));
		reader.readLine();
		reader.mark(50000);
		reader.reset();

		InputStream is2 = NeuralNetwork.class.getResourceAsStream("/unprotesting/com/github/trainingData2.txt");
		reader2 = new BufferedReader(new InputStreamReader(is2));
		reader2.readLine();
		reader2.mark(50000);
		reader2.reset();

		Layer[] neuralGANLayers = new Layer[5];
		neuralGANLayers[0] = null;
		neuralGANLayers[1] = new Layer(25, 100, true); // Hidden Layer 10, 125
		neuralGANLayers[2] = new Layer(100, 200, true); // Hidden Layer 125, 750
		neuralGANLayers[3] = new Layer(200, 75, true); // Hidden Layer 750, 500
		neuralGANLayers[4] = new Layer(75, 1, true); // Hidden Layer 750, 500

		Layer[][] layers = generateLayers(100);

		layers = ganTrain(layers, neuralGANLayers, 100);
		neuralGANLayers = layers[11];
		System.out.println("Testing..");
		for (int i=0;i<5;i++){
			float[] f = {loadRandomFloat(0, 100)};
			layers[0] = forward(f, layers[0], false);
			for (int k = 0; k < layers[0][3].neurons.length;k++){
				System.out.println((int)layers[0][3].neurons[k].value);
				System.out.println(Character.toChars((int)layers[0][3].neurons[k].value));
			}
			System.out.print("\n");
		}

	}

	public static Layer[][] ganTrain(Layer[][] input, Layer[] ganInput, int iterations) throws IOException, InterruptedException {
		for (int i = 0;i<5;i++){
			TrainingData[] trData = loadGANInputs(100);
			reader.reset();
			reader2.reset();
			System.out.println("GAN Initialization " + i);
			ganInput = train(1, 0.01f*i, trData, ganInput, true);
			// checkGANProgress(ganInput, 5, false);
		}
		System.out.println("Finished GAN Initialization - Checking..");
		checkGANProgress(ganInput, 10, true);
		Layer[][] x = trainMainLayers(input, ganInput, false);
		ganInput = x[10];
		x = returnAllButLast(x);
		x = duplicateAndRandomize(x, 10, 0.025f);
		Thread.sleep(500);
		for (int i = 0;i<iterations;i++){
			System.out.println(i + "/" + iterations);
			x = trainMainLayers(input, ganInput, true);
			ganInput = x[10];
			x = returnAllButLast(x);
			x = duplicateAndRandomize(x, 10, 0.25f);
		}
		x = trainMainLayers(input, ganInput, false);
		return x;
	}


   
	public static Layer[] forward(float[] inputs, Layer[] input, boolean sigmoid) {
		// First bring the inputs into the input layer layers[0]
		input[0] = new Layer(inputs);

		for (int i = 1; i < input.length; i++) {
			for (int j = 0; j < input[i].neurons.length; j++) {
				float sum = 0;
				for (int k = 0; k < input[i - 1].neurons.length; k++) {
					// System.out.println("K: " + k);				
					// System.out.println("SUM: " + sum);
					sum += input[i - 1].neurons[k].value * input[i].neurons[j].weights[k];
				}
				sum += input[i].neurons[j].bias;
				if (sigmoid){
					input[i].neurons[j].value = StatUtil.Sigmoid(sum);
				}
				else{
					input[i].neurons[j].value = StatUtil.leakyRLEU(sum);
				}
			}
		}
		return input;
	}


	//	Return the top 10 Layer[]/weights/biases combinations
	public static Layer[][] trainMainLayers(Layer[][] input, Layer[] ganInput, boolean displayInfo) throws IOException, InterruptedException {
		Layer[][] out = new Layer[12][];
		FullLayer[] check = new FullLayer[10];
		for (int i = 0;i<input.length;i++){
			FullLayer[] full = trainIndividualLayers(input[i], ganInput);
			ganInput = full[1].layers;
			if (i < 10){
				check[i] = full[0];
			}
			else{
				for (int k = 0;k<10;k++){
					if (check[k].successValue<full[0].successValue){
						check[k] = full[0];
					}
				}
			}			
		}
		if (displayInfo){
			float fitness= 0;
			for (int v = 0;v<10;v++){
				fitness += check[v].successValue;
			}
			System.out.println("Success Rate: " + (fitness/10));
		}
		for (int v = 0;v<10;v++){
			out[v] = check[v].layers;
		}
		out[10] = ganInput;
		return out;
	}

	public static FullLayer[] trainIndividualLayers(Layer[] input, Layer[] ganInput)
			throws IOException, InterruptedException {
		int length = input[4].neurons.length;
		float[] f = {loadRandomFloat(0, 100)};
		input = forward(f, input, false);
		float[] test = new float[length];
		for (int i = 0;i<length;i++){
			test[i] = input[4].neurons[i].value;
		}
		float[] k = {0};
		TrainingData trData = new TrainingData(test, k);
		TrainingData[] trDataArr = {trData};
		train2(1, 0.025f, trDataArr, ganInput, true);
		FullLayer full = new FullLayer(input, (ganInput[4].neurons[0].value));
		FullLayer gan = new FullLayer(ganInput);
		FullLayer[] out = {full, gan};
		return out;
	}

	public static Layer[][] returnAllButLast(Layer[][] layers){
		Layer[][] out = new Layer[layers.length-1][];
		for (int i = 0;i<layers.length-1;i++){
			out[i] = layers[i];
		}
		return out;
	}

	public static TrainingData[] loadGANInputs(int inputs) throws IOException {
		TrainingData[] out = new TrainingData[inputs];
		for (int z = 0;z<inputs;z++){
			int f = (int)StatUtil.RandomFloat(0, 3);
			if (f < 2){
				String input = reader.readLine();
				if (input == null){
					reader.reset();
					input = reader.readLine();
				}
				char[] arr = input.toCharArray();
				float[] data = new float[25];
				int i = 0;
				for (char b : arr){
					data[i] = (float)((int)b);
					i++;
				}
				for (;i<25;){
					data[i] = 0;
					i++;
				}
				float[] desiredOutput = {1};
				TrainingData output = new TrainingData(data, desiredOutput);
				out[z] = output;
			}
			else if (f == 2){
				String input = reader2.readLine();
				while (input == null || input.length()>25){
					reader2.reset();
					input = reader2.readLine();
				}
				char[] arr = input.toCharArray();
				float[] data = new float[25];
				int i = 0;
				for (char b : arr){
					data[i] = (float)((int)b);
					i++;
				}
				for (;i<25;){
					data[i] = 0;
					i++;
				}
				float[] desiredOutput = {0};
				TrainingData output = new TrainingData(data, desiredOutput);
				out[z] = output;
			}
			else {
				char[] arr = new char[25];
				for (int i = 0; i < arr.length;i++){
					arr[i] = Character.toChars((int)loadRandomFloat(64, 123))[0];
				}
				float[] data = new float[25];
				int i = 0;
				for (char b : arr){
					data[i] = (float)((int)b);
					i++;
				}
				for (;i<25;){
					data[i] = 0;
					i++;
				}
				float[] desiredOutput = {0};
				TrainingData output = new TrainingData(data, desiredOutput);
				out[z] = output;
			}
		}
		return out;
	}
	
	public static Layer[][] generateLayers(int amount){
		Layer[][] out = new Layer[amount][];
		for (int i = 0;i<amount;i++){
			Layer[] layers = new Layer[5];
			layers[0] = null; // Input Layer 0,10
			layers[1] = new Layer(1, 250, true); // Hidden Layer 10, 125
			layers[2] = new Layer(250, 500, true); // Hidden Layer 125, 750
			layers[3] = new Layer(500, 450, true); // Hidden Layer 750, 500
			layers[4] = new Layer(450, 25, true); // Output Layer 500, 2
			out[i] = layers;
		}
		return out;
	}

	public static Layer[][] duplicateAndRandomize(Layer[][] input, int multiply, float learning_rate)
			throws IOException, InterruptedException {
		int size = input.length;
		int totalSize = size*multiply;
		Layer[][] out = new Layer[(totalSize)][];
		int i = 0;
		for (;i<size;i++){
			out[i] = input[i];
		}
		for (;i<totalSize;i++){
			for (int k = 0;k<multiply;k++){
				while (i<totalSize){
					Layer[] l = input[k];
					l = backwardAndRandomize(learning_rate, l);
					out[i] = l;
					i++;
				}
			}
		}
		return out;
	}

	public static Layer[] backwardAndRandomize(float learning_rate, Layer[] input) throws IOException, InterruptedException {
    	
    	int number_layers = input.length;
		int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output
    	for(int i = 0; i < input[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = 0.5f;
    		float target = 1;
			float derivative = output-target;
			float delta = derivative*(output*(1-output));
			public_error = delta;
    		input[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < input[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = input[out_index-1].neurons[j].value;
				float error = delta*previous_output;
				input[out_index].neurons[i].cache_weights[j] = input[out_index].neurons[i].weights[j] - learning_rate*error;
				input[out_index].neurons[i].cache_bias = input[out_index].neurons[i].bias - learning_rate*error;
			}
		}

    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-2; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < input[i].neurons.length; j++) {
    			float output = input[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1, input);
    			float delta = (gradient_sum)*(output*(1-output));
    			input[i].neurons[j].gradient = delta;
				// And for all their weights
    			for(int k = 0; k < input[i].neurons[j].weights.length; k++) {
    				float previous_output = input[i-1].neurons[k].value;
    				float error = delta*previous_output;
					input[i].neurons[j].cache_weights[k] = input[i].neurons[j].weights[k] - (StatUtil.RandomFloat(-1, 1))*learning_rate*error;
					input[i].neurons[j].cache_bias = input[i].neurons[j].bias - (StatUtil.RandomFloat(-1, 1))*learning_rate*error;
				}
			}
    	}
    	
    	// Here we do another pass where we update all the weights/biases
    	for(int i = 0; i< input.length;i++) {
    		for(int j = 0; j < input[i].neurons.length;j++) {
				input[i].neurons[j].update_weight();
				input[i].neurons[j].update_biases();
    		}
		}

		return input;
    	
    }

	// Function to load a random float from 1-9999
	public static float loadRandomFloat(int leftLimit, int rightLimit) {
		int generatedInteger = leftLimit + (int) (new Random().nextFloat() * (rightLimit - leftLimit));
		return(float)generatedInteger;
	}

    // The idea is that you calculate a gradient and cache the updated weights in the neurons.
    // When ALL the neurons new weight have been calculated we refresh the neurons.
    // Meaning we do the following:
    // Calculate the output layer weights, calculate the hidden layer weight then update all the weights
	public static Layer[] backward(float learning_rate,TrainingData tData, Layer[] input) throws IOException, InterruptedException {
    	
    	int number_layers = input.length;
		int out_index = number_layers-1;
    	
    	// Update the output layers 
    	// For each output
    	for(int i = 0; i < input[out_index].neurons.length; i++) {
    		// and for each of their weights
    		float output = input[out_index].neurons[i].value;
    		float target = tData.expectedOutput[i];
			float derivative = output-target;
			float delta = derivative*(output*(1-output));
			public_error = delta;
    		input[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < input[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = input[out_index-1].neurons[j].value;
				float error = delta*previous_output;
				input[out_index].neurons[i].cache_weights[j] = input[out_index].neurons[i].weights[j] - learning_rate*error;
				input[out_index].neurons[i].cache_bias = input[out_index].neurons[i].bias - learning_rate*error;
			}
		}

    	
    	//Update all the subsequent hidden layers
    	for(int i = out_index-2; i > 0; i--) {
    		// For all neurons in that layers
    		for(int j = 0; j < input[i].neurons.length; j++) {
    			float output = input[i].neurons[j].value;
    			float gradient_sum = sumGradient(j,i+1, input);
    			float delta = (gradient_sum)*(output*(1-output));
    			input[i].neurons[j].gradient = delta;
				// And for all their weights
    			for(int k = 0; k < input[i].neurons[j].weights.length; k++) {
    				float previous_output = input[i-1].neurons[k].value;
    				float error = delta*previous_output;
					input[i].neurons[j].cache_weights[k] = input[i].neurons[j].weights[k] - learning_rate*error;
					input[i].neurons[j].cache_bias = input[i].neurons[j].bias - learning_rate*error;
				}
			}
    	}
    	
    	// Here we do another pass where we update all the weights/biases
    	for(int i = 0; i< input.length;i++) {
    		for(int j = 0; j < input[i].neurons.length;j++) {
				input[i].neurons[j].update_weight();
				input[i].neurons[j].update_biases();
    		}
		}

		return input;
    	
    }
    // This function sums up all the gradient connecting a given neuron in a given layer
    public static float sumGradient(int n_index,int l_index, Layer[] input) {
    	float gradient_sum = 0;
    	Layer current_layer = input[l_index];
    	for(int i = 0; i < current_layer.neurons.length; i++) {
    		Neuron current_neuron = current_layer.neurons[i];
    		gradient_sum += current_neuron.weights[n_index]*current_neuron.gradient;
    	}
    	return gradient_sum;
	}
	
	//	Check progress of training
	public static int checkGANProgress(Layer[] ganInput, int iterations, boolean showData) throws IOException {
		int successes = 0;
		for (int k = 0;k<iterations;k++){
			TrainingData[] trData = loadGANInputs(1);
			if (showData){
				System.out.println("Input: ");
				for (int i = 0;i<trData[0].data.length;i++){
					System.out.print(Character.toChars((int)(trData[0].data[i])));
				}
			}
			forward(trData[0].data, ganInput, true);
			if (showData){
				System.out.println("Value: " + ganInput[4].neurons[0].value);
				System.out.println("Expected: " + trData[0].expectedOutput[0]);
			}
			if (Math.round(ganInput[4].neurons[0].value) == trData[0].expectedOutput[0]){
				successes++;
			}
		}
		reader.reset();
		reader2.reset();
		return successes;
	}

	//	Check success rate by loading new test data
	//	This function is used to train data by pushing it forward and backward
	//	It then checks the success rate and resets the layers to the highest success rate weight settings if the success rate is lower than its record
    public static Layer[] train(int training_iterations,float learning_rate, TrainingData[] traningData, Layer[] input, boolean sigmoid)
			throws IOException, InterruptedException {
    	for(int i = 1; i < training_iterations+1; i++) {		
    		for(int j = 0; j < traningData.length; j++) {
    			input = forward(traningData[j].data, input, sigmoid);
				input = backward(learning_rate,traningData[j], input);
				if (j%25 == 0 && j != 0){
					float currentSuccessRate = checkGANProgress(input, 100, false);
					if (debugMode){
						csvWriter.append(Float.toString(currentSuccessRate));
						csvWriter.append("\n");
					}
					else if (currentSuccessRate < cachedLayerSuccessRate) {
						input = ganLayersCache;
					}
					else if (currentSuccessRate > cachedLayerSuccessRate){
						System.out.println("New Record: %" + (currentSuccessRate));
						if (debugMode){
							csvWriter.flush();
						}
						ganLayersCache = input;
						cachedLayerSuccessRate = currentSuccessRate;
						algorithmImprovmentFunction *= 1.05;
					}
					else{
						ganLayersCache = input;
						cachedLayerSuccessRate = currentSuccessRate;
					}
				}
			}
		}
		input = ganLayersCache;
		return input;
	}

	public static Layer[] train2(int training_iterations,float learning_rate, TrainingData[] traningData, Layer[] input, boolean sigmoid)
			throws IOException, InterruptedException {
    	for(int i = 1; i < training_iterations+1; i++) {		
    		for(int j = 0; j < traningData.length; j++) {
    			input = forward(traningData[j].data, input, sigmoid);
				input = backward(learning_rate,traningData[j], input);
				}
			}
		return input;
	}
	
	
	//	Train without checking success rate
	public static Layer[] initialTrain(int training_iterations, float learning_rate, TrainingData[] traningData, boolean sigmoid, Layer[] input)
	throws IOException, InterruptedException {
		for(int i = 0; i < training_iterations; i++) {
			time = (i * 100 / training_iterations);
			i_stat = i;
			for(int j = 0; j < traningData.length; j++) {
				input = forward(traningData[j].data, input, sigmoid);
				input = backward(learning_rate,traningData[j], input);
			}
		}
		return input;	
	}
}