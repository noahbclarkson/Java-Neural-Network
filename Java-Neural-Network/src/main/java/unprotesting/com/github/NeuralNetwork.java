package unprotesting.com.github;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {
	// Variable Declaration
	
    // Layers
    static Layer[] layers;
    
    // Training data
	
	static TrainingData[] tData1, tData200, tData5000; // My changes

	static String[] list;
   
    // Main Method
    public static void main(String[] args) throws InterruptedException {
    	// My changes
        // Set the Min and Max weight value for all Neurons
    	Neuron.setRangeWeight(-1,1);
    	
    	// Create the layers
    	layers = new Layer[5];
    	layers[0] = null; // Input Layer 0,10
		layers[1] = new Layer(10,24); // Hidden Layer 10,24
		layers[2] = new Layer(24,25); // Hidden Layer 24,25
		layers[3] = new Layer(25,32); // Hidden Layer 25,32
    	layers[4] = new Layer(32,2); // Output Layer 32,2
        
    	// Create the training data
		tData200 = loadInputs(200);
    	
        System.out.println("============");
        System.out.println("Output before training - 1");
        System.out.println("============");
        for(int i = 0; i < tData200.length; i++) {
            forward(tData200[i].data);
            System.out.println(layers[4].neurons[0].value + " - " +  layers[4].neurons[1].value);
		}
		
		Thread.sleep(5);
       
        train(25000, 0.005f, tData200);

        System.out.println("============");
        System.out.println("Output after training - 1");
        System.out.println("============");
        for(int i = 0; i < tData200.length; i++) {
            forward(tData200[i].data);
            System.out.println((layers[4].neurons[0].value)+ " - "+ (layers[4].neurons[1].value) + " =(this should equal)= " + list[i]);
		}

		Thread.sleep(5);

		tData5000 = loadInputs(5000);
    	
        System.out.println("============");
        System.out.println("Output before training - 2");
        System.out.println("============");
        for(int i = 0; i < tData5000.length; i++) {
            forward(tData5000[i].data);
            System.out.println((layers[4].neurons[0].value)+ " - " +(layers[4].neurons[1].value) + " =(this should equal)= " + list[i]);
		}
		
		Thread.sleep(5);
       
        train(1000, 0.0015f, tData5000);

        System.out.println("============");
        System.out.println("Output after training - 2");
        System.out.println("============");
        for(int i = 0; i < tData5000.length; i++) {
            forward(tData5000[i].data);
            System.out.println((layers[4].neurons[0].value)+ " - "+ (layers[4].neurons[1].value) + " =(this should equal)= " + list[i]);
		}

		Thread.sleep(100);

		tData1 = loadInputs(1);
		System.out.println("============");
        System.out.println("Output before testing");
        System.out.println("============");
		float[] test = tData1[0].data;
		System.out.println(Arrays.toString(test));

		train(5, 0.001f, tData1);
		System.out.println("============");
        System.out.println("Output after testing");
        System.out.println("============");
        for(int i = 0; i < tData1.length; i++) {
            forward(tData1[i].data);
            System.out.println((layers[4].neurons[0].value)+ " - "+ (layers[4].neurons[1].value) + " =(this should equal)= " + list[i]);
		}
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
				// sum += layers[i].neurons[j].bias;
				layers[i].neurons[j].value = StatUtil.Sigmoid(sum);
			}
		}
	}

	public static TrainingData[] loadInputs(int inputs) {
		TrainingData[] outputTrainingData = new TrainingData[inputs];
		list = new String[inputs];
		int i = 0;
		for (;i<inputs;){
			float temp = loadRandomFloat();
			if (temp > 7000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				float[] sortedInput = ascendingBubbleSortFloatArray(z);
				outputTrainingData[i] = new TrainingData(sortedInput, new float[]{1, 0});
				list[i] = "one, zero";
				i++;
			}
			if (temp <= 3000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				float[] sortedInput = descendingBubbleSortFloatArray(z);
				outputTrainingData[i] = new TrainingData(sortedInput, new float[]{0, 0});
				list[i] = "zero, zero";
				i++;
			}
			if (7000 > temp && temp > 3000){
				float[] z = { loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(),  loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat(), loadRandomFloat()};
				boolean isasorted = isAscendingSorted(z);
				boolean isdsorted = isDecsendingSorted(z);
				if (isdsorted == true){
					outputTrainingData[i] = new TrainingData(z, new float[]{0, 0});
					list[i] = "zero, zero";
					i++;
				}
				else if (isasorted == true){
					outputTrainingData[i] = new TrainingData(z, new float[]{1, 0});
					list[i] = "one, zero";
					i++;
				}
				else {
					outputTrainingData[i] = new TrainingData(z, new float[]{0, 1});
					list[i] = "zero, one";
					i++;
				}
			}
		}
		return outputTrainingData;

	}

	public static boolean isAscendingSorted(float[] a) {
		for (int i = 0; i < a.length - 1; i++) {
			if (a[i] > a[i + 1]) {
				return false; // It is proven that the array is not sorted.
			}
		}
	
		return true; // If this part has been reached, the array must be sorted.
	}

	public static boolean isDecsendingSorted(float[] a) {
		for (int i = 0; i < a.length - 1; i++) {
			if (a[i] < a[i + 1]) {
				return false; // It is proven that the array is not sorted.
			}
		}
	
		return true; // If this part has been reached, the array must be sorted.
	}

	public static float loadRandomFloat() {
		int leftLimit = 1;
		int rightLimit = 9999;
		int generatedInteger = leftLimit + (int) (new Random().nextFloat() * (rightLimit - leftLimit));
		return(float)generatedInteger;
	}

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
    public static void backward(float learning_rate,TrainingData tData) {
    	
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
    		layers[out_index].neurons[i].gradient = delta;
    		for(int j = 0; j < layers[out_index].neurons[i].weights.length;j++) { 
    			float previous_output = layers[out_index-1].neurons[j].value;
    			float error = delta*previous_output;
				layers[out_index].neurons[i].cache_weights[j] = layers[out_index].neurons[i].weights[j] - learning_rate*error;
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
    			}
    		}
    	}
    	
    	// Here we do another pass where we update all the weights
    	for(int i = 0; i< layers.length;i++) {
    		for(int j = 0; j < layers[i].neurons.length;j++) {
    			layers[i].neurons[j].update_weight();
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
 
    
    // This function is used to train being forward and backward.
    public static void train(int training_iterations,float learning_rate, TrainingData[] traningData) {
    	for(int i = 0; i < training_iterations; i++) {
    		for(int j = 0; j < traningData.length; j++) {
    			forward(traningData[j].data);
				backward(learning_rate,traningData[j]);
			}
			if (i%311 == 0 && i != 0){
				System.out.println("Percentage complete: %" + (i*100/training_iterations));
			}
			
		}
		
    }
}