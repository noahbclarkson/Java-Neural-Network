package unprotesting.com.github;

public class Layer {
    public Neuron[] neurons;
	
	// Constructor for the hidden and output layer
	public Layer(int inNeurons,int numberNeurons,boolean sigmoid) {
		this.neurons = new Neuron[numberNeurons];
		if (sigmoid){
			for(int i = 0; i < numberNeurons; i++) {
				float[] weights = new float[inNeurons];
				for(int j = 0; j < inNeurons; j++) {
					weights[j] = StatUtil.RandomFloat(-1, 1);
				}
				neurons[i] = new Neuron(weights,StatUtil.RandomFloat(-1, 1));
			}
		}
		else{
			for(int i = 0; i < numberNeurons; i++) {
				float[] weights = new float[inNeurons];
				for(int j = 0; j < inNeurons; j++) {
					weights[j] = StatUtil.RandomFloat(Neuron.minWeightValue, Neuron.maxWeightValue);
				}
				neurons[i] = new Neuron(weights,StatUtil.RandomFloat(-1, 1));
			}
		}

	}
	
	
	// Constructor for the input layer
	public Layer(float input[]) {
		this.neurons = new Neuron[input.length];
		for(int i = 0; i < input.length; i++) {
			this.neurons[i] = new Neuron(input[i]);
		}
	}
}