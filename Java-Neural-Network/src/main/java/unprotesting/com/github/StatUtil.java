package unprotesting.com.github;

public class StatUtil {
    // Get a random numbers between min and max
    public static float RandomFloat(float min, float max) {
        float a = (float) Math.random();
        float num = min + (float) Math.random() * (max - min);
        if(a < 0.5)
            return num;
        else
            return -num;
    }
    
    // Sigmoid function
    public static float Sigmoid(float x) {
        return (float) (1/(1+Math.pow(Math.E, -x)));
    }
    
    // Derivative of the sigmoid function
    public static float SigmoidDerivative(float x) {
        return Sigmoid(x)*(1-Sigmoid(x));
    }

    public static float leakyRLEU(float x){
        if (x>122){
            return 122;
        }
        if (x<65){
            return 65;
        }
        else{
            return x;
        }
    }

    public static float leakyRELUDerivative(float x){
        if (x>0){
            return 1f;
        }
        else{
            return 0.25f;
        }
    }
    
    // Used for the backpropagation
    public static float squaredError(float output,float target) {
    	return (float) (0.5*Math.pow(2,(target-output)));
    }
    
    // Used to calculate the overall error rate (not yet used)
    public static float sumSquaredError(float[] outputs,float[] targets) {
    	float sum = 0;
    	for(int i=0;i<outputs.length;i++) {
    		sum += squaredError(outputs[i],targets[i]);
    	}
    	return sum;
    }
}