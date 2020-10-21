package unprotesting.com.github;

public class FullLayer {
    public Layer[] layers;
    public float successValue;

    FullLayer(Layer[] layers, float successValue){
        this.layers = layers;
        this.successValue = successValue;
    }

    FullLayer(Layer[] ganLayers){
        this.layers = ganLayers;
    }
}
