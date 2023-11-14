
# Java Neural Network
[![Generic badge](https://img.shields.io/badge/Java-NeuralNetwork-green.svg)](https://shields.io/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://shields.io/)
[![Version](https://img.shields.io/badge/version-1.2.1-blue.svg)](https://shields.io/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://shields.io/)

## Overview
Java Neural Network is an innovative project aimed at implementing a neural network purely in Java without the use of external libraries or APIs. It is designed to be a flexible and extendable framework for various neural network experiments.

### Features
- Pure Java implementation
- Customizable neural network architecture
- No external dependencies
- Simple and intuitive API
- Comprehensive documentation and examples

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
- Java JDK 1.8 or later
- Basic understanding of neural networks and Java programming

### Installation
1. Clone the repository:
   ```
   git clone https://github.com/noahbclarkson/Java-Neural-Network.git
   ```
2. Navigate to the cloned directory:
   ```
   cd Java-Neural-Network
   ```
3. Compile and run the project using your preferred Java IDE or command line tools.

## Usage
The Java Neural Network can be used to create, train, and test neural network models for various applications such as pattern recognition, data classification, and more.

### Example
Here is a simple example of using the Java Neural Network to train a model:

```java
import unprotesting.com.github.*;

public class Example {
    public static void main(String[] args) {
        // Create and train the neural network
        NeuralNetwork network = new NeuralNetwork();
        network.train(trainingData);

        // Test the neural network
        float[] testInput = { ... };
        float[] output = network.predict(testInput);
        System.out.println("Predicted output: " + Arrays.toString(output));
    }
}
```

## Authors
- **Noah B. Clarkson** - *Initial work* - [noahbclarkson](https://github.com/noahbclarkson)

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE.md](LICENSE.md) file for details

## Note

This is a template for other neural networks but standalone, it checks whether a float array is sorted ascendingly, descendingly or not at all.

- A result of 1, 0, 0 means the array the AI thinks the array is sorted ascendingly. 
 
- A result of 0, 1, 0 means the the AI thinks the array is sorted descendingly. 
 
- A result of 0, 0, 1 means the AI thinks the array is not sorted.
 
- A result of 0, 0, 0 means the AI does not know / an error.
