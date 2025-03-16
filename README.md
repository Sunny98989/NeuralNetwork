# NeuralNetwork
This project is a simple neural network built from scratch using NumPy, designed for image classification. Specifically, it's used to recognize handwritten digits from the MNIST dataset, which contains 28x28 grayscale images of digits ranging from 0 to 9.

The network has a basic two-layer structure:

Input Layer: Each image is first flattened into a 784-dimensional vector (since 28 × 28 = 784) so it can be fed into the network.
Hidden Layer: This layer processes the input using the ReLU (Rectified Linear Unit) activation function to introduce non-linearity.
Output Layer: The final layer uses the softmax function to output a probability distribution across the 10 digit classes (0 through 9).
The network is trained using gradient descent, aiming to minimize the cross-entropy loss function. The code handles everything from forward propagation to backpropagation and parameter updates.

As the model trains, it prints out accuracy updates to show how it's learning over time. There's also a section that visualizes how the hidden layer neurons activate for a sample input—giving a cool insight into what the network is "looking at" while making predictions.

