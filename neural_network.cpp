#include "neural_network.h"
#include <cmath>
#include <random>
#include <algorithm>

NeuralNetwork::NeuralNetwork(size_t input_size, size_t hidden_size1, size_t hidden_size2, size_t hidden_size3, size_t output_size) {

    // Construction of the Network
    W1.resize(hidden_size1, std::vector<float>(input_size));
    W2.resize(hidden_size2, std::vector<float>(hidden_size1));
    W3.resize(hidden_size3, std::vector<float>(hidden_size2));
    W4.resize(output_size, std::vector<float>(hidden_size3));
    b1.resize(hidden_size1);
    b2.resize(hidden_size2);
    b3.resize(hidden_size3);
    b4.resize(output_size);

    // Intermediate Values for the Forward and Backward pass
    hidden1.resize(hidden_size1);
    hidden2.resize(hidden_size2);
    hidden3.resize(hidden_size3);
    output.resize(output_size);
    hidden_gradient1.resize(hidden_size1);
    hidden_gradient2.resize(hidden_size2);
    hidden_gradient3.resize(hidden_size3);
    output_gradient.resize(output_size);

    // Initialize the Weights and Biases
    initialize_weights(W1, hidden_size1, input_size);
    initialize_weights(W2, hidden_size2, hidden_size1);
    initialize_weights(W3, hidden_size3, hidden_size2);
    initialize_weights(W4, output_size, hidden_size3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.1, 0.1);

    for (auto& val : b1)
        val = dis(gen);
    for (auto& val : b2)
        val = dis(gen);
    for (auto& val : b3)
        val = dis(gen);
    for (auto& val : b4)
        val = dis(gen);
}

void NeuralNetwork::initialize_weights(std::vector<std::vector<float>>& weights, size_t rows, size_t cols) {
    // Using Xavier Initialization as i want to use Sigmoid Activations (vanishing gradient)
    std::random_device rd;
    std::mt19937 gen(rd());
    float stddev = std::sqrt(2.0 / (rows + cols));
    std::normal_distribution<> dis(0, stddev);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            weights[i][j] = dis(gen);
        }
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) const {
    for (size_t i = 0; i < hidden1.size(); ++i) {
        hidden1[i] = b1[i];
        for (size_t j = 0; j < input.size(); ++j) {
            hidden1[i] += W1[i][j] * input[j];
        }
        hidden1[i] = sigmoid(hidden1[i]);
    }

    for (size_t i = 0; i < hidden2.size(); ++i) {
        hidden2[i] = b2[i];
        for (size_t j = 0; j < hidden1.size(); ++j) {
            hidden2[i] += W2[i][j] * hidden1[j];
        }
        hidden2[i] = sigmoid(hidden2[i]);
    }

    for (size_t i = 0; i < hidden3.size(); ++i) {
        hidden3[i] = b3[i];
        for (size_t j = 0; j < hidden2.size(); ++j) {
            hidden3[i] += W3[i][j] * hidden2[j];
        }
        hidden3[i] = sigmoid(hidden3[i]);
    }

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = b4[i];
        for (size_t j = 0; j < hidden3.size(); ++j) {
            output[i] += W4[i][j] * hidden3[j];
        }
    }

    return softmax(output);
}

void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate) {
    // Softmax Derivate
    for (size_t i = 0; i < output.size(); ++i) {
        output_gradient[i] = output[i] - target[i];
    }

    // Sigmoid Derivate 
    for (size_t i = 0; i < hidden3.size(); ++i) {
        hidden_gradient3[i] = 0;
        for (size_t j = 0; j < output.size(); ++j) {
            hidden_gradient3[i] += output_gradient[j] * W4[j][i];
        }
        hidden_gradient3[i] *= sigmoid_derivative(hidden3[i]);
    }

    for (size_t i = 0; i < hidden2.size(); ++i) {
        hidden_gradient2[i] = 0;
        for (size_t j = 0; j < hidden3.size(); ++j) {
            hidden_gradient2[i] += hidden_gradient3[j] * W3[j][i];
        }
        hidden_gradient2[i] *= sigmoid_derivative(hidden2[i]);
    }

    for (size_t i = 0; i < hidden1.size(); ++i) {
        hidden_gradient1[i] = 0;
        for (size_t j = 0; j < hidden2.size(); ++j) {
            hidden_gradient1[i] += hidden_gradient2[j] * W2[j][i];
        }
        hidden_gradient1[i] *= sigmoid_derivative(hidden1[i]);
    }

    update_weights(learning_rate, input);
}

void NeuralNetwork::update_weights(float learning_rate, const std::vector<float>& input) {
    // Subtract the gradient to take a step in the opposite direction
    for (size_t i = 0; i < W4.size(); ++i) {
        for (size_t j = 0; j < W4[i].size(); ++j) {
            W4[i][j] -= learning_rate * output_gradient[i] * hidden3[j];
        }
        b4[i] -= learning_rate * output_gradient[i];
    }

    for (size_t i = 0; i < W3.size(); ++i) {
        for (size_t j = 0; j < W3[i].size(); ++j) {
            W3[i][j] -= learning_rate * hidden_gradient3[i] * hidden2[j];
        }
        b3[i] -= learning_rate * hidden_gradient3[i];
    }

    for (size_t i = 0; i < W2.size(); ++i) {
        for (size_t j = 0; j < W2[i].size(); ++j) {
            W2[i][j] -= learning_rate * hidden_gradient2[i] * hidden1[j];
        }
        b2[i] -= learning_rate * hidden_gradient2[i];
    }

    for (size_t i = 0; i < W1.size(); ++i) {
        for (size_t j = 0; j < W1[i].size(); ++j) {
            W1[i][j] -= learning_rate * hidden_gradient1[i] * input[j];
        }
        b1[i] -= learning_rate * hidden_gradient1[i];
    }
}

float NeuralNetwork::sigmoid(float x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

float NeuralNetwork::sigmoid_derivative(float x) const {
    return x * (1.0 - x);
}


float NeuralNetwork::cross_entropy_loss(const std::vector<float>& output, const std::vector<float>& target) const {
    float loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        loss -= target[i] * std::log(output[i] + 1e-10); // small epsilon to avoid log(0)
        // (only rewarding not punishing ensuring it doesent kill me later)
    }
    return loss;
}

std::vector<float> NeuralNetwork::softmax(const std::vector<float>& x) const {
    std::vector<float> result(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }
    return result;
}
