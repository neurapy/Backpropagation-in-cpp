#pragma once
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t hidden_size1, size_t hidden_size2, size_t hidden_size3, size_t output_size);
    std::vector<float> forward(const std::vector<float>& input) const;
    void backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate);
    void update_weights(float learning_rate, const std::vector<float>& input);
    float cross_entropy_loss(const std::vector<float>& output, const std::vector<float>& target) const;

private:
    std::vector<std::vector<float>> W1, W2, W3, W4;
    std::vector<float> b1, b2, b3, b4;
    mutable std::vector<float> hidden1, hidden2, hidden3, output;
    mutable std::vector<float> hidden_gradient1, hidden_gradient2, hidden_gradient3, output_gradient;

    float sigmoid(float x) const;
    float sigmoid_derivative(float x) const;
    std::vector<float> softmax(const std::vector<float>& x) const;
    void initialize_weights(std::vector<std::vector<float>>& weights, size_t rows, size_t cols);
};
