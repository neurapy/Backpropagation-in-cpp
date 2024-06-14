#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <random>
#include "dataset.h"
#include "neural_network.h"

float calculate_accuracy(const NeuralNetwork& nn, const std::vector<std::vector<uint8_t>>& images, const std::vector<uint8_t>& labels) {
    int correct_predictions = 0;

    for (size_t i = 0; i < images.size(); ++i) {
        std::vector<float> input(images[i].begin(), images[i].end());
        auto output = nn.forward(input);

        int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
        if (predicted_label == labels[i]) {
            correct_predictions++;
        }
    }

    return static_cast<float>(correct_predictions) / images.size();
}

void display_random_images(const NeuralNetwork& nn, const std::vector<std::vector<uint8_t>>& images, const std::vector<uint8_t>& labels, size_t num_images) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, images.size() - 1);

    for (size_t i = 0; i < num_images; ++i) {
        int index = dis(gen);
        std::vector<float> input(images[index].begin(), images[index].end());
        auto output = nn.forward(input);

        std::cout << "Image " << index << " (label: " << static_cast<int>(labels[index]) << "):" << std::endl;
        print_image(images, index);

        std::cout << "Predicted likelihoods: ";
        for (size_t j = 0; j < output.size(); ++j) {
            std::cout << std::fixed << std::setprecision(4) << output[j] << " ";
        }
        std::cout << std::endl << std::endl;
    }
}

int main() {
    std::string train_images_path = "train-images-idx3-ubyte";
    std::string train_labels_path = "train-labels-idx1-ubyte";
    std::string test_images_path = "t10k-images-idx3-ubyte";
    std::string test_labels_path = "t10k-labels-idx1-ubyte";
    
    auto train_images = read_mnist_images(train_images_path);
    auto train_labels = read_mnist_labels(train_labels_path);
    auto test_images = read_mnist_images(test_images_path);
    auto test_labels = read_mnist_labels(test_labels_path);
    
    std::cout << "Loaded " << train_images.size() << " training images and " << train_labels.size() << " training labels." << std::endl;
    std::cout << "Loaded " << test_images.size() << " test images and " << test_labels.size() << " test labels." << std::endl;

    NeuralNetwork nn(28 * 28, 512, 256, 128, 10);
    size_t batch_size = 128;
    float learning_rate = 0.0001;

    for (size_t epoch = 0; epoch < 10; ++epoch) {
        float total_loss = 0.0;

        for (size_t i = 0; i < train_images.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, train_images.size());

            int correct_predictions = 0;
            for (size_t j = i; j < end; ++j) {
                std::vector<float> input(train_images[j].begin(), train_images[j].end());
                std::vector<float> target(10, 0.0);
                target[train_labels[j]] = 1.0; 

                auto output = nn.forward(input);

                if (j == i) {
                    std::cout << "Image 1 (correct label: " << static_cast<int>(train_labels[j]) << "): ";
                    std::cout << std::fixed << std::setprecision(4);
                    for (auto val : output) {
                        std::cout << val << " ";
                    }
                    std::cout << std::endl;
                }

                nn.backward(input, target, learning_rate);
                total_loss += nn.cross_entropy_loss(output, target);

                int predicted_label = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
                if (predicted_label == train_labels[j]) {
                    correct_predictions++;
                }
            }

            float batch_accuracy = static_cast<float>(correct_predictions) / (end - i);
            std::cout << "Epoch " << epoch + 1 << ", Batch " << (i / batch_size) + 1 << ", Loss: " << total_loss / (end - i) << ", Accuracy: " << batch_accuracy * 100.0f << "%" << std::endl;
            total_loss = 0.0;
        }

        std::cout << "Calculating total epoche-stats now..." << std::endl;
        std::cout << "==TOTAL STATS==" << std::endl;
        float train_accuracy = calculate_accuracy(nn, train_images, train_labels);
        std::cout << "Epoch " << epoch + 1 << ", Training Accuracy: " << train_accuracy * 100.0f << "%" << std::endl;

        float test_accuracy = calculate_accuracy(nn, test_images, test_labels);
        std::cout << "Epoch " << epoch + 1 << ", Test Accuracy: " << test_accuracy * 100.0f << "%" << std::endl;

        display_random_images(nn, test_images, test_labels, 5);
    }

    return 0;
}
