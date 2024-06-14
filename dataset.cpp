#include "dataset.h"
#include <iostream>
#include <fstream>
#include <stdexcept>

std::vector<uint8_t> read_mnist_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        int32_t magic_number = 0;
        int32_t number_of_items = 0;
        
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&number_of_items), 4);
        
        magic_number = __builtin_bswap32(magic_number);
        number_of_items = __builtin_bswap32(number_of_items);
        
        std::vector<uint8_t> labels(number_of_items);
        file.read(reinterpret_cast<char*>(labels.data()), number_of_items);
        return labels;
    } else {
        throw std::runtime_error("Unable to open file " + filepath);
    }
}

std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (file.is_open()) {
        int32_t magic_number = 0;
        int32_t number_of_images = 0;
        int32_t rows = 0;
        int32_t cols = 0;
        
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&number_of_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        
        magic_number = __builtin_bswap32(magic_number);
        number_of_images = __builtin_bswap32(number_of_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        
        std::vector<std::vector<uint8_t>> images(number_of_images, std::vector<uint8_t>(rows * cols));
        for (int i = 0; i < number_of_images; ++i) {
            file.read(reinterpret_cast<char*>(images[i].data()), rows * cols);
        }

        return images;
    } else {
        throw std::runtime_error("Unable to open file " + filepath);
    }
}

// Output the image as ANCII Art just for fun
void print_image(const std::vector<std::vector<uint8_t>>& images, int index) {
    if (index >= images.size() || index < 0) {
        std::cerr << "Index out of range." << std::endl;
        return;
    }

    for (int row = 0; row < 28; ++row) {
        for (int col = 0; col < 28; ++col) {
            std::cout <<
                (images[index][row * 28 + col] > 210 ? '#' :
                (images[index][row * 28 + col] > 180 ? '*' :
                (images[index][row * 28 + col] > 150 ? '=' :
                (images[index][row * 28 + col] > 130 ? '+' :
                (images[index][row * 28 + col] > 70 ? '-' :
                (images[index][row * 28 + col] > 25 ? '.' : ' '))))));
        }
        std::cout << std::endl;
    }
}
