#pragma once
#include <vector>
#include <string>
#include <cstdint>

std::vector<uint8_t> read_mnist_labels(const std::string& filepath);
std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& filepath);
void print_image(const std::vector<std::vector<uint8_t>>& images, int index);
