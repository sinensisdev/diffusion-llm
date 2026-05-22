// print to console
#include <iostream>
// --

// read_file
#include <filesystem>
#include <fstream>
#include <string>
// --

// data structures
#include <unordered_map>
#include <vector>
// --

// random
#include <cstdlib>
// --

// algorithms
#include <algorithm>
// --

#include <limits>
#include <cmath>

std::string read_file(std::filesystem::path path)
{
    // Open the stream to 'lock' the file.
    std::ifstream ifstream(path, std::ios::in | std::ios::binary);

    // Obtain the size of the file.
    const auto size = std::filesystem::file_size(path);

    // Create a buffer.
    std::string result(size, '\0');

    // Read the whole file into the buffer.
    ifstream.read(result.data(), size);

    return result;
}

std::string translate_data(const std::vector<int>& data, std::unordered_map<int, char>& int_char_vocabulary, int data_size)
{
    std::string output(data_size, '\0');
    for (int i = 0; i < data_size; i++)
        output[i] = int_char_vocabulary[data[i]];
    return output;
}


int randomRange(int min, int max) {
    return min + rand() % (max - min + 1);
}

void mask(const std::vector<int>& data, std::vector<int>& masked_data, int chance_out_of_hundred, int mask_token)
{
    masked_data = data;
    for (int i = 0; i < data.size(); i++)
    {
        if (randomRange(1, 100) <= chance_out_of_hundred)
            masked_data[i] = mask_token;
    }
}

void forward(const std::vector<int>& original, const std::vector<int>& input, const std::vector<std::vector<float>>& matrix, std::vector<int>& output, int vocab_size, int mask_token)
{
    output.resize(input.size());
    int index_of_highest_float_temp;
    float highest_float_temp;
    int inputValue;
    for (int i = 0; i < input.size(); i++)
    {
        if (input[i] != mask_token)
        {
            output[i] = original[i];
            continue;
        }
        inputValue = input[i];
        index_of_highest_float_temp = 0;
        highest_float_temp = matrix[inputValue][0];
        for (int j = 1; j < vocab_size; j++)
        {
            if (matrix[inputValue][j] > highest_float_temp)
            {
                index_of_highest_float_temp = j;
                highest_float_temp = matrix[inputValue][j];
            }
        }
        output[i] = index_of_highest_float_temp;
    }
}

float loss(const std::vector<int>& masked_data, const std::vector<int>& original_data, const std::vector<std::vector<float>>& matrix, int data_size, int vocab_size, int mask_token)
{
    float negative_log_sum = 0.0f;
    float softmax_temp;
    float sum_temp;
    int masked_count = 0;
    for (int i = 0; i < data_size; i++)
    {
        if (masked_data[i] != mask_token)
            continue;
        masked_count++;

        float max_val = matrix[masked_data[i]][0];
        for (int j = 1; j < vocab_size; j++)
            if (matrix[masked_data[i]][j] > max_val)
                max_val = matrix[masked_data[i]][j];

        sum_temp = 0.0f;
        for (int j = 0; j < vocab_size; j++)
            sum_temp += std::exp(matrix[masked_data[i]][j] - max_val);
        softmax_temp = std::exp(matrix[masked_data[i]][original_data[i]] - max_val) / sum_temp;

        negative_log_sum -= std::log(softmax_temp);
    }
    return negative_log_sum / masked_count;
}

void train(const std::vector<int>& masked_data, const std::vector<int>& original_data, std::vector<std::vector<float>>& matrix, int data_size, int vocab_size, int mask_token, float learning_rate)
{
    float sum_temp, grad_temp, softmax_j, grad_j;
    for (int i = 0; i < data_size; i++)
    {
        if (masked_data[i] != mask_token)
            continue;

        float max_val = matrix[masked_data[i]][0];
        for (int j = 1; j < vocab_size; j++)
            if (matrix[masked_data[i]][j] > max_val)
                max_val = matrix[masked_data[i]][j];

        sum_temp = 0.0f;
        for (int j = 0; j < vocab_size; j++)
            sum_temp += std::exp(matrix[masked_data[i]][j] - max_val);

        for (int j = 0; j < vocab_size; j++)
        {
            softmax_j = std::exp(matrix[masked_data[i]][j] - max_val) / sum_temp;
            grad_j = softmax_j;
            if (j == original_data[i])
                grad_j = softmax_j - 1;
            matrix[masked_data[i]][j] -= learning_rate * grad_j;
        }
    }
}

std::vector<std::vector<float>> embed(const std::vector<int>& tokens, const std::vector<std::vector<float>>& embedding_layer)
{
    std::vector<std::vector<float>> output;
    output.resize(tokens.size());
    for (int i = 0; i < tokens.size(); i++)
        output[i] = embedding_layer[tokens[i]];
    return output;
}

void add_positional_encoding(std::vector<std::vector<float>>& embedding, int d_model)
{
    for (int pos = 0; pos < embedding.size(); pos++)
    {
        for (int i = 0; i < d_model / 2; i++)
        {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / d_model);
            embedding[pos][2*i]   += std::sin(angle);
            embedding[pos][2*i+1] += std::cos(angle);
        }
    }
}

std::vector<float> matvec(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec)
{
    // multiply matrix by vec, return result vector
}

int main()
{
    srand(time(0));

    std::unordered_map<char, int> char_int_vocabulary;
    std::unordered_map<int, char> int_char_vocabulary;
    uint next_vocabulary_int_value = 0;

    std::string str_data = read_file("data.txt");
    std::vector<int> data;
    int data_size = str_data.size();
    data.resize(data_size);
    for (int i = 0; i < data_size; i++)
    {
        if (char_int_vocabulary.find(str_data[i]) == char_int_vocabulary.end()) {
            char_int_vocabulary[str_data[i]] = next_vocabulary_int_value;
            int_char_vocabulary[next_vocabulary_int_value] = str_data[i];
            next_vocabulary_int_value++;
        }
        data[i] = char_int_vocabulary[str_data[i]];
    }

    int mask_token = next_vocabulary_int_value;
    char_int_vocabulary['_'] = mask_token;
    int_char_vocabulary[mask_token] = '_';

    // mask data
    std::vector<int> masked_data;
    mask(data, masked_data, 50, mask_token);

    int vocab_size = char_int_vocabulary.size(); 
    int rows = vocab_size + 1;
    int cols = vocab_size;
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols, 0));
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            matrix[r][c] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }

    std::vector<int> output_data;
    forward(data, masked_data, matrix, output_data, vocab_size, mask_token);

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        mask(data, masked_data, 50, mask_token);
        train(masked_data, data, matrix, data_size, vocab_size, mask_token, 0.01f);
        if (epoch % 100 == 0)
        {
            float l = loss(masked_data, data, matrix, data_size, vocab_size, mask_token);
            // std::cout << "Epoch " << epoch << " Loss: " << l << std::endl;
        }
    }

    mask(data, masked_data, 50, mask_token);
    forward(data, masked_data, matrix, output_data, vocab_size, mask_token);

    // std::cout << "Original: " << translate_data(data, int_char_vocabulary, data_size) << std::endl;
    // std::cout << "Maksed: " << translate_data(masked_data, int_char_vocabulary, data_size) << std::endl;
    // std::cout << "Predicted: " << translate_data(output_data, int_char_vocabulary, data_size) << std::endl;
    // std::cout << "Loss: " << loss(masked_data, data, matrix, data_size, vocab_size, mask_token) << std::endl;

    int d_model = 128;

    std::vector<std::vector<float>> embedding_layer(vocab_size, std::vector<float>(d_model, 0));
    for (int i = 0; i < vocab_size; i++)
        for (int j = 0; j < d_model; j++)
            embedding_layer[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;

    std::vector<std::vector<float>> W_Q(d_model, std::vector<float>(d_model, 0));
    std::vector<std::vector<float>> W_K(d_model, std::vector<float>(d_model, 0));
    std::vector<std::vector<float>> W_V(d_model, std::vector<float>(d_model, 0));

    for (int i = 0; i < d_model; i++)
    {
        for (int j = 0; j < d_model; j++)
        {
            W_Q[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
            W_K[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
            W_V[i][j] = ((float)rand() / RAND_MAX) * 0.02f - 0.01f;
        }
    }
    
    return 0;
}