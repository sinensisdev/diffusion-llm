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
    std::ifstream ifstream(path, std::ios::in | std::ios::binary);
    const auto size = std::filesystem::file_size(path);
    std::string result(size, '\0');
    ifstream.read(result.data(), size);
    return result;
}

std::string translate_tokens(const std::vector<int>& data, std::unordered_map<int, char>& int_char_vocabulary, int data_size)
{
    std::string output(data_size, '\0');
    for (int i = 0; i < data_size; i++)
        output[i] = int_char_vocabulary[data[i]];
    return output;
}

int randomRange(int min, int max) {
    return min + rand() % (max - min + 1);
}

std::vector<std::vector<float>> embed(const std::vector<int>& data, const std::vector<std::vector<float>>& embedding_layer)
{
    std::vector<std::vector<float>> output;
    output.resize(data.size());
    for (int i = 0; i < data.size(); i++)
        output[i] = embedding_layer[data[i]];
    return output;
}

void add_positional_encoding(std::vector<std::vector<float>>& embedding, int d_model)
{
    for (int pos = 0; pos < embedding.size(); pos++)
        for (int i = 0; i < d_model / 2; i++)
        {
            float angle = pos / std::pow(10000.0f, (2.0f * i) / d_model);
            embedding[pos][2*i]   += std::sin(angle);
            embedding[pos][2*i+1] += std::cos(angle);
        }
}

std::vector<float> matvec(const std::vector<std::vector<float>>& matrix, const std::vector<float>& vec)
{
    std::vector<float> output;
    output.resize(matrix.size());
    for (int i = 0; i < matrix.size(); i++)
    {
        float sum = 0;
        for (int j = 0; j < matrix[i].size(); j++)
            sum += matrix[i][j] * vec[j];
        output[i] = sum;
    }
    return output;
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

float dot(std::vector<float> vec1, std::vector<float> vec2)
{
    float result = 0;
    for (int i = 0; i < vec1.size(); i++)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

float vec_dist(std::vector<float> vec1, std::vector<float> vec2)
{
    float distance = 0;
    for (int i = 0; i < vec1.size(); i++)
    {
        distance += abs(vec1[i] - vec2[i]);
    }
    return distance;
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

    int d_model = 128;

    std::vector<std::vector<float>> embedding_layer(next_vocabulary_int_value, std::vector<float>(d_model, 0));
    for (int i = 0; i < next_vocabulary_int_value; i++)
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

    auto X = embed(data, embedding_layer);
    add_positional_encoding(X, d_model);
    
    std::vector<std::vector<float>> Q(data_size, std::vector<float>(d_model, 0));
    std::vector<std::vector<float>> K(data_size, std::vector<float>(d_model, 0));
    std::vector<std::vector<float>> V(data_size, std::vector<float>(d_model, 0));
    for (int i = 0; i < data_size; i++)
    {
        Q[i] = matvec(W_Q, X[i]);
        K[i] = matvec(W_K, X[i]);
        V[i] = matvec(W_V, X[i]);
    }

    std::vector<std::vector<float>> relevance(data_size, std::vector<float>(data_size, 0));
    for (int i = 0; i < data_size; i++)
    {
        for (int j = 0; j < data_size; j++)
        {
            relevance[i][j] = dot(Q[i], K[j]);
        }
    }

    std::vector<std::vector<float>> relevance_normalized(data_size, std::vector<float>(data_size, 0));
    for (int i = 0; i < data_size; i++)
    {
        float exp_sum = 0.0f;
        for (int j = 0; j < data_size; j++)
        {
            relevance_normalized[i][j] = exp(relevance[i][j]);
            exp_sum += relevance_normalized[i][j];
        }
        for (int j = 0; j < data_size; j++)
            relevance_normalized[i][j] /= exp_sum;
    }

    std::vector<int> output(data_size, 0);

    for (int i = 0; i < data_size; i++)
    {
        if (masked_data[i] == mask_token)
        {
            std::vector<float> weighted_sum(d_model, 0);
            for (int j = 0; j < data_size; j++)
            {
                if (masked_data[j] != mask_token)
                {
                    for (int k = 0; k < d_model; k++)
                        weighted_sum[k] += V[j][k] * relevance_normalized[i][j];
                }
            }

            int closest_token = 0;
            float closest_distance = vec_dist(weighted_sum, embedding_layer[0]);
            for (int j = 1; j < next_vocabulary_int_value-1; j++)
            {
                float dist = vec_dist(weighted_sum, embedding_layer[j]);
                if (dist < closest_distance)
                {
                    closest_distance = dist;
                    closest_token = j;
                }
            }
            output[i] = closest_token;
        }
        else
        {
            output[i] = data[i];
        }
    }

    std::cout << "Original: " << translate_tokens(data, int_char_vocabulary, data_size) << std::endl;
    std::cout << "Maksed: " << translate_tokens(masked_data, int_char_vocabulary, data_size) << std::endl;
    std::cout << "Predicted: " << translate_tokens(output, int_char_vocabulary, data_size) << std::endl;

    return 0;
}