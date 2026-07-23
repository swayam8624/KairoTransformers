#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

import Kairo.Transformers;
import Kairo.Transformers.Runtime;
import Kairo.Transformers.Weights;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorTraining;

namespace
{
    using kairo::foundation::math::Tensor;
    using kairo::foundation::math::TrainingRandom;
    using namespace kairo::transformers;

    Tensor<float> RandomTensor(
        std::vector<std::size_t> shape, TrainingRandom& random, float scale)
    {
        Tensor<float> tensor(std::move(shape), 0.0f);
        for (std::size_t index = 0; index < tensor.Size(); ++index)
            tensor[index] = (random.Uniform() * 2.0f - 1.0f) * scale;
        return tensor;
    }

    DecoderBlockWeights MakeBlock(
        const TransformerConfig& config, TrainingRandom& random)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(config.modelWidth));
        return {
            .attentionNormScale = Tensor<float>({ config.modelWidth }, 1.0f),
            .attentionNormBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .queryWeight = RandomTensor(
                { config.modelWidth, config.modelWidth }, random, scale),
            .queryBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .keyWeight = RandomTensor(
                { config.modelWidth, config.modelWidth }, random, scale),
            .keyBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .valueWeight = RandomTensor(
                { config.modelWidth, config.modelWidth }, random, scale),
            .valueBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .attentionOutputWeight = RandomTensor(
                { config.modelWidth, config.modelWidth }, random, scale),
            .attentionOutputBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .feedForwardNormScale = Tensor<float>({ config.modelWidth }, 1.0f),
            .feedForwardNormBias = Tensor<float>({ config.modelWidth }, 0.0f),
            .feedForwardFirstWeight = RandomTensor(
                { config.modelWidth, config.feedForwardWidth }, random, scale),
            .feedForwardFirstBias = Tensor<float>({ config.feedForwardWidth }, 0.0f),
            .feedForwardSecondWeight = RandomTensor(
                { config.feedForwardWidth, config.modelWidth }, random, scale),
            .feedForwardSecondBias = Tensor<float>({ config.modelWidth }, 0.0f)
        };
    }

    std::uint64_t PeakRSSBytes()
    {
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
        return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
        return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024u;
#endif
    }
}

int main(int argc, char** argv)
{
    using namespace kairo::transformers;
    const TransformerConfig config{
        .vocabularySize = 64,
        .contextLength = 32,
        .modelWidth = 16,
        .headCount = 4,
        .layerCount = 2,
        .feedForwardWidth = 32,
        .activation = Activation::GELU
    };
    TrainingRandom random(20260723);
    DecoderModelWeights weights{
        .tokenEmbedding = RandomTensor(
            { config.vocabularySize, config.modelWidth }, random, 0.2f),
        .blocks = {},
        .finalNormScale = Tensor<float>({ config.modelWidth }, 1.0f),
        .finalNormBias = Tensor<float>({ config.modelWidth }, 0.0f),
        .languageModelHead = RandomTensor(
            { config.modelWidth, config.vocabularySize }, random, 0.2f),
        .languageModelBias = Tensor<float>({ config.vocabularySize }, 0.0f)
    };
    for (std::size_t layer = 0; layer < config.layerCount; ++layer)
        weights.blocks.push_back(MakeBlock(config, random));
    const DecoderModel model(config, weights);
    std::vector<std::size_t> tokens(24);
    for (std::size_t index = 0; index < tokens.size(); ++index)
        tokens[index] = (index * 7 + 3) % config.vocabularySize;

    const Tensor<float> full = model.Forward(tokens);
    KVCache cache(config);
    float maximumEquivalenceError = 0.0f;
    const auto begin = std::chrono::steady_clock::now();
    for (std::size_t position = 0; position < tokens.size(); ++position)
    {
        const Tensor<float> cached = model.Decode(tokens[position], position, cache);
        for (std::size_t token = 0; token < config.vocabularySize; ++token)
            maximumEquivalenceError = std::max(
                maximumEquivalenceError,
                std::abs(cached(0, token) - full(position, token)));
    }
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    const double tokensPerSecond = static_cast<double>(tokens.size()) / seconds;

    const Tensor<float>& dense = weights.blocks[0].feedForwardFirstWeight;
    const Tensor<float> activation =
        RandomTensor({ 8, config.modelWidth }, random, 1.0f);
    const Tensor<float> reference =
        kairo::foundation::math::MatMul(activation, dense);
    const Tensor<float> int8 =
        QuantizedMatMul(activation, QuantizeInt8(dense));
    const Tensor<float> int4 =
        QuantizedMatMul(activation, QuantizeInt4(dense));
    float int8Error = 0.0f;
    float int4Error = 0.0f;
    for (std::size_t index = 0; index < reference.Size(); ++index)
    {
        int8Error = std::max(int8Error, std::abs(reference[index] - int8[index]));
        int4Error = std::max(int4Error, std::abs(reference[index] - int4[index]));
    }

    std::ostringstream json;
    json << std::setprecision(10)
         << "{\n"
         << "  \"schema\": \"kairo.transformer.benchmark.v1\",\n"
         << "  \"seed\": 20260723,\n"
         << "  \"tokens\": " << tokens.size() << ",\n"
         << "  \"elapsed_seconds\": " << seconds << ",\n"
         << "  \"tokens_per_second\": " << tokensPerSecond << ",\n"
         << "  \"peak_rss_bytes\": " << PeakRSSBytes() << ",\n"
         << "  \"cached_full_max_abs_error\": " << maximumEquivalenceError << ",\n"
         << "  \"int8_max_abs_error\": " << int8Error << ",\n"
         << "  \"int4_max_abs_error\": " << int4Error << "\n"
         << "}\n";
    std::cout << json.str();
    if (argc > 1)
    {
        std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
        if (!output) return 2;
        output << json.str();
        if (!output) return 3;
    }
    return maximumEquivalenceError <= 1e-5f
        && int8Error < 0.05f && int4Error < 0.8f
        && std::isfinite(tokensPerSecond) && tokensPerSecond > 0.0 ? 0 : 1;
}
