module;

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

export module Kairo.Transformers;

import Kairo.Foundation.Math.Tensor;

export namespace kairo::transformers
{
    using kairo::foundation::math::Tensor;
    enum class Activation
    {
        ReLU,
        GELU,
        SiLU
    };

    struct TransformerConfig final
    {
        std::size_t vocabularySize = 0;
        std::size_t contextLength = 0;
        std::size_t modelWidth = 0;
        std::size_t headCount = 0;
        std::size_t layerCount = 0;
        std::size_t feedForwardWidth = 0;
        Activation activation = Activation::GELU;

        [[nodiscard]]
        bool Valid() const noexcept
        {
            return vocabularySize > 0 &&
                   contextLength > 0 &&
                   modelWidth > 0 &&
                   headCount > 0 &&
                   layerCount > 0 &&
                   feedForwardWidth >= modelWidth &&
                   modelWidth % headCount == 0;
        }

        [[nodiscard]]
        std::size_t HeadWidth() const
        {
            if (!Valid())
            {
                throw std::logic_error("TransformerConfig is invalid.");
            }
            return modelWidth / headCount;
        }
    };

    struct RopeConfig final
    {
        float theta = 10000.0f;
        float scale = 1.0f;
        bool enabled = true;
    };

    struct AttentionShape final
    {
        std::size_t batch = 0;
        std::size_t sequence = 0;
        std::size_t heads = 0;
        std::size_t headWidth = 0;

        [[nodiscard]]
        std::size_t QueryElementCount() const noexcept
        {
            return batch * sequence * heads * headWidth;
        }

        [[nodiscard]]
        std::size_t ScoreElementCount() const noexcept
        {
            return batch * heads * sequence * sequence;
        }
    };

    struct KVCacheDesc final
    {
        std::size_t batch = 0;
        std::size_t maxSequence = 0;
        std::size_t layers = 0;
        std::size_t heads = 0;
        std::size_t headWidth = 0;

        [[nodiscard]]
        std::size_t ElementCount() const noexcept
        {
            return 2 * batch * maxSequence * layers * heads * headWidth;
        }
    };

    struct TokenBatch final
    {
        std::size_t batchSize = 0;
        std::size_t sequenceLength = 0;
        std::vector<std::size_t> tokenIds;

        [[nodiscard]]
        bool ValidFor(const TransformerConfig& config) const noexcept
        {
            return config.Valid() &&
                   sequenceLength <= config.contextLength &&
                   tokenIds.size() == batchSize * sequenceLength;
        }
    };

    [[nodiscard]]
    inline float CausalMaskValue(std::size_t row, std::size_t column) noexcept
    {
        return column > row ? -1.0e30f : 0.0f;
    }

    [[nodiscard]]
    inline AttentionShape MakeAttentionShape(
        const TransformerConfig& config,
        std::size_t batch,
        std::size_t sequence)
    {
        if (!config.Valid() || sequence > config.contextLength)
        {
            throw std::logic_error("Invalid transformer attention shape.");
        }
        return {
            .batch = batch,
            .sequence = sequence,
            .heads = config.headCount,
            .headWidth = config.HeadWidth()
        };
    }

    [[nodiscard]]
    inline KVCacheDesc MakeKVCacheDesc(const TransformerConfig& config, std::size_t batch)
    {
        if (!config.Valid())
        {
            throw std::logic_error("Invalid transformer config.");
        }
        return {
            .batch = batch,
            .maxSequence = config.contextLength,
            .layers = config.layerCount,
            .heads = config.headCount,
            .headWidth = config.HeadWidth()
        };
    }

    [[nodiscard]]
    inline std::size_t ParameterCountEstimate(const TransformerConfig& config)
    {
        if (!config.Valid())
        {
            return 0;
        }

        const std::size_t embeddings = config.vocabularySize * config.modelWidth;
        const std::size_t attentionPerLayer = 4 * config.modelWidth * config.modelWidth;
        const std::size_t mlpPerLayer = 2 * config.modelWidth * config.feedForwardWidth;
        const std::size_t normsPerLayer = 4 * config.modelWidth;
        return embeddings + config.layerCount * (attentionPerLayer + mlpPerLayer + normsPerLayer);
    }

    /// Input: [rows, width] activations plus [width] scale and bias vectors.
    /// Output: row-wise LayerNorm with epsilon-stabilized variance.
    [[nodiscard]]
    inline Tensor<float> LayerNorm(
        const Tensor<float>& input,
        const Tensor<float>& scale,
        const Tensor<float>& bias,
        float epsilon = 1e-5f)
    {
        if (input.Rank() != 2 || scale.Rank() != 1 || bias.Rank() != 1
            || scale.Dim(0) != input.Dim(1) || bias.Dim(0) != input.Dim(1) || !(epsilon > 0.0f))
        {
            throw std::invalid_argument("LayerNorm expects [rows,width], [width], [width], and positive epsilon.");
        }
        Tensor<float> output(input.GetShape());
        const float inverseWidth = 1.0f / static_cast<float>(input.Dim(1));
        for (std::size_t row = 0; row < input.Dim(0); ++row)
        {
            float mean = 0.0f;
            for (std::size_t column = 0; column < input.Dim(1); ++column) mean += input(row, column);
            mean *= inverseWidth;
            float variance = 0.0f;
            for (std::size_t column = 0; column < input.Dim(1); ++column)
            {
                const float delta = input(row, column) - mean;
                variance += delta * delta;
            }
            const float inverseStdDev = 1.0f / std::sqrt(variance * inverseWidth + epsilon);
            for (std::size_t column = 0; column < input.Dim(1); ++column)
            {
                output(row, column) = (input(row, column) - mean) * inverseStdDev * scale[column] + bias[column];
            }
        }
        return output;
    }

    [[nodiscard]]
    inline Tensor<float> GELU(const Tensor<float>& input)
    {
        return input.Map([](float value)
        {
            constexpr float kSqrtTwoOverPi = 0.7978845608f;
            return 0.5f * value * (1.0f + std::tanh(kSqrtTwoOverPi * (value + 0.044715f * value * value * value)));
        });
    }

    /// Input: query/key/value tensors [sequence, headWidth].
    /// Output: causal scaled-dot-product attention [sequence, headWidth].
    /// No allocation or state from a previous call is retained; KV caching is
    /// intentionally a separate incremental-decoding layer.
    [[nodiscard]]
    inline Tensor<float> CausalScaledDotProductAttention(
        const Tensor<float>& query,
        const Tensor<float>& key,
        const Tensor<float>& value)
    {
        if (query.Rank() != 2 || key.Rank() != 2 || value.Rank() != 2
            || query.GetShape() != key.GetShape() || query.GetShape() != value.GetShape()
            || query.Dim(0) == 0 || query.Dim(1) == 0)
        {
            throw std::invalid_argument("CausalScaledDotProductAttention expects matching non-empty [sequence,headWidth] tensors.");
        }
        const std::size_t sequence = query.Dim(0);
        const std::size_t width = query.Dim(1);
        const float inverseScale = 1.0f / std::sqrt(static_cast<float>(width));
        Tensor<float> output({ sequence, width });
        std::vector<float> scores(sequence);
        for (std::size_t row = 0; row < sequence; ++row)
        {
            float maxScore = -std::numeric_limits<float>::infinity();
            for (std::size_t column = 0; column <= row; ++column)
            {
                float score = 0.0f;
                for (std::size_t channel = 0; channel < width; ++channel) score += query(row, channel) * key(column, channel);
                score *= inverseScale;
                scores[column] = score;
                maxScore = std::max(maxScore, score);
            }
            float sum = 0.0f;
            for (std::size_t column = 0; column <= row; ++column) { scores[column] = std::exp(scores[column] - maxScore); sum += scores[column]; }
            for (std::size_t channel = 0; channel < width; ++channel)
            {
                float attended = 0.0f;
                for (std::size_t column = 0; column <= row; ++column) attended += (scores[column] / sum) * value(column, channel);
                output(row, channel) = attended;
            }
        }
        return output;
    }
}
