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

    /// Input: query/key/value [sequence, modelWidth] and a valid transformer
    /// config. Each contiguous model-width row is split into `headCount`
    /// logical heads, attended causally, then packed back in the same layout.
    [[nodiscard]]
    inline Tensor<float> MultiHeadCausalAttention(
        const TransformerConfig& config,
        const Tensor<float>& query,
        const Tensor<float>& key,
        const Tensor<float>& value)
    {
        if (!config.Valid() || query.Rank() != 2 || key.Rank() != 2 || value.Rank() != 2
            || query.Dim(0) == 0 || query.Dim(0) > config.contextLength
            || query.Dim(1) != config.modelWidth || key.Dim(0) != query.Dim(0)
            || key.Dim(1) != config.modelWidth || value.Dim(0) != query.Dim(0)
            || value.Dim(1) != config.modelWidth)
        {
            throw std::invalid_argument("MultiHeadCausalAttention expects matching [sequence, modelWidth] tensors and a valid config.");
        }
        const std::size_t sequence = query.Dim(0);
        const std::size_t headWidth = config.HeadWidth();
        Tensor<float> output({ sequence, config.modelWidth });
        for (std::size_t head = 0; head < config.headCount; ++head)
        {
            Tensor<float> headQuery({ sequence, headWidth });
            Tensor<float> headKey({ sequence, headWidth });
            Tensor<float> headValue({ sequence, headWidth });
            const std::size_t offset = head * headWidth;
            for (std::size_t row = 0; row < sequence; ++row)
            {
                for (std::size_t channel = 0; channel < headWidth; ++channel)
                {
                    headQuery(row, channel) = query(row, offset + channel);
                    headKey(row, channel) = key(row, offset + channel);
                    headValue(row, channel) = value(row, offset + channel);
                }
            }
            const Tensor<float> attended = CausalScaledDotProductAttention(headQuery, headKey, headValue);
            for (std::size_t row = 0; row < sequence; ++row)
            {
                for (std::size_t channel = 0; channel < headWidth; ++channel)
                {
                    output(row, offset + channel) = attended(row, channel);
                }
            }
        }
        return output;
    }

    /// Input: activations [sequence, modelWidth], dense weights
    /// [modelWidth, feedForwardWidth] and [feedForwardWidth, modelWidth], and
    /// their matching bias vectors. Output preserves [sequence, modelWidth].
    [[nodiscard]]
    inline Tensor<float> FeedForward(
        const TransformerConfig& config,
        const Tensor<float>& input,
        const Tensor<float>& firstWeight,
        const Tensor<float>& firstBias,
        const Tensor<float>& secondWeight,
        const Tensor<float>& secondBias)
    {
        if (!config.Valid() || input.Rank() != 2 || input.Dim(0) == 0 || input.Dim(1) != config.modelWidth
            || firstWeight.Rank() != 2 || firstWeight.Dim(0) != config.modelWidth || firstWeight.Dim(1) != config.feedForwardWidth
            || firstBias.Rank() != 1 || firstBias.Dim(0) != config.feedForwardWidth
            || secondWeight.Rank() != 2 || secondWeight.Dim(0) != config.feedForwardWidth || secondWeight.Dim(1) != config.modelWidth
            || secondBias.Rank() != 1 || secondBias.Dim(0) != config.modelWidth)
        {
            throw std::invalid_argument("FeedForward tensor shapes do not match TransformerConfig.");
        }
        Tensor<float> hidden = MatMul(input, firstWeight);
        for (std::size_t row = 0; row < hidden.Dim(0); ++row)
            for (std::size_t column = 0; column < hidden.Dim(1); ++column) hidden(row, column) += firstBias[column];
        switch (config.activation)
        {
        case Activation::ReLU: hidden = kairo::foundation::math::ReLU(hidden); break;
        case Activation::GELU: hidden = GELU(hidden); break;
        case Activation::SiLU: hidden = hidden.Map([](float value) { return value / (1.0f + std::exp(-value)); }); break;
        }
        Tensor<float> output = MatMul(hidden, secondWeight);
        for (std::size_t row = 0; row < output.Dim(0); ++row)
            for (std::size_t column = 0; column < output.Dim(1); ++column) output(row, column) += secondBias[column];
        return output;
    }

    [[nodiscard]]
    inline Tensor<float> AddResidual(const Tensor<float>& input, const Tensor<float>& branch)
    {
        if (input.Rank() != branch.Rank()) throw std::invalid_argument("Residual tensors must have equal shapes.");
        for (std::size_t axis = 0; axis < input.Rank(); ++axis)
            if (input.Dim(axis) != branch.Dim(axis)) throw std::invalid_argument("Residual tensors must have equal shapes.");
        return input + branch;
    }

    /// All learned parameters for one pre-norm decoder-only transformer block.
    /// Dense matrices are row-major Tensor shapes [inputWidth, outputWidth].
    struct DecoderBlockWeights final
    {
        Tensor<float> attentionNormScale;
        Tensor<float> attentionNormBias;
        Tensor<float> queryWeight;
        Tensor<float> queryBias;
        Tensor<float> keyWeight;
        Tensor<float> keyBias;
        Tensor<float> valueWeight;
        Tensor<float> valueBias;
        Tensor<float> attentionOutputWeight;
        Tensor<float> attentionOutputBias;
        Tensor<float> feedForwardNormScale;
        Tensor<float> feedForwardNormBias;
        Tensor<float> feedForwardFirstWeight;
        Tensor<float> feedForwardFirstBias;
        Tensor<float> feedForwardSecondWeight;
        Tensor<float> feedForwardSecondBias;
    };

    [[nodiscard]]
    inline Tensor<float> Dense(
        const Tensor<float>& input,
        const Tensor<float>& weight,
        const Tensor<float>& bias)
    {
        if (input.Rank() != 2 || weight.Rank() != 2 || bias.Rank() != 1
            || input.Dim(1) != weight.Dim(0) || weight.Dim(1) != bias.Dim(0))
        {
            throw std::invalid_argument("Dense expects [rows,input], [input,output], and [output] tensors.");
        }
        Tensor<float> output = MatMul(input, weight);
        for (std::size_t row = 0; row < output.Dim(0); ++row)
            for (std::size_t column = 0; column < output.Dim(1); ++column) output(row, column) += bias[column];
        return output;
    }

    /// Input: token activations [sequence, modelWidth] and all parameters for
    /// one decoder block. Output: same shape after pre-norm attention and MLP
    /// residual branches. This path is causal and stateless; use a future KV
    /// cache component for incremental decoding rather than reusing its output.
    [[nodiscard]]
    inline Tensor<float> DecoderBlock(
        const TransformerConfig& config,
        const Tensor<float>& input,
        const DecoderBlockWeights& weights,
        float epsilon = 1e-5f)
    {
        if (!config.Valid() || input.Rank() != 2 || input.Dim(0) == 0
            || input.Dim(0) > config.contextLength || input.Dim(1) != config.modelWidth)
        {
            throw std::invalid_argument("DecoderBlock expects a non-empty [sequence, modelWidth] input within context.");
        }

        const Tensor<float> attentionInput = LayerNorm(input, weights.attentionNormScale, weights.attentionNormBias, epsilon);
        const Tensor<float> query = Dense(attentionInput, weights.queryWeight, weights.queryBias);
        const Tensor<float> key = Dense(attentionInput, weights.keyWeight, weights.keyBias);
        const Tensor<float> value = Dense(attentionInput, weights.valueWeight, weights.valueBias);
        const Tensor<float> attended = MultiHeadCausalAttention(config, query, key, value);
        const Tensor<float> attentionBranch = Dense(attended, weights.attentionOutputWeight, weights.attentionOutputBias);
        const Tensor<float> afterAttention = AddResidual(input, attentionBranch);
        const Tensor<float> feedForwardInput = LayerNorm(afterAttention, weights.feedForwardNormScale, weights.feedForwardNormBias, epsilon);
        const Tensor<float> feedForwardBranch = FeedForward(
            config,
            feedForwardInput,
            weights.feedForwardFirstWeight,
            weights.feedForwardFirstBias,
            weights.feedForwardSecondWeight,
            weights.feedForwardSecondBias);
        return AddResidual(afterAttention, feedForwardBranch);
    }
}
