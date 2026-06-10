module;

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

export module Kairo.Transformers;

export namespace kairo::transformers
{
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
}
