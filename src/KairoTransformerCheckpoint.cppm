module;

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

export module Kairo.Transformers.Checkpoint;

export import Kairo.Transformers.Runtime;
export import Kairo.Transformers.Weights;

export namespace kairo::transformers
{
    /// Writes the complete decoder parameter set using stable version-1 names.
    /// Configuration is intentionally supplied separately so callers retain
    /// control over architecture and can validate loaded tensor shapes.
    inline void SaveDecoderCheckpoint(
        const std::filesystem::path& path,
        const DecoderModelWeights& weights)
    {
        std::vector<NamedTensor> tensors{
            { "token_embedding", weights.tokenEmbedding },
            { "final_norm.scale", weights.finalNormScale },
            { "final_norm.bias", weights.finalNormBias },
            { "lm_head.weight", weights.languageModelHead },
            { "lm_head.bias", weights.languageModelBias }
        };
        for (std::size_t layer = 0; layer < weights.blocks.size(); ++layer)
        {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            const DecoderBlockWeights& block = weights.blocks[layer];
            tensors.insert(tensors.end(), {
                { prefix + "attention_norm.scale", block.attentionNormScale },
                { prefix + "attention_norm.bias", block.attentionNormBias },
                { prefix + "query.weight", block.queryWeight },
                { prefix + "query.bias", block.queryBias },
                { prefix + "key.weight", block.keyWeight },
                { prefix + "key.bias", block.keyBias },
                { prefix + "value.weight", block.valueWeight },
                { prefix + "value.bias", block.valueBias },
                { prefix + "attention_output.weight", block.attentionOutputWeight },
                { prefix + "attention_output.bias", block.attentionOutputBias },
                { prefix + "feed_forward_norm.scale", block.feedForwardNormScale },
                { prefix + "feed_forward_norm.bias", block.feedForwardNormBias },
                { prefix + "feed_forward_first.weight", block.feedForwardFirstWeight },
                { prefix + "feed_forward_first.bias", block.feedForwardFirstBias },
                { prefix + "feed_forward_second.weight", block.feedForwardSecondWeight },
                { prefix + "feed_forward_second.bias", block.feedForwardSecondBias }
            });
        }
        BoundedTensorArchive::Save(path, tensors);
    }

    /// Loads one complete decoder under a per-tensor resident-byte bound.
    /// DecoderModel construction performs the final architecture shape checks.
    [[nodiscard]] inline DecoderModelWeights LoadDecoderCheckpoint(
        const std::filesystem::path& path,
        const TransformerConfig& config,
        std::size_t maximumTensorBytes)
    {
        BoundedTensorArchive archive(path);
        const auto load = [&](const std::string& name)
        {
            return archive.Load(name, maximumTensorBytes);
        };
        DecoderModelWeights weights{
            .tokenEmbedding = load("token_embedding"),
            .blocks = {},
            .finalNormScale = load("final_norm.scale"),
            .finalNormBias = load("final_norm.bias"),
            .languageModelHead = load("lm_head.weight"),
            .languageModelBias = load("lm_head.bias")
        };
        weights.blocks.reserve(config.layerCount);
        for (std::size_t layer = 0; layer < config.layerCount; ++layer)
        {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            weights.blocks.push_back({
                .attentionNormScale = load(prefix + "attention_norm.scale"),
                .attentionNormBias = load(prefix + "attention_norm.bias"),
                .queryWeight = load(prefix + "query.weight"),
                .queryBias = load(prefix + "query.bias"),
                .keyWeight = load(prefix + "key.weight"),
                .keyBias = load(prefix + "key.bias"),
                .valueWeight = load(prefix + "value.weight"),
                .valueBias = load(prefix + "value.bias"),
                .attentionOutputWeight = load(prefix + "attention_output.weight"),
                .attentionOutputBias = load(prefix + "attention_output.bias"),
                .feedForwardNormScale = load(prefix + "feed_forward_norm.scale"),
                .feedForwardNormBias = load(prefix + "feed_forward_norm.bias"),
                .feedForwardFirstWeight = load(prefix + "feed_forward_first.weight"),
                .feedForwardFirstBias = load(prefix + "feed_forward_first.bias"),
                .feedForwardSecondWeight = load(prefix + "feed_forward_second.weight"),
                .feedForwardSecondBias = load(prefix + "feed_forward_second.bias")
            });
        }
        return weights;
    }
}
