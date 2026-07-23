module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module Kairo.Transformers.Runtime;

export import Kairo.Transformers;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorTraining;

export namespace kairo::transformers
{
    using kairo::foundation::math::TrainingRandom;

    class Tokenizer
    {
    public:
        virtual ~Tokenizer() = default;
        [[nodiscard]] virtual std::size_t VocabularySize() const noexcept = 0;
        [[nodiscard]] virtual std::vector<std::size_t> Encode(
            std::string_view text) const = 0;
        [[nodiscard]] virtual std::string Decode(
            std::span<const std::size_t> tokens) const = 0;
    };

    class ByteTokenizer final : public Tokenizer
    {
    public:
        [[nodiscard]] std::size_t VocabularySize() const noexcept override { return 256; }

        [[nodiscard]] std::vector<std::size_t> Encode(
            std::string_view text) const override
        {
            std::vector<std::size_t> tokens;
            tokens.reserve(text.size());
            for (unsigned char byte : text) tokens.push_back(byte);
            return tokens;
        }

        [[nodiscard]] std::string Decode(
            std::span<const std::size_t> tokens) const override
        {
            std::string text;
            text.reserve(tokens.size());
            for (std::size_t token : tokens)
            {
                if (token >= VocabularySize())
                    throw std::out_of_range("ByteTokenizer token exceeds vocabulary.");
                text.push_back(static_cast<char>(token));
            }
            return text;
        }
    };

    [[nodiscard]] inline Tensor<float> TokenEmbedding(
        std::span<const std::size_t> tokens,
        const Tensor<float>& table)
    {
        if (tokens.empty() || table.Rank() != 2)
            throw std::invalid_argument("TokenEmbedding requires tokens and [vocabulary,width] table.");
        Tensor<float> output({ tokens.size(), table.Dim(1) }, 0.0f);
        for (std::size_t row = 0; row < tokens.size(); ++row)
        {
            if (tokens[row] >= table.Dim(0))
                throw std::out_of_range("Token id exceeds embedding vocabulary.");
            for (std::size_t column = 0; column < table.Dim(1); ++column)
                output(row, column) = table(tokens[row], column);
        }
        return output;
    }

    [[nodiscard]] inline Tensor<float> ApplyRoPE(
        const Tensor<float>& input,
        std::size_t headCount,
        std::size_t positionOffset = 0,
        RopeConfig config = {})
    {
        if (!config.enabled) return input.Contiguous();
        if (input.Rank() != 2 || headCount == 0 || input.Dim(1) % headCount != 0
            || (input.Dim(1) / headCount) % 2 != 0 || !(config.theta > 0.0f)
            || !(config.scale > 0.0f))
            throw std::invalid_argument("ApplyRoPE requires [sequence,width] with even head width.");
        Tensor<float> output = input.Contiguous();
        const std::size_t headWidth = input.Dim(1) / headCount;
        for (std::size_t row = 0; row < input.Dim(0); ++row)
            for (std::size_t head = 0; head < headCount; ++head)
                for (std::size_t pair = 0; pair < headWidth; pair += 2)
                {
                    const float exponent = static_cast<float>(pair)
                        / static_cast<float>(headWidth);
                    const float angle = static_cast<float>(positionOffset + row)
                        * config.scale / std::pow(config.theta, exponent);
                    const float cosine = std::cos(angle);
                    const float sine = std::sin(angle);
                    const std::size_t first = head * headWidth + pair;
                    const float x = input(row, first);
                    const float y = input(row, first + 1);
                    output(row, first) = x * cosine - y * sine;
                    output(row, first + 1) = x * sine + y * cosine;
                }
        return output;
    }

    [[nodiscard]] inline Tensor<float> RMSNorm(
        const Tensor<float>& input,
        const Tensor<float>& scale,
        float epsilon = 1e-5f)
    {
        if (input.Rank() != 2 || scale.Rank() != 1 || scale.Dim(0) != input.Dim(1))
            throw std::invalid_argument("RMSNorm expects [rows,width] and [width].");
        Tensor<float> output = kairo::foundation::math::RMSNormLastDim(input, epsilon);
        for (std::size_t row = 0; row < output.Dim(0); ++row)
            for (std::size_t column = 0; column < output.Dim(1); ++column)
                output(row, column) *= scale[column];
        return output;
    }

    /// Per-layer cache in [layer,position,head,channel] logical layout.
    class KVCache final
    {
    public:
        KVCache(
            const TransformerConfig& config,
            std::size_t batchSize = 1,
            std::size_t keyValueHeads = 0)
            : config_(config), batchSize_(batchSize),
              keyValueHeads_(keyValueHeads == 0 ? config.headCount : keyValueHeads),
              keys_({ config.layerCount, batchSize, config.contextLength,
                  keyValueHeads_, config.HeadWidth() }, 0.0f),
              values_(keys_.GetShape(), 0.0f),
              lengths_(config.layerCount, 0)
        {
            if (!config.Valid() || batchSize != 1 || keyValueHeads_ == 0
                || config.headCount % keyValueHeads_ != 0)
                throw std::invalid_argument(
                    "KVCache requires a valid config, batch one, and divisible KV heads.");
        }

        [[nodiscard]] std::size_t KeyValueHeads() const noexcept
        {
            return keyValueHeads_;
        }

        [[nodiscard]] std::size_t StorageElements() const noexcept
        {
            return keys_.Size() + values_.Size();
        }

        void Clear()
        {
            keys_.Fill(0.0f);
            values_.Fill(0.0f);
            std::fill(lengths_.begin(), lengths_.end(), 0);
        }

        [[nodiscard]] std::size_t Length(std::size_t layer) const
        {
            if (layer >= lengths_.size()) throw std::out_of_range("KV cache layer out of range.");
            return lengths_[layer];
        }

        void Append(
            std::size_t layer,
            std::size_t position,
            const Tensor<float>& key,
            const Tensor<float>& value)
        {
            if (layer >= config_.layerCount || position >= config_.contextLength
                || key.Rank() != 2 || value.Rank() != 2
                || key.Dim(0) != 1 || value.Dim(0) != 1
                || key.Dim(1) != keyValueHeads_ * config_.HeadWidth()
                || value.Dim(1) != keyValueHeads_ * config_.HeadWidth()
                || position != lengths_[layer])
                throw std::invalid_argument("KVCache Append must be contiguous and shape-compatible.");
            const std::size_t headWidth = config_.HeadWidth();
            for (std::size_t head = 0; head < keyValueHeads_; ++head)
                for (std::size_t channel = 0; channel < headWidth; ++channel)
                {
                    keys_.At({ layer, 0, position, head, channel }) =
                        key(0, head * headWidth + channel);
                    values_.At({ layer, 0, position, head, channel }) =
                        value(0, head * headWidth + channel);
                }
            ++lengths_[layer];
        }

        [[nodiscard]] Tensor<float> Attend(
            std::size_t layer, const Tensor<float>& query) const
        {
            if (layer >= config_.layerCount || query.Rank() != 2
                || query.Dim(0) != 1 || query.Dim(1) != config_.modelWidth
                || lengths_[layer] == 0)
                throw std::invalid_argument("KVCache Attend requires one compatible query.");
            const std::size_t length = lengths_[layer];
            const std::size_t headWidth = config_.HeadWidth();
            const float inverseScale = 1.0f / std::sqrt(static_cast<float>(headWidth));
            const std::size_t queryHeadsPerKV = config_.headCount / keyValueHeads_;
            Tensor<float> output({ 1, config_.modelWidth }, 0.0f);
            std::vector<float> scores(length);
            for (std::size_t head = 0; head < config_.headCount; ++head)
            {
                const std::size_t keyValueHead = head / queryHeadsPerKV;
                float maximum = -std::numeric_limits<float>::infinity();
                for (std::size_t position = 0; position < length; ++position)
                {
                    float score = 0.0f;
                    for (std::size_t channel = 0; channel < headWidth; ++channel)
                        score += query(0, head * headWidth + channel)
                            * keys_.At({
                                layer, 0, position, keyValueHead, channel });
                    scores[position] = score * inverseScale;
                    maximum = std::max(maximum, scores[position]);
                }
                float denominator = 0.0f;
                for (float& score : scores)
                {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (std::size_t channel = 0; channel < headWidth; ++channel)
                    for (std::size_t position = 0; position < length; ++position)
                        output(0, head * headWidth + channel) +=
                            scores[position] / denominator
                            * values_.At({
                                layer, 0, position, keyValueHead, channel });
            }
            return output;
        }

    private:
        TransformerConfig config_;
        std::size_t batchSize_;
        std::size_t keyValueHeads_;
        Tensor<float> keys_;
        Tensor<float> values_;
        std::vector<std::size_t> lengths_;
    };

    struct DecoderModelWeights final
    {
        Tensor<float> tokenEmbedding;
        std::vector<DecoderBlockWeights> blocks;
        Tensor<float> finalNormScale;
        Tensor<float> finalNormBias;
        Tensor<float> languageModelHead;
        Tensor<float> languageModelBias;
    };

    struct SamplingConfig final
    {
        float temperature = 1.0f;
        std::size_t topK = 0;
        float topP = 1.0f;
        std::uint64_t seed = 0x4B4149524F4C4DULL;
    };

    /// Returns the lowest token id among equal maximum logits. This explicit
    /// deterministic path does not consume random state.
    [[nodiscard]] inline std::size_t GreedyToken(const Tensor<float>& logits)
    {
        if (logits.Rank() != 1 || logits.Empty())
            throw std::invalid_argument("GreedyToken expects non-empty rank-one logits.");
        std::size_t selected = 0;
        for (std::size_t token = 1; token < logits.Size(); ++token)
            if (logits[token] > logits[selected]) selected = token;
        return selected;
    }

    [[nodiscard]] inline std::size_t SampleLogits(
        const Tensor<float>& logits,
        SamplingConfig config,
        TrainingRandom& random)
    {
        if (logits.Rank() != 1 || logits.Empty() || !(config.temperature > 0.0f)
            || !(config.topP > 0.0f && config.topP <= 1.0f))
            throw std::invalid_argument("SampleLogits received invalid logits or sampling config.");
        std::vector<std::pair<float, std::size_t>> candidates;
        candidates.reserve(logits.Size());
        for (std::size_t token = 0; token < logits.Size(); ++token)
            candidates.emplace_back(logits[token] / config.temperature, token);
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& lhs, const auto& rhs)
            {
                return lhs.first != rhs.first ? lhs.first > rhs.first : lhs.second < rhs.second;
            });
        if (config.topK > 0 && candidates.size() > config.topK)
            candidates.resize(config.topK);
        const float maximum = candidates.front().first;
        float total = 0.0f;
        for (auto& candidate : candidates)
        {
            candidate.first = std::exp(candidate.first - maximum);
            total += candidate.first;
        }
        if (config.topP < 1.0f)
        {
            float cumulative = 0.0f;
            std::size_t retained = 0;
            for (; retained < candidates.size(); ++retained)
            {
                cumulative += candidates[retained].first / total;
                if (cumulative >= config.topP) { ++retained; break; }
            }
            candidates.resize(std::max<std::size_t>(retained, 1));
            total = 0.0f;
            for (const auto& candidate : candidates) total += candidate.first;
        }
        float selected = random.Uniform() * total;
        for (const auto& candidate : candidates)
        {
            selected -= candidate.first;
            if (selected <= 0.0f) return candidate.second;
        }
        return candidates.back().second;
    }

    class DecoderModel final
    {
    public:
        DecoderModel(
            TransformerConfig config,
            DecoderModelWeights weights,
            RopeConfig rope = {})
            : config_(config), weights_(std::move(weights)), rope_(rope)
        {
            Validate();
        }

        [[nodiscard]] const TransformerConfig& Config() const noexcept { return config_; }

        [[nodiscard]] Tensor<float> Forward(std::span<const std::size_t> tokens) const
        {
            if (tokens.empty() || tokens.size() > config_.contextLength)
                throw std::invalid_argument("Decoder Forward token count exceeds context.");
            Tensor<float> hidden = TokenEmbedding(tokens, weights_.tokenEmbedding);
            for (std::size_t layer = 0; layer < weights_.blocks.size(); ++layer)
                hidden = ForwardBlock(layer, hidden, 0);
            hidden = LayerNorm(
                hidden, weights_.finalNormScale, weights_.finalNormBias);
            return Dense(hidden, weights_.languageModelHead, weights_.languageModelBias);
        }

        /// Decodes exactly one token at the requested contiguous cache position.
        [[nodiscard]] Tensor<float> Decode(
            std::size_t token, std::size_t position, KVCache& cache) const
        {
            if (position >= config_.contextLength)
                throw std::out_of_range("Decoder position exceeds context length.");
            const std::size_t tokenBuffer[] = { token };
            Tensor<float> hidden = TokenEmbedding(tokenBuffer, weights_.tokenEmbedding);
            for (std::size_t layer = 0; layer < weights_.blocks.size(); ++layer)
            {
                const DecoderBlockWeights& weights = weights_.blocks[layer];
                const Tensor<float> attentionInput = LayerNorm(
                    hidden, weights.attentionNormScale, weights.attentionNormBias);
                Tensor<float> query = ApplyRoPE(
                    Dense(attentionInput, weights.queryWeight, weights.queryBias),
                    config_.headCount, position, rope_);
                Tensor<float> key = ApplyRoPE(
                    Dense(attentionInput, weights.keyWeight, weights.keyBias),
                    config_.headCount, position, rope_);
                const Tensor<float> value =
                    Dense(attentionInput, weights.valueWeight, weights.valueBias);
                cache.Append(layer, position, key, value);
                const Tensor<float> attended = cache.Attend(layer, query);
                hidden = AddResidual(hidden, Dense(
                    attended, weights.attentionOutputWeight, weights.attentionOutputBias));
                const Tensor<float> feedForwardInput = LayerNorm(
                    hidden, weights.feedForwardNormScale, weights.feedForwardNormBias);
                hidden = AddResidual(hidden, FeedForward(
                    config_, feedForwardInput,
                    weights.feedForwardFirstWeight, weights.feedForwardFirstBias,
                    weights.feedForwardSecondWeight, weights.feedForwardSecondBias));
            }
            hidden = LayerNorm(hidden, weights_.finalNormScale, weights_.finalNormBias);
            return Dense(hidden, weights_.languageModelHead, weights_.languageModelBias);
        }

        [[nodiscard]] std::vector<std::size_t> Generate(
            std::vector<std::size_t> tokens,
            std::size_t newTokenCount,
            SamplingConfig sampling = {}) const
        {
            if (tokens.empty() || tokens.size() + newTokenCount > config_.contextLength)
                throw std::invalid_argument("Generate prompt/output exceeds context.");
            KVCache cache(config_);
            Tensor<float> logits;
            for (std::size_t position = 0; position < tokens.size(); ++position)
                logits = Decode(tokens[position], position, cache);
            TrainingRandom random(sampling.seed);
            for (std::size_t generated = 0; generated < newTokenCount; ++generated)
            {
                Tensor<float> last = logits.Slice(0, 0, 1).Reshape({ config_.vocabularySize });
                const std::size_t token = SampleLogits(last, sampling, random);
                tokens.push_back(token);
                if (generated + 1 < newTokenCount)
                    logits = Decode(token, tokens.size() - 1, cache);
            }
            return tokens;
        }

        [[nodiscard]] std::vector<std::size_t> GenerateGreedy(
            std::vector<std::size_t> tokens,
            std::size_t newTokenCount) const
        {
            if (tokens.empty() || tokens.size() + newTokenCount > config_.contextLength)
                throw std::invalid_argument("GenerateGreedy prompt/output exceeds context.");
            KVCache cache(config_);
            Tensor<float> logits;
            for (std::size_t position = 0; position < tokens.size(); ++position)
                logits = Decode(tokens[position], position, cache);
            for (std::size_t generated = 0; generated < newTokenCount; ++generated)
            {
                const Tensor<float> last =
                    logits.Slice(0, 0, 1).Reshape({ config_.vocabularySize });
                const std::size_t token = GreedyToken(last);
                tokens.push_back(token);
                if (generated + 1 < newTokenCount)
                    logits = Decode(token, tokens.size() - 1, cache);
            }
            return tokens;
        }

    private:
        TransformerConfig config_;
        DecoderModelWeights weights_;
        RopeConfig rope_;

        [[nodiscard]] Tensor<float> ForwardBlock(
            std::size_t layer, const Tensor<float>& input, std::size_t positionOffset) const
        {
            const DecoderBlockWeights& weights = weights_.blocks[layer];
            const Tensor<float> attentionInput = LayerNorm(
                input, weights.attentionNormScale, weights.attentionNormBias);
            const Tensor<float> query = ApplyRoPE(
                Dense(attentionInput, weights.queryWeight, weights.queryBias),
                config_.headCount, positionOffset, rope_);
            const Tensor<float> key = ApplyRoPE(
                Dense(attentionInput, weights.keyWeight, weights.keyBias),
                config_.headCount, positionOffset, rope_);
            const Tensor<float> value =
                Dense(attentionInput, weights.valueWeight, weights.valueBias);
            const Tensor<float> attended =
                MultiHeadCausalAttention(config_, query, key, value);
            Tensor<float> hidden = AddResidual(input, Dense(
                attended, weights.attentionOutputWeight, weights.attentionOutputBias));
            const Tensor<float> feedForwardInput = LayerNorm(
                hidden, weights.feedForwardNormScale, weights.feedForwardNormBias);
            return AddResidual(hidden, FeedForward(
                config_, feedForwardInput,
                weights.feedForwardFirstWeight, weights.feedForwardFirstBias,
                weights.feedForwardSecondWeight, weights.feedForwardSecondBias));
        }

        void Validate() const
        {
            if (!config_.Valid() || weights_.blocks.size() != config_.layerCount
                || weights_.tokenEmbedding.Rank() != 2
                || weights_.tokenEmbedding.Dim(0) != config_.vocabularySize
                || weights_.tokenEmbedding.Dim(1) != config_.modelWidth
                || weights_.finalNormScale.Rank() != 1
                || weights_.finalNormScale.Dim(0) != config_.modelWidth
                || weights_.finalNormBias.Rank() != 1
                || weights_.finalNormBias.Dim(0) != config_.modelWidth
                || weights_.languageModelHead.Rank() != 2
                || weights_.languageModelHead.Dim(0) != config_.modelWidth
                || weights_.languageModelHead.Dim(1) != config_.vocabularySize
                || weights_.languageModelBias.Rank() != 1
                || weights_.languageModelBias.Dim(0) != config_.vocabularySize)
                throw std::invalid_argument("DecoderModel weights do not match config.");
            // DecoderBlock performs complete per-layer shape validation on use.
        }
    };
}
