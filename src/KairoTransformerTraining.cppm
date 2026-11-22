// TEACHING MODULE: TrainableDecoder creates TensorAutograd parameters
// and performs selected next-token training via categorical loss,
// optimizer updates and optional gradient checkpointing. This is a
// bounded training surface, not a guarantee of large-scale LLM training.
//
// LoRAProjection adds low-rank A*B updates to a frozen base projection,
// scaled by alpha/rank. Distinguish these trainable adapter values from
// full dense model weights. Gradient accumulation must manage scale
// and clearing across microbatches before an optimizer step.
//
// TensorTrainingCheckpoint saves parameters, complete optimizer state
// and RNG so resumed runs can be compared with uninterrupted training.
 // Verify actual parameter shapes/orders and deterministic test seeds.
module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

export module Kairo.Transformers.Training;

export import Kairo.Transformers;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorAutograd;
import Kairo.Foundation.Math.TensorTraining;

export namespace kairo::transformers
{
    using kairo::foundation::math::Add;
    using kairo::foundation::math::AutogradGELU;
    using kairo::foundation::math::AutogradCheckpoint;
    using kairo::foundation::math::AutogradGatherRows;
    using kairo::foundation::math::AutogradLayerNormLastDim;
    using kairo::foundation::math::AutogradMatMul;
    using kairo::foundation::math::AutogradMultiHeadCausalAttention;
    using kairo::foundation::math::SoftmaxCrossEntropyLoss;
    using kairo::foundation::math::TensorOptimizer;
    using kairo::foundation::math::TensorOptimizerConfig;
    using kairo::foundation::math::TensorTrainingCheckpoint;
    using kairo::foundation::math::TrainingRandom;
    using kairo::foundation::math::Variable;

    struct TransformerTrainingMetrics final
    {
        float loss = 0.0f;
        float accuracy = 0.0f;
        std::uint64_t optimizerStep = 0;
    };

    /// Low-rank trainable update for a frozen dense matrix:
    /// output = input*base + (alpha/rank)*(input*down)*up.
    // Trainable low-rank adapter with immutable base projection; its
    // rank/scale and input-output dimensions determine the update.
    class LoRAProjection final
    {
    public:
        LoRAProjection(
            std::size_t inputWidth,
            std::size_t outputWidth,
            std::size_t rank,
            float alpha,
            std::uint64_t seed = 0x4C4F5241ULL)
            : scale_(alpha / static_cast<float>(rank)),
              down_(Tensor<float>({ inputWidth, rank }, 0.0f), true),
              up_(Tensor<float>({ rank, outputWidth }, 0.0f), true)
        {
            if (inputWidth == 0 || outputWidth == 0 || rank == 0 || !(alpha > 0.0f))
                throw std::invalid_argument("LoRAProjection dimensions and alpha must be positive.");
            TrainingRandom random(seed);
            Tensor<float> initialized({ inputWidth, rank }, 0.0f);
            const float bound = 1.0f / std::sqrt(static_cast<float>(inputWidth));
            for (std::size_t index = 0; index < initialized.Size(); ++index)
                initialized[index] = (random.Uniform() * 2.0f - 1.0f) * bound;
            down_.LoadValue(std::move(initialized));
        }

        [[nodiscard]] Variable Forward(
            const Variable& input, const Tensor<float>& frozenBase) const
        {
            if (frozenBase.Rank() != 2
                || frozenBase.Dim(0) != down_.Value().Dim(0)
                || frozenBase.Dim(1) != up_.Value().Dim(1))
                throw std::invalid_argument("LoRA frozen base shape mismatch.");
            const Variable base = AutogradMatMul(input, Variable(frozenBase));
            const Variable lowRank =
                AutogradMatMul(AutogradMatMul(input, down_), up_);
            const Variable scale(Tensor<float>(lowRank.Value().GetShape(), scale_));
            return Add(base, kairo::foundation::math::Multiply(lowRank, scale));
        }

        [[nodiscard]] std::vector<Variable*> Parameters()
        {
            return { &down_, &up_ };
        }

    private:
        float scale_;
        Variable down_;
        Variable up_;
    };

    namespace training_detail
    {
        [[nodiscard]] inline Tensor<float> RandomTensor(
            std::vector<std::size_t> shape,
            TrainingRandom& random,
            float scale)
        {
            Tensor<float> tensor(std::move(shape), 0.0f);
            for (std::size_t index = 0; index < tensor.Size(); ++index)
                tensor[index] = (random.Uniform() * 2.0f - 1.0f) * scale;
            return tensor;
        }

        struct Layer final
        {
            Variable query;
            Variable key;
            Variable value;
            Variable attentionOutput;
            Variable feedForwardFirst;
            Variable feedForwardSecond;

            Layer(const TransformerConfig& config, TrainingRandom& random, float scale)
                : query(RandomTensor(
                    { config.modelWidth, config.modelWidth }, random, scale), true),
                  key(RandomTensor(
                    { config.modelWidth, config.modelWidth }, random, scale), true),
                  value(RandomTensor(
                    { config.modelWidth, config.modelWidth }, random, scale), true),
                  attentionOutput(RandomTensor(
                    { config.modelWidth, config.modelWidth }, random, scale), true),
                  feedForwardFirst(RandomTensor(
                    { config.modelWidth, config.feedForwardWidth }, random, scale), true),
                  feedForwardSecond(RandomTensor(
                    { config.feedForwardWidth, config.modelWidth }, random, scale), true)
            {
            }
        };
    }

    /// Minimal decoder-only training model built entirely from KairoMath
    /// autograd Variables. It intentionally keeps biases and affine norm terms
    /// out of the core parameter set so corpus-overfit tests exercise attention,
    /// embeddings, MLP, residual, normalization, and language-head gradients
    /// directly.
    // Owns trainable layer variables, random state and optimizer-facing
    // parameter order. Inference DecoderModel owns a distinct weight API.
    class TrainableDecoder final
    {
    public:
        // Own trainable embedding/head matrices, layer variables and the seeded
        // TrainingRandom stream. Training has a separate API from inference
        // DecoderModel, but both use TransformerConfig shapes and KairoMath.
        explicit TrainableDecoder(
            TransformerConfig config,
            std::uint64_t seed = 0x545241494E4552ULL)
            : config_(config),
              random_(seed),
              embedding_(training_detail::RandomTensor(
                  { config.vocabularySize, config.modelWidth }, random_, 0.2f), true),
              languageHead_(training_detail::RandomTensor(
                  { config.modelWidth, config.vocabularySize }, random_, 0.2f), true)
        {
            if (!config_.Valid())
                throw std::invalid_argument("TrainableDecoder requires a valid config.");
            layers_.reserve(config_.layerCount);
            const float scale = 1.0f / std::sqrt(static_cast<float>(config_.modelWidth));
            for (std::size_t layer = 0; layer < config_.layerCount; ++layer)
                layers_.emplace_back(config_, random_, scale);
        }

        [[nodiscard]] const TransformerConfig& Config() const noexcept { return config_; }
        [[nodiscard]] TrainingRandom& Random() noexcept { return random_; }
        // Recompute layer activations during backward to trade extra compute
        // for lower retained autograd memory. Not a disk save/load checkpoint.
        void SetGradientCheckpointing(bool enabled) noexcept
        {
            gradientCheckpointing_ = enabled;
        }
        [[nodiscard]] bool GradientCheckpointing() const noexcept
        {
            return gradientCheckpointing_;
        }

        // Return borrowed pointers in stable order: embeddings, each layer's
        // projections and MLP, then head. Optimizer/checkpoints rely on this
        // order and pointers die when the TrainableDecoder is destroyed.
        [[nodiscard]] std::vector<Variable*> Parameters()
        {
            std::vector<Variable*> result{ &embedding_ };
            for (auto& layer : layers_)
            {
                result.push_back(&layer.query);
                result.push_back(&layer.key);
                result.push_back(&layer.value);
                result.push_back(&layer.attentionOutput);
                result.push_back(&layer.feedForwardFirst);
                result.push_back(&layer.feedForwardSecond);
            }
            result.push_back(&languageHead_);
            return result;
        }

        // Embed token IDs and evaluate every decoder autograd layer. Optional
        // recomputation checkpoints preserve gradient connectivity; final
        // projection outputs [sequence,vocabulary] logits for loss/training.
        [[nodiscard]] Variable Forward(std::span<const std::size_t> inputTokens) const
        {
            if (inputTokens.empty() || inputTokens.size() > config_.contextLength)
                throw std::invalid_argument("Transformer training input exceeds context.");
            Variable hidden = AutogradGatherRows(embedding_, inputTokens);
            for (std::size_t layer = 0; layer < layers_.size(); ++layer)
            {
                if (gradientCheckpointing_)
                {
                    const Variable checkpointInput = hidden;
                    hidden = AutogradCheckpoint([this, checkpointInput, layer]
                    {
                        return ForwardLayer(checkpointInput, layers_[layer]);
                    });
                }
                else
                    hidden = ForwardLayer(hidden, layers_[layer]);
            }
            return AutogradMatMul(
                AutogradLayerNormLastDim(hidden, 1e-5f), languageHead_);
        }

        // Convert target IDs into one-hot labels, requiring matching input/
        // target sequence lengths and valid vocabulary IDs, then compute a
        // softmax cross-entropy objective from forward token logits.
        [[nodiscard]] Variable Loss(
            std::span<const std::size_t> inputTokens,
            std::span<const std::size_t> targetTokens) const
        {
            if (inputTokens.size() != targetTokens.size())
                throw std::invalid_argument("Transformer input/target lengths must match.");
            Tensor<float> labels(
                { targetTokens.size(), config_.vocabularySize }, 0.0f);
            for (std::size_t row = 0; row < targetTokens.size(); ++row)
            {
                if (targetTokens[row] >= config_.vocabularySize)
                    throw std::out_of_range("Transformer target token exceeds vocabulary.");
                labels(row, targetTokens[row]) = 1.0f;
            }
            return SoftmaxCrossEntropyLoss(Forward(inputTokens), labels);
        }

        /// Accumulates gradients over all supplied sequences and performs one
        /// optimizer step divided by sequence count.
        // Clear parameter grads once, Backward each sequence's loss, and
        // TensorOptimizer::Step after all sequences with its gradient-count
        // divisor. Return mean training loss, in-sample token accuracy and
        // completed optimizer-step count, not held-out generalization.
        [[nodiscard]] TransformerTrainingMetrics TrainAccumulated(
            const std::vector<std::vector<std::size_t>>& inputSequences,
            const std::vector<std::vector<std::size_t>>& targetSequences,
            TensorOptimizer& optimizer)
        {
            if (inputSequences.empty() || inputSequences.size() != targetSequences.size())
                throw std::invalid_argument("Training accumulation requires paired sequences.");
            std::vector<Variable*> parameters = Parameters();
            for (Variable* parameter : parameters) parameter->ZeroGradient();
            float loss = 0.0f;
            for (std::size_t sequence = 0; sequence < inputSequences.size(); ++sequence)
            {
                const Variable sequenceLoss =
                    Loss(inputSequences[sequence], targetSequences[sequence]);
                loss += sequenceLoss.Value()[0];
                sequenceLoss.Backward();
            }
            optimizer.Step(parameters, static_cast<float>(inputSequences.size()));
            const float accuracy = Accuracy(inputSequences, targetSequences);
            return {
                loss / static_cast<float>(inputSequences.size()),
                accuracy,
                optimizer.CompletedSteps()
            };
        }

        // Re-evaluate current model on supplied sequences and count argmax
        // token matches; this is in-sample training accuracy unless callers
        // explicitly pass separate held-out sequences.
        [[nodiscard]] float Accuracy(
            const std::vector<std::vector<std::size_t>>& inputSequences,
            const std::vector<std::vector<std::size_t>>& targetSequences) const
        {
            std::size_t correct = 0;
            std::size_t total = 0;
            for (std::size_t sequence = 0; sequence < inputSequences.size(); ++sequence)
            {
                const Variable forward = Forward(inputSequences[sequence]);
                const Tensor<float>& logits = forward.Value();
                for (std::size_t row = 0; row < logits.Dim(0); ++row)
                {
                    std::size_t predicted = 0;
                    for (std::size_t token = 1; token < logits.Dim(1); ++token)
                        if (logits(row, token) > logits(row, predicted)) predicted = token;
                    correct += predicted == targetSequences[sequence][row] ? 1 : 0;
                    ++total;
                }
            }
            return total == 0 ? 0.0f
                : static_cast<float>(correct) / static_cast<float>(total);
        }

        // Persist parameters AND TensorOptimizer/TrainingRandom state through
        // KairoMath's TensorTrainingCheckpoint. Distinguish this from decoder
        // inference weight archives and MLLibrary's KMLCP01 weights-only file.
        void Save(
            const std::filesystem::path& path,
            const TensorOptimizer& optimizer) 
        {
            std::vector<Variable*> parameters = Parameters();
            TensorTrainingCheckpoint::Save(path, parameters, optimizer, random_);
        }

        // Restore compatible parameter list, optimizer and RNG. Verify
        // matching architecture, tensor order and next-step parity before
        // claiming reproducible resumption on other compilers/devices.
        void Load(
            const std::filesystem::path& path,
            TensorOptimizer& optimizer)
        {
            std::vector<Variable*> parameters = Parameters();
            TensorTrainingCheckpoint::Load(path, parameters, optimizer, random_);
        }

    private:
        TransformerConfig config_;
        TrainingRandom random_;
        Variable embedding_;
        std::vector<training_detail::Layer> layers_;
        Variable languageHead_;
        bool gradientCheckpointing_ = false;

        [[nodiscard]] Variable ForwardLayer(
            const Variable& input,
            const training_detail::Layer& layer) const
        {
            const Variable attentionInput =
                AutogradLayerNormLastDim(input, 1e-5f);
            const Variable query = AutogradMatMul(attentionInput, layer.query);
            const Variable key = AutogradMatMul(attentionInput, layer.key);
            const Variable value = AutogradMatMul(attentionInput, layer.value);
            const Variable attended = AutogradMultiHeadCausalAttention(
                query, key, value, config_.headCount);
            Variable hidden =
                Add(input, AutogradMatMul(attended, layer.attentionOutput));
            const Variable feedForwardInput =
                AutogradLayerNormLastDim(hidden, 1e-5f);
            const Variable feedForward = AutogradGELU(
                AutogradMatMul(feedForwardInput, layer.feedForwardFirst));
            return Add(
                hidden, AutogradMatMul(feedForward, layer.feedForwardSecond));
        }
    };

    [[nodiscard]] inline TensorOptimizerConfig DefaultTransformerAdamW(
        float learningRate = 3e-3f)
    {
        TensorOptimizerConfig config;
        config.kind = kairo::foundation::math::TensorOptimizerKind::AdamW;
        config.schedule.baseRate = learningRate;
        config.weightDecay = 1e-4f;
        config.maximumGradientNorm = 1.0f;
        return config;
    }
}
