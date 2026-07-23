#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

import Kairo.Transformers;
import Kairo.Transformers.Runtime;
import Kairo.Transformers.Weights;
import Kairo.Transformers.Checkpoint;
import Kairo.Foundation.Math.Tensor;

int main()
{
    kairo::transformers::TransformerConfig config{};
    config.vocabularySize = 32000;
    config.contextLength = 128;
    config.modelWidth = 256;
    config.headCount = 8;
    config.layerCount = 4;
    config.feedForwardWidth = 1024;
    assert(config.Valid());
    assert(config.HeadWidth() == 32);
    assert(kairo::transformers::CausalMaskValue(0, 1) < 0.0f);
    assert(kairo::transformers::CausalMaskValue(3, 0) == 0.0f);
    auto attention = kairo::transformers::MakeAttentionShape(config, 2, 16);
    assert(attention.ScoreElementCount() == 2 * 8 * 16 * 16);
    auto cache = kairo::transformers::MakeKVCacheDesc(config, 2);
    assert(cache.ElementCount() == 2 * 2 * 128 * 4 * 8 * 32);

    using kairo::foundation::math::Tensor;
    const Tensor<float> activations({ 2, 2 }, { 1.0f, 3.0f, 2.0f, 6.0f });
    const Tensor<float> scale({ 2 }, { 1.0f, 1.0f });
    const Tensor<float> bias({ 2 }, { 0.0f, 0.0f });
    const auto normalized = kairo::transformers::LayerNorm(activations, scale, bias);
    assert(normalized(0, 0) < 0.0f && normalized(0, 1) > 0.0f);
    const auto gelu = kairo::transformers::GELU(activations);
    assert(gelu(1, 1) > gelu(1, 0));
    const auto positions = kairo::transformers::SinusoidalEncoding(2, 3);
    assert(positions(0, 0) == 0.0f && positions(0, 1) == 1.0f);
    assert(std::abs(positions(1, 0) - std::sin(1.0f)) < 1e-6f);
    const Tensor<float> packedWeight({ 2, 6 }, {
        1, 0, 2, 0, 3, 0,
        0, 1, 0, 2, 0, 3
    });
    const auto projected = kairo::transformers::FusedQKVProjection(
        activations, packedWeight, Tensor<float>({ 6 }, 0.0f));
    assert(projected.query(0, 0) == 1.0f && projected.query(0, 1) == 3.0f);
    assert(projected.key(0, 0) == 2.0f && projected.key(0, 1) == 6.0f);
    assert(projected.value(0, 0) == 3.0f && projected.value(0, 1) == 9.0f);

    const Tensor<float> query({ 2, 2 }, { 1.0f, 0.0f, 0.0f, 1.0f });
    const Tensor<float> key({ 2, 2 }, { 1.0f, 0.0f, 0.0f, 1.0f });
    const Tensor<float> value({ 2, 2 }, { 3.0f, 4.0f, 7.0f, 8.0f });
    const auto attended = kairo::transformers::CausalScaledDotProductAttention(query, key, value);
    assert(attended(0, 0) == 3.0f && attended(0, 1) == 4.0f);
    assert(attended(1, 0) > 3.0f && attended(1, 0) < 7.0f);

    kairo::transformers::TransformerConfig twoHead{};
    twoHead.vocabularySize = 8;
    twoHead.contextLength = 2;
    twoHead.modelWidth = 4;
    twoHead.headCount = 2;
    twoHead.layerCount = 1;
    twoHead.feedForwardWidth = 8;
    const Tensor<float> multiQuery({ 2, 4 }, { 1, 0, 1, 0, 0, 1, 0, 1 });
    const Tensor<float> multiValue({ 2, 4 }, { 3, 4, 30, 40, 7, 8, 70, 80 });
    const auto multi = kairo::transformers::MultiHeadCausalAttention(twoHead, multiQuery, multiQuery, multiValue);
    assert(multi(0, 0) == 3.0f && multi(0, 2) == 30.0f);
    assert(multi(1, 0) > 3.0f && multi(1, 0) < 7.0f);
    assert(multi(1, 2) > 30.0f && multi(1, 2) < 70.0f);
    const auto groupedEquivalent = kairo::transformers::GroupedQueryCausalAttention(
        multiQuery, multiQuery, multiValue, 2, 2);
    for (std::size_t index = 0; index < multi.Size(); ++index)
        assert(std::abs(groupedEquivalent[index] - multi[index]) < 1e-6f);
    const Tensor<float> sharedKey({ 2, 2 }, { 1, 0, 0, 1 });
    const Tensor<float> sharedValue({ 2, 2 }, { 3, 4, 7, 8 });
    const auto multiQueryOutput = kairo::transformers::GroupedQueryCausalAttention(
        multiQuery, sharedKey, sharedValue, 2, 1);
    assert(multiQueryOutput.Dim(1) == 4);
    assert(multiQueryOutput(0, 0) == 3.0f && multiQueryOutput(0, 2) == 3.0f);
    kairo::transformers::KVCache groupedCache(twoHead, 1, 1);
    kairo::transformers::KVCache fullHeadCache(twoHead);
    assert(groupedCache.StorageElements() * 2 == fullHeadCache.StorageElements());
    for (std::size_t position = 0; position < 2; ++position)
    {
        const Tensor<float> oneKey =
            sharedKey.Slice(0, position, 1);
        const Tensor<float> oneValue =
            sharedValue.Slice(0, position, 1);
        groupedCache.Append(0, position, oneKey, oneValue);
        const Tensor<float> oneQuery =
            multiQuery.Slice(0, position, 1);
        const Tensor<float> cachedGrouped = groupedCache.Attend(0, oneQuery);
        for (std::size_t channel = 0; channel < 4; ++channel)
            assert(std::abs(cachedGrouped(0, channel)
                - multiQueryOutput(position, channel)) < 1e-6f);
    }

    kairo::transformers::ByteTokenizer tokenizer;
    const std::string tokenizerText = "Kairo \xF0\x9F\xA7\xA0";
    const auto encodedText = tokenizer.Encode(tokenizerText);
    assert(tokenizer.VocabularySize() == 256);
    assert(tokenizer.Decode(encodedText) == tokenizerText);

    twoHead.activation = kairo::transformers::Activation::ReLU;
    const Tensor<float> firstWeight({ 4, 8 }, 1.0f);
    const Tensor<float> firstBias({ 8 }, 0.0f);
    const Tensor<float> secondWeight({ 8, 4 }, 0.25f);
    const Tensor<float> secondBias({ 4 }, 0.0f);
    const auto feedForward = kairo::transformers::FeedForward(twoHead, multiQuery, firstWeight, firstBias, secondWeight, secondBias);
    assert(feedForward.Dim(0) == 2 && feedForward.Dim(1) == 4 && feedForward(0, 0) > 0.0f);
    const auto residual = kairo::transformers::AddResidual(multiQuery, feedForward);
    assert(residual(0, 0) > multiQuery(0, 0));

    kairo::transformers::DecoderBlockWeights blockWeights{
        .attentionNormScale = Tensor<float>({ 4 }, 1.0f),
        .attentionNormBias = Tensor<float>({ 4 }, 0.0f),
        .queryWeight = Tensor<float>({ 4, 4 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }),
        .queryBias = Tensor<float>({ 4 }, 0.0f),
        .keyWeight = Tensor<float>({ 4, 4 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }),
        .keyBias = Tensor<float>({ 4 }, 0.0f),
        .valueWeight = Tensor<float>({ 4, 4 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 }),
        .valueBias = Tensor<float>({ 4 }, 0.0f),
        .attentionOutputWeight = Tensor<float>({ 4, 4 }, { 0.1f, 0, 0, 0, 0, 0.1f, 0, 0, 0, 0, 0.1f, 0, 0, 0, 0, 0.1f }),
        .attentionOutputBias = Tensor<float>({ 4 }, 0.0f),
        .feedForwardNormScale = Tensor<float>({ 4 }, 1.0f),
        .feedForwardNormBias = Tensor<float>({ 4 }, 0.0f),
        .feedForwardFirstWeight = Tensor<float>({ 4, 8 }, 1.0f),
        .feedForwardFirstBias = Tensor<float>({ 8 }, 0.0f),
        .feedForwardSecondWeight = Tensor<float>({ 8, 4 }, 0.1f),
        .feedForwardSecondBias = Tensor<float>({ 4 }, 0.0f)
    };
    const auto decoder = kairo::transformers::DecoderBlock(twoHead, multiQuery, blockWeights);
    assert(decoder.Dim(0) == 2 && decoder.Dim(1) == 4);
    assert(decoder(0, 0) != multiQuery(0, 0));

    twoHead.vocabularySize = 6;
    twoHead.contextLength = 8;
    kairo::transformers::DecoderModelWeights modelWeights{
        .tokenEmbedding = Tensor<float>({ 6, 4 }, {
            0.1f, 0.2f, 0.3f, 0.4f,
            0.5f, -0.2f, 0.1f, 0.3f,
            -0.1f, 0.4f, 0.2f, -0.3f,
            0.7f, 0.1f, -0.2f, 0.2f,
            -0.4f, 0.3f, 0.6f, 0.1f,
            0.2f, -0.5f, 0.4f, 0.8f
        }),
        .blocks = { blockWeights },
        .finalNormScale = Tensor<float>({ 4 }, 1.0f),
        .finalNormBias = Tensor<float>({ 4 }, 0.0f),
        .languageModelHead = Tensor<float>({ 4, 6 }, {
            0.2f, -0.1f, 0.3f, 0.1f, -0.2f, 0.4f,
            -0.3f, 0.2f, 0.1f, 0.5f, 0.3f, -0.1f,
            0.4f, 0.1f, -0.2f, 0.2f, 0.6f, 0.3f,
            0.1f, 0.5f, 0.2f, -0.4f, 0.1f, 0.2f
        }),
        .languageModelBias = Tensor<float>({ 6 }, 0.0f)
    };
    const kairo::transformers::DecoderModel model(twoHead, modelWeights);
    assert(kairo::transformers::GreedyToken(
        Tensor<float>({ 4 }, { 1.0f, 3.0f, 3.0f, 2.0f })) == 1);
    const std::vector<std::size_t> prompt{ 1, 4, 2 };
    const Tensor<float> fullLogits = model.Forward(prompt);
    const auto greedyFirst = model.GenerateGreedy(prompt, 2);
    const auto greedySecond = model.GenerateGreedy(prompt, 2);
    assert(greedyFirst == greedySecond);
    const std::filesystem::path decoderCheckpoint =
        std::filesystem::temp_directory_path() / "kairo-decoder-checkpoint.bin";
    std::filesystem::remove(decoderCheckpoint);
    kairo::transformers::SaveDecoderCheckpoint(decoderCheckpoint, modelWeights);
    auto importedWeights = kairo::transformers::LoadDecoderCheckpoint(
        decoderCheckpoint, twoHead, 1u << 20);
    const kairo::transformers::DecoderModel importedModel(
        twoHead, std::move(importedWeights));
    assert(importedModel.GenerateGreedy(prompt, 2) == greedyFirst);
    kairo::transformers::KVCache runtimeCache(twoHead);
    Tensor<float> cachedLogits;
    for (std::size_t position = 0; position < prompt.size(); ++position)
    {
        cachedLogits = model.Decode(prompt[position], position, runtimeCache);
        for (std::size_t token = 0; token < twoHead.vocabularySize; ++token)
            assert(std::abs(cachedLogits(0, token) - fullLogits(position, token)) < 1e-5f);
    }
    assert(runtimeCache.Length(0) == prompt.size());

    kairo::transformers::SamplingConfig sampling{
        .temperature = 0.8f,
        .topK = 3,
        .topP = 0.9f,
        .seed = 12345
    };
    const auto firstGeneration = model.Generate(prompt, 3, sampling);
    const auto secondGeneration = model.Generate(prompt, 3, sampling);
    assert(firstGeneration == secondGeneration);
    assert(firstGeneration.size() == prompt.size() + 3);

    const Tensor<float> denseWeight({ 4, 3 }, {
        0.12f, -0.31f, 0.52f,
        -0.44f, 0.23f, 0.71f,
        0.35f, 0.08f, -0.63f,
        0.91f, -0.57f, 0.19f
    });
    const auto quantized = kairo::transformers::QuantizeInt8(denseWeight);
    assert(quantized.StorageBytes() < denseWeight.Size() * sizeof(float));
    const Tensor<float> quantizedInput({ 2, 4 }, {
        1.0f, -0.5f, 0.25f, 0.75f,
        -0.2f, 0.8f, -0.7f, 0.1f
    });
    const Tensor<float> floatProduct =
        kairo::foundation::math::MatMul(quantizedInput, denseWeight);
    const Tensor<float> quantizedProduct =
        kairo::transformers::QuantizedMatMul(quantizedInput, quantized);
    for (std::size_t index = 0; index < floatProduct.Size(); ++index)
        assert(std::abs(floatProduct[index] - quantizedProduct[index]) < 0.01f);
    const auto quantizedInt4 = kairo::transformers::QuantizeInt4(denseWeight);
    assert(quantizedInt4.StorageBytes() < quantized.StorageBytes());
    const Tensor<float> int4Product =
        kairo::transformers::QuantizedMatMul(quantizedInput, quantizedInt4);
    float maximumInt4Error = 0.0f;
    for (std::size_t index = 0; index < floatProduct.Size(); ++index)
        maximumInt4Error = std::max(
            maximumInt4Error, std::abs(floatProduct[index] - int4Product[index]));
    assert(maximumInt4Error < 0.18f);

    const std::filesystem::path archivePath =
        std::filesystem::temp_directory_path() / "kairo-transformer-weights.bin";
    std::filesystem::remove(archivePath);
    kairo::transformers::BoundedTensorArchive::Save(archivePath, {
        { "embedding", modelWeights.tokenEmbedding },
        { "layers.0.weight", denseWeight },
        { "layers.1.weight", denseWeight * 2.0f }
    });
    const kairo::transformers::BoundedTensorArchive archive(archivePath);
    assert(archive.TensorCount() == 3);
    const Tensor<float> loadedEmbedding =
        archive.Load("embedding", modelWeights.tokenEmbedding.Size() * sizeof(float));
    assert(loadedEmbedding(4, 2) == modelWeights.tokenEmbedding(4, 2));
    bool budgetRejected = false;
    try
    {
        (void)archive.Load("embedding", sizeof(float));
    }
    catch (const std::length_error&)
    {
        budgetRejected = true;
    }
    assert(budgetRejected);
    std::size_t streamedLayers = 0;
    archive.ForEachLayer(2, { "weight" }, denseWeight.Size() * sizeof(float),
        [&](std::size_t layer,
            const std::unordered_map<std::string, Tensor<float>>& resident)
        {
            assert(resident.size() == 1);
            assert(resident.at("weight")(0, 0) == denseWeight(0, 0) * (layer + 1));
            ++streamedLayers;
        });
    assert(streamedLayers == 2);
    std::filesystem::remove(decoderCheckpoint);
    std::filesystem::remove(archivePath);
    return 0;
}
