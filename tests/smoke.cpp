import Kairo.Transformers;
import Kairo.Foundation.Math.Tensor;

#include <cassert>

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

    const Tensor<float> query({ 2, 2 }, { 1.0f, 0.0f, 0.0f, 1.0f });
    const Tensor<float> key({ 2, 2 }, { 1.0f, 0.0f, 0.0f, 1.0f });
    const Tensor<float> value({ 2, 2 }, { 3.0f, 4.0f, 7.0f, 8.0f });
    const auto attended = kairo::transformers::CausalScaledDotProductAttention(query, key, value);
    assert(attended(0, 0) == 3.0f && attended(0, 1) == 4.0f);
    assert(attended(1, 0) > 3.0f && attended(1, 0) < 7.0f);
    return 0;
}
