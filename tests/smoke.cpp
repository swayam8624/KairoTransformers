import Kairo.Transformers;

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
    return 0;
}
