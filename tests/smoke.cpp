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
    return 0;
}
