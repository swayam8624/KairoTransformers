#include <cassert>
#include <cstddef>
#include <filesystem>
#include <vector>

import Kairo.Transformers;
import Kairo.Transformers.Training;
import Kairo.Foundation.Math.TensorTraining;

int main()
{
    using namespace kairo::transformers;
    TransformerConfig config{
        .vocabularySize = 4,
        .contextLength = 7,
        .modelWidth = 8,
        .headCount = 2,
        .layerCount = 1,
        .feedForwardWidth = 16,
        .activation = Activation::GELU
    };
    const std::vector<std::vector<std::size_t>> inputs{
        { 0, 1, 2, 3, 0, 1, 2 },
        { 2, 3, 0, 1, 2, 3, 0 }
    };
    const std::vector<std::vector<std::size_t>> targets{
        { 1, 2, 3, 0, 1, 2, 3 },
        { 3, 0, 1, 2, 3, 0, 1 }
    };

    TrainableDecoder model(config, 1234);
    kairo::foundation::math::TensorOptimizer optimizer(
        DefaultTransformerAdamW(0.02f));
    float initialLoss = 0.0f;
    TransformerTrainingMetrics metrics;
    for (std::size_t step = 0; step < 500; ++step)
    {
        metrics = model.TrainAccumulated(inputs, targets, optimizer);
        if (step == 0) initialLoss = metrics.loss;
    }
    assert(metrics.loss < initialLoss);
    assert(metrics.loss < 0.03f);
    assert(metrics.accuracy == 1.0f);
    assert(metrics.optimizerStep == 500);

    const std::filesystem::path checkpoint =
        std::filesystem::temp_directory_path() / "kairo-transformer-training.bin";
    std::filesystem::remove(checkpoint);
    model.Save(checkpoint, optimizer);
    TrainableDecoder restored(config, 9999);
    kairo::foundation::math::TensorOptimizer restoredOptimizer;
    restored.Load(checkpoint, restoredOptimizer);
    assert(restored.Accuracy(inputs, targets) == 1.0f);
    assert(restoredOptimizer.CompletedSteps() == optimizer.CompletedSteps());
    const auto resumed = restored.TrainAccumulated(inputs, targets, restoredOptimizer);
    const auto uninterrupted = model.TrainAccumulated(inputs, targets, optimizer);
    assert(resumed.loss == uninterrupted.loss);
    const auto restoredParameters = restored.Parameters();
    const auto modelParameters = model.Parameters();
    for (std::size_t parameter = 0; parameter < modelParameters.size(); ++parameter)
        for (std::size_t index = 0; index < modelParameters[parameter]->Value().Size(); ++index)
            assert(restoredParameters[parameter]->Value()[index]
                == modelParameters[parameter]->Value()[index]);
    std::filesystem::remove(checkpoint);
    return 0;
}
