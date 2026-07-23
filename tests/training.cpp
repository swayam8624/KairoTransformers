#include <cassert>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <vector>

import Kairo.Transformers;
import Kairo.Transformers.Training;
import Kairo.Foundation.Math.Tensor;
import Kairo.Foundation.Math.TensorAutograd;
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

    TrainableDecoder retainedGraph(config, 4321);
    TrainableDecoder recomputedGraph(config, 4321);
    recomputedGraph.SetGradientCheckpointing(true);
    assert(recomputedGraph.GradientCheckpointing());
    const auto retainedLoss = retainedGraph.Loss(inputs[0], targets[0]);
    const auto recomputedLoss = recomputedGraph.Loss(inputs[0], targets[0]);
    assert(retainedLoss.Value()[0] == recomputedLoss.Value()[0]);
    retainedLoss.Backward();
    recomputedLoss.Backward();
    const auto retainedParameters = retainedGraph.Parameters();
    const auto recomputedParameters = recomputedGraph.Parameters();
    for (std::size_t parameter = 0; parameter < retainedParameters.size(); ++parameter)
        for (std::size_t index = 0;
             index < retainedParameters[parameter]->Gradient().Size(); ++index)
            assert(std::abs(retainedParameters[parameter]->Gradient()[index]
                - recomputedParameters[parameter]->Gradient()[index]) < 1e-5f);

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

    using kairo::foundation::math::MeanSquaredLoss;
    using kairo::foundation::math::Tensor;
    using kairo::foundation::math::Variable;
    const Tensor<float> adapterInputs({ 4, 2 }, {
        1, 0, 0, 1, 1, 1, -1, 0.5f
    });
    const Tensor<float> frozenBase({ 2, 2 }, 0.0f);
    const Tensor<float> adapterTargets({ 4, 2 }, {
        2, -1, 0.5f, 3, 2.5f, 2, -1.75f, 2.5f
    });
    LoRAProjection adapter(2, 2, 2, 2.0f, 77);
    kairo::foundation::math::TensorOptimizer adapterOptimizer(
        DefaultTransformerAdamW(0.05f));
    float adapterLoss = 0.0f;
    for (std::size_t step = 0; step < 400; ++step)
    {
        auto adapterParameters = adapter.Parameters();
        for (Variable* parameter : adapterParameters) parameter->ZeroGradient();
        const Variable output = adapter.Forward(Variable(adapterInputs), frozenBase);
        const Variable loss = MeanSquaredLoss(output, adapterTargets);
        adapterLoss = loss.Value()[0];
        loss.Backward();
        adapterOptimizer.Step(adapterParameters);
    }
    assert(adapterLoss < 1e-4f);
    for (std::size_t index = 0; index < frozenBase.Size(); ++index)
        assert(frozenBase[index] == 0.0f);
    std::filesystem::remove(checkpoint);
    return 0;
}
