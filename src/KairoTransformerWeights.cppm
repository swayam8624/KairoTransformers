module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

export module Kairo.Transformers.Weights;

import Kairo.Foundation.Math.Tensor;

export namespace kairo::transformers
{
    using kairo::foundation::math::Tensor;

    struct QuantizedMatrixInt8 final
    {
        std::size_t rows = 0;
        std::size_t columns = 0;
        std::vector<std::int8_t> values;
        std::vector<float> columnScales;

        [[nodiscard]] std::size_t StorageBytes() const noexcept
        {
            return values.size() * sizeof(std::int8_t)
                + columnScales.size() * sizeof(float);
        }
    };

    struct QuantizedMatrixInt4 final
    {
        std::size_t rows = 0;
        std::size_t columns = 0;
        std::vector<std::uint8_t> packedValues;
        std::vector<float> columnScales;

        [[nodiscard]] std::size_t StorageBytes() const noexcept
        {
            return packedValues.size() + columnScales.size() * sizeof(float);
        }
    };

    /// Symmetric per-output-column INT8 quantization for dense weights
    /// [inputWidth,outputWidth]. Zero columns use scale one and remain zero.
    [[nodiscard]] inline QuantizedMatrixInt8 QuantizeInt8(const Tensor<float>& weight)
    {
        if (weight.Rank() != 2 || weight.Empty())
            throw std::invalid_argument("QuantizeInt8 expects a non-empty rank-2 weight.");
        QuantizedMatrixInt8 result{
            .rows = weight.Dim(0),
            .columns = weight.Dim(1),
            .values = std::vector<std::int8_t>(weight.Size()),
            .columnScales = std::vector<float>(weight.Dim(1), 1.0f)
        };
        for (std::size_t column = 0; column < weight.Dim(1); ++column)
        {
            float maximum = 0.0f;
            for (std::size_t row = 0; row < weight.Dim(0); ++row)
                maximum = std::max(maximum, std::abs(weight(row, column)));
            const float scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
            result.columnScales[column] = scale;
            for (std::size_t row = 0; row < weight.Dim(0); ++row)
            {
                const float quantized = std::round(weight(row, column) / scale);
                result.values[row * weight.Dim(1) + column] =
                    static_cast<std::int8_t>(std::clamp(quantized, -127.0f, 127.0f));
            }
        }
        return result;
    }

    [[nodiscard]] inline Tensor<float> Dequantize(const QuantizedMatrixInt8& weight)
    {
        if (weight.rows == 0 || weight.columns == 0
            || weight.values.size() != weight.rows * weight.columns
            || weight.columnScales.size() != weight.columns)
            throw std::invalid_argument("Quantized INT8 matrix metadata is invalid.");
        Tensor<float> output({ weight.rows, weight.columns }, 0.0f);
        for (std::size_t row = 0; row < weight.rows; ++row)
            for (std::size_t column = 0; column < weight.columns; ++column)
                output(row, column) =
                    static_cast<float>(weight.values[row * weight.columns + column])
                    * weight.columnScales[column];
        return output;
    }

    /// Float32 activation x INT8 weight with Float32 accumulation.
    [[nodiscard]] inline Tensor<float> QuantizedMatMul(
        const Tensor<float>& input, const QuantizedMatrixInt8& weight)
    {
        if (input.Rank() != 2 || input.Dim(1) != weight.rows
            || weight.values.size() != weight.rows * weight.columns
            || weight.columnScales.size() != weight.columns)
            throw std::invalid_argument("QuantizedMatMul shape mismatch.");
        Tensor<float> output({ input.Dim(0), weight.columns }, 0.0f);
        for (std::size_t row = 0; row < input.Dim(0); ++row)
            for (std::size_t column = 0; column < weight.columns; ++column)
            {
                float sum = 0.0f;
                for (std::size_t inner = 0; inner < weight.rows; ++inner)
                    sum += input(row, inner)
                        * static_cast<float>(weight.values[inner * weight.columns + column]);
                output(row, column) = sum * weight.columnScales[column];
            }
        return output;
    }

    [[nodiscard]] inline QuantizedMatrixInt4 QuantizeInt4(const Tensor<float>& weight)
    {
        if (weight.Rank() != 2 || weight.Empty())
            throw std::invalid_argument("QuantizeInt4 expects a non-empty rank-2 weight.");
        QuantizedMatrixInt4 result{
            .rows = weight.Dim(0),
            .columns = weight.Dim(1),
            .packedValues = std::vector<std::uint8_t>((weight.Size() + 1) / 2, 0),
            .columnScales = std::vector<float>(weight.Dim(1), 1.0f)
        };
        for (std::size_t column = 0; column < weight.Dim(1); ++column)
        {
            float maximum = 0.0f;
            for (std::size_t row = 0; row < weight.Dim(0); ++row)
                maximum = std::max(maximum, std::abs(weight(row, column)));
            const float scale = maximum == 0.0f ? 1.0f : maximum / 7.0f;
            result.columnScales[column] = scale;
            for (std::size_t row = 0; row < weight.Dim(0); ++row)
            {
                const int quantized = static_cast<int>(std::clamp(
                    std::round(weight(row, column) / scale), -7.0f, 7.0f));
                const std::size_t linear = row * weight.Dim(1) + column;
                const std::uint8_t nibble = static_cast<std::uint8_t>(quantized + 8);
                if ((linear & 1u) == 0)
                    result.packedValues[linear / 2] = static_cast<std::uint8_t>(
                        (result.packedValues[linear / 2] & 0xF0u) | nibble);
                else
                    result.packedValues[linear / 2] = static_cast<std::uint8_t>(
                        (result.packedValues[linear / 2] & 0x0Fu) | (nibble << 4));
            }
        }
        return result;
    }

    [[nodiscard]] inline Tensor<float> QuantizedMatMul(
        const Tensor<float>& input, const QuantizedMatrixInt4& weight)
    {
        if (input.Rank() != 2 || input.Dim(1) != weight.rows
            || weight.packedValues.size() != (weight.rows * weight.columns + 1) / 2
            || weight.columnScales.size() != weight.columns)
            throw std::invalid_argument("INT4 QuantizedMatMul shape mismatch.");
        Tensor<float> output({ input.Dim(0), weight.columns }, 0.0f);
        for (std::size_t row = 0; row < input.Dim(0); ++row)
            for (std::size_t column = 0; column < weight.columns; ++column)
            {
                float sum = 0.0f;
                for (std::size_t inner = 0; inner < weight.rows; ++inner)
                {
                    const std::size_t linear = inner * weight.columns + column;
                    const std::uint8_t packed = weight.packedValues[linear / 2];
                    const std::uint8_t nibble =
                        (linear & 1u) == 0 ? packed & 0x0Fu : packed >> 4;
                    sum += input(row, inner)
                        * static_cast<float>(static_cast<int>(nibble) - 8);
                }
                output(row, column) = sum * weight.columnScales[column];
            }
        return output;
    }

    struct NamedTensor final
    {
        std::string name;
        Tensor<float> value;
    };

    /// Indexed Float32 tensor archive. Construction reads only metadata.
    /// `Load` seeks directly to one payload and enforces the caller's maximum
    /// resident tensor bytes. `ForEachLayer` releases each layer map before
    /// loading the next, providing AirLLM-style bounded layer streaming.
    class BoundedTensorArchive final
    {
    public:
        static void Save(
            const std::filesystem::path& path,
            const std::vector<NamedTensor>& tensors)
        {
            if (path.empty() || tensors.empty())
                throw std::invalid_argument("Tensor archive Save requires path and tensors.");
            const std::filesystem::path temporary = path.string() + ".tmp";
            std::FILE* file = std::fopen(temporary.c_str(), "wb");
            if (!file) throw std::runtime_error("Cannot create tensor archive.");
            constexpr char magic[8] = { 'K', 'A', 'I', 'R', 'O', 'W', 'G', 'T' };
            const std::uint32_t version = 1;
            const std::uint64_t count = tensors.size();
            bool success = WriteBytes(file, magic, sizeof(magic))
                && Write(file, version) && Write(file, count);
            std::unordered_map<std::string, bool> names;
            for (const NamedTensor& named : tensors)
            {
                if (named.name.empty() || named.value.Empty()
                    || !names.emplace(named.name, true).second)
                {
                    success = false;
                    break;
                }
                const Tensor<float> tensor = named.value.Contiguous();
                const std::uint64_t nameSize = named.name.size();
                const std::uint64_t rank = tensor.Rank();
                const std::uint64_t bytes = tensor.Size() * sizeof(float);
                success = success && Write(file, nameSize)
                    && WriteBytes(file, named.name.data(), named.name.size())
                    && Write(file, rank);
                for (std::size_t axis = 0; success && axis < tensor.Rank(); ++axis)
                {
                    const std::uint64_t dimension = tensor.Dim(axis);
                    success = Write(file, dimension);
                }
                success = success && Write(file, bytes)
                    && WriteBytes(file, tensor.Data(), static_cast<std::size_t>(bytes));
            }
            success = std::fclose(file) == 0 && success;
            if (!success)
            {
                std::filesystem::remove(temporary);
                throw std::runtime_error("Tensor archive write failed.");
            }
            std::error_code error;
            std::filesystem::rename(temporary, path, error);
            if (error)
            {
                std::filesystem::remove(temporary);
                throw std::runtime_error("Tensor archive atomic rename failed.");
            }
        }

        explicit BoundedTensorArchive(std::filesystem::path path)
            : path_(std::move(path))
        {
            Index();
        }

        [[nodiscard]] std::size_t TensorCount() const noexcept { return entries_.size(); }

        [[nodiscard]] Tensor<float> Load(
            const std::string& name, std::size_t maximumBytes) const
        {
            const auto iterator = entries_.find(name);
            if (iterator == entries_.end())
                throw std::out_of_range("Tensor archive has no entry named " + name);
            const Entry& entry = iterator->second;
            if (entry.bytes > maximumBytes)
                throw std::length_error("Tensor exceeds bounded archive load budget.");
            std::FILE* file = std::fopen(path_.c_str(), "rb");
            if (!file) throw std::runtime_error("Cannot reopen tensor archive.");
            if (std::fseek(file, static_cast<long>(entry.offset), SEEK_SET) != 0)
            {
                std::fclose(file);
                throw std::runtime_error("Tensor archive seek failed.");
            }
            Tensor<float> tensor(entry.shape, 0.0f);
            const bool success = ReadBytes(
                file, tensor.Data(), static_cast<std::size_t>(entry.bytes));
            std::fclose(file);
            if (!success) throw std::runtime_error("Tensor archive payload is truncated.");
            return tensor;
        }

        void ForEachLayer(
            std::size_t layerCount,
            const std::vector<std::string>& suffixes,
            std::size_t maximumTensorBytes,
            const std::function<void(
                std::size_t, const std::unordered_map<std::string, Tensor<float>>&)>& callback) const
        {
            if (!callback || suffixes.empty())
                throw std::invalid_argument("Layer streaming requires suffixes and callback.");
            for (std::size_t layer = 0; layer < layerCount; ++layer)
            {
                std::unordered_map<std::string, Tensor<float>> resident;
                for (const std::string& suffix : suffixes)
                {
                    const std::string name =
                        "layers." + std::to_string(layer) + "." + suffix;
                    resident.emplace(suffix, Load(name, maximumTensorBytes));
                }
                callback(layer, resident);
            }
        }

    private:
        struct Entry final
        {
            std::vector<std::size_t> shape;
            std::uint64_t offset = 0;
            std::uint64_t bytes = 0;
        };
        std::filesystem::path path_;
        std::unordered_map<std::string, Entry> entries_;

        template<typename T>
        static bool Write(std::FILE* file, const T& value)
        {
            return WriteBytes(file, &value, sizeof(value));
        }
        template<typename T>
        static bool Read(std::FILE* file, T& value)
        {
            return ReadBytes(file, &value, sizeof(value));
        }
        static bool WriteBytes(std::FILE* file, const void* data, std::size_t bytes)
        {
            return std::fwrite(data, 1, bytes, file) == bytes;
        }
        static bool ReadBytes(std::FILE* file, void* data, std::size_t bytes)
        {
            return std::fread(data, 1, bytes, file) == bytes;
        }

        void Index()
        {
            std::FILE* file = std::fopen(path_.c_str(), "rb");
            if (!file) throw std::runtime_error("Cannot open tensor archive.");
            char magic[8]{};
            constexpr char expected[8] = { 'K', 'A', 'I', 'R', 'O', 'W', 'G', 'T' };
            std::uint32_t version = 0;
            std::uint64_t count = 0;
            bool success = ReadBytes(file, magic, sizeof(magic))
                && std::memcmp(magic, expected, sizeof(magic)) == 0
                && Read(file, version) && version == 1
                && Read(file, count) && count > 0 && count <= 1'000'000;
            for (std::uint64_t item = 0; success && item < count; ++item)
            {
                std::uint64_t nameSize = 0, rank = 0, bytes = 0;
                success = Read(file, nameSize) && nameSize > 0 && nameSize <= 4096;
                std::string name(static_cast<std::size_t>(nameSize), '\0');
                success = success && ReadBytes(file, name.data(), name.size())
                    && Read(file, rank) && rank > 0 && rank <= 16;
                std::vector<std::size_t> shape(static_cast<std::size_t>(rank));
                std::uint64_t elements = 1;
                for (std::uint64_t axis = 0; success && axis < rank; ++axis)
                {
                    std::uint64_t dimension = 0;
                    success = Read(file, dimension) && dimension > 0
                        && elements <= std::numeric_limits<std::uint64_t>::max() / dimension;
                    if (success)
                    {
                        elements *= dimension;
                        shape[static_cast<std::size_t>(axis)] =
                            static_cast<std::size_t>(dimension);
                    }
                }
                success = success && Read(file, bytes)
                    && bytes == elements * sizeof(float);
                const long offset = std::ftell(file);
                success = success && offset >= 0
                    && bytes <= static_cast<std::uint64_t>(std::numeric_limits<long>::max())
                    && std::fseek(file, static_cast<long>(bytes), SEEK_CUR) == 0
                    && entries_.emplace(name, Entry{
                        std::move(shape), static_cast<std::uint64_t>(offset), bytes
                    }).second;
            }
            if (success)
            {
                const int trailing = std::fgetc(file);
                success = trailing == EOF;
            }
            std::fclose(file);
            if (!success)
            {
                entries_.clear();
                throw std::runtime_error("Tensor archive is malformed.");
            }
        }
    };
}
