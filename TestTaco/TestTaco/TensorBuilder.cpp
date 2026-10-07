#include "TensorBuilder.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#include <omp.h>
#include <parallel/algorithm>


namespace
{

    // Создаёт формат TACO из признаков плотности измерений.
    taco::Format createFormat(
        int dimensionCount,
        const std::vector<bool>& denseDimensions
    )
    {
        if (
            denseDimensions.size() !=
            static_cast<size_t>(dimensionCount)
            )
        {
            throw std::runtime_error(
                "Количество признаков плотности не совпадает "
                "с количеством измерений"
            );
        }

        std::vector<taco::ModeFormatPack> modeFormats(
            dimensionCount,
            taco::ModeFormat::Sparse
        );

        for (int dimension = 0;
            dimension < dimensionCount;
            ++dimension)
        {
            if (denseDimensions[dimension])
            {
                modeFormats[dimension] =
                    taco::ModeFormat::Dense;
            }
        }

        return taco::Format(modeFormats);
    }

}


// ================================================================
// Старый способ построения.
// ================================================================

taco::Tensor<double> buildTensorOld(
    const std::vector<DataPoint>& data,
    int dimensionCount,
    int elementsPerDimension,
    const std::vector<bool>& denseDimensions
)
{
    taco::Format format =
        createFormat(
            dimensionCount,
            denseDimensions
        );

    std::vector<int> dimensions(
        dimensionCount,
        elementsPerDimension
    );

    taco::Tensor<double> tensor(
        dimensions,
        format
    );

    for (const auto& point : data)
    {
        tensor.insert(
            point.coordinates,
            point.value
        );
    }

    tensor.pack();

    return tensor;
}

// ================================================================
// Новый способ построения.
// ================================================================

struct DimensionStructure
{
    std::vector<int> previousSparse;
    std::vector<size_t> denseProduct;
};

struct ModeIndices
{
    std::vector<std::vector<int>> allIdx;
    std::vector<std::vector<int>> allPos;
};

std::vector<size_t> sortData(
    const std::vector<DataPoint>& data,
    int dimensionCount)
{
    std::vector<size_t> order(data.size());
    std::iota(order.begin(), order.end(), 0);

    __gnu_parallel::sort(
        order.begin(),
        order.end(),
        [&](size_t a, size_t b)
        {
            const auto& ca = data[a].coordinates;
            const auto& cb = data[b].coordinates;

            for (int d = 0; d < dimensionCount; ++d)
            {
                if (ca[d] != cb[d])
                    return ca[d] < cb[d];
            }

            return false;
        }
    );

    return order;
}

void prepareSortedData(
    const std::vector<DataPoint>& data,
    const std::vector<size_t>& order,
    int dimensionCount,
    std::vector<int>& coordinates,
    std::vector<double>& values)
{
    coordinates.resize(order.size() * dimensionCount);
    values.resize(order.size());

    size_t uniqueCount = 0;

    for (size_t position = 0; position < order.size(); ++position)
    {
        const auto& point = data[order[position]];

        if (uniqueCount == 0)
        {
            for (int d = 0; d < dimensionCount; ++d)
                coordinates[d] = point.coordinates[d];

            values[0] = point.value;
            uniqueCount = 1;
            continue;
        }

        const int* previous =
            coordinates.data() + (uniqueCount - 1) * dimensionCount;

        bool same = true;

        for (int d = 0; d < dimensionCount; ++d)
        {
            if (previous[d] != point.coordinates[d])
            {
                same = false;
                break;
            }
        }

        if (same)
        {
            values[uniqueCount - 1] += point.value;
        }
        else
        {
            int* current =
                coordinates.data() + uniqueCount * dimensionCount;

            for (int d = 0; d < dimensionCount; ++d)
                current[d] = point.coordinates[d];

            values[uniqueCount] = point.value;
            ++uniqueCount;
        }
    }

    coordinates.resize(uniqueCount * dimensionCount);
    values.resize(uniqueCount);
}

std::vector<int> calculateFirstChanged(
    const std::vector<int>& coordinates,
    size_t uniqueCount,
    int dimensionCount)
{
    std::vector<int> firstChanged(
        uniqueCount,
        dimensionCount
    );

    for (size_t row = 1; row < uniqueCount; ++row)
    {
        const int* current =
            coordinates.data() + row * dimensionCount;

        const int* previous = current - dimensionCount;

        for (int d = 0; d < dimensionCount; ++d)
        {
            if (current[d] != previous[d])
            {
                firstChanged[row] = d;
                break;
            }
        }
    }

    return firstChanged;
}

std::vector<size_t> calculateSparseIdxCount(
    const std::vector<int>& firstChanged,
    size_t uniqueCount,
    int dimensionCount)
{
    std::vector<size_t> changedAt(
        dimensionCount,
        0
    );

    for (size_t row = 1; row < uniqueCount; ++row)
        ++changedAt[firstChanged[row]];

    std::vector<size_t> sparseIdxCount(
        dimensionCount,
        0
    );

    size_t changedCount = 0;

    for (int d = 0; d < dimensionCount; ++d)
    {
        changedCount += changedAt[d];
        sparseIdxCount[d] = changedCount + 1;
    }

    return sparseIdxCount;
}

DimensionStructure calculateDimensionStructure(
    int dimensionCount,
    const std::vector<int>& dimensions,
    const std::vector<bool>& denseDimensions)
{
    DimensionStructure result;

    result.previousSparse.resize(
        dimensionCount,
        -1
    );

    int lastSparse = -1;

    for (int d = 0; d < dimensionCount; ++d)
    {
        result.previousSparse[d] = lastSparse;

        if (!denseDimensions[d])
            lastSparse = d;
    }

    result.denseProduct.resize(
        dimensionCount + 1,
        1
    );

    for (int d = dimensionCount - 1; d >= 0; --d)
    {
        result.denseProduct[d] =
            result.denseProduct[d + 1];

        if (denseDimensions[d])
        {
            result.denseProduct[d] *=
                static_cast<size_t>(
                    dimensions[d]
                    );
        }
    }

    return result;
}

ModeIndices buildModeIndices(
    const std::vector<int>& coordinates,
    const std::vector<int>& firstChanged,
    const std::vector<size_t>& sparseIdxCount,
    const std::vector<int>& previousSparse,
    const std::vector<size_t>& denseProduct,
    size_t uniqueCount,
    int dimensionCount,
    const std::vector<int>& dimensions,
    const std::vector<bool>& denseDimensions)
{
    ModeIndices result;

    result.allIdx.resize(dimensionCount);
    result.allPos.resize(dimensionCount);

#pragma omp parallel for schedule(dynamic)
    for (int d = 0; d < dimensionCount; ++d)
    {
        if (denseDimensions[d])
        {
            result.allIdx[d] = { dimensions[d] };
            continue;
        }

        const int previous = previousSparse[d];

        auto& idx = result.allIdx[d];
        auto& pos = result.allPos[d];

        idx.reserve(sparseIdxCount[d]);
        pos.push_back(0);

        size_t sparseRank = 0;
        size_t currentParent = 0;
        size_t idxPosition = 0;

        for (size_t row = 0; row < uniqueCount; ++row)
        {
            if (row > 0 &&
                previous >= 0 &&
                firstChanged[row] <= previous)
            {
                ++sparseRank;
            }

            size_t parent = 0;

            if (previous >= 0)
            {
                const size_t denseCount =
                    denseProduct[previous + 1] /
                    denseProduct[d];

                size_t denseOffset = 0;

                for (int k = previous + 1; k < d; ++k)
                {
                    if (denseDimensions[k])
                    {
                        denseOffset =
                            denseOffset *
                            static_cast<size_t>(
                                dimensions[k]
                                ) +
                            static_cast<size_t>(
                                coordinates[
                                    row * dimensionCount + k
                                ]
                                );
                    }
                }

                parent =
                    sparseRank * denseCount +
                    denseOffset;
            }
            else
            {
                for (int k = 0; k < d; ++k)
                {
                    if (denseDimensions[k])
                    {
                        parent =
                            parent *
                            static_cast<size_t>(
                                dimensions[k]
                                ) +
                            static_cast<size_t>(
                                coordinates[
                                    row * dimensionCount + k
                                ]
                                );
                    }
                }
            }

            while (currentParent < parent)
            {
                ++currentParent;

                pos.push_back(
                    static_cast<int>(idxPosition)
                );
            }

            if (row == 0 ||
                firstChanged[row] <= d)
            {
                idx.push_back(
                    coordinates[
                        row * dimensionCount + d
                    ]
                );

                ++idxPosition;
            }
        }

        while (currentParent <
            (previous >= 0
                ? sparseIdxCount[previous] *
                (
                    denseProduct[previous + 1] /
                    denseProduct[d]
                    )
                : denseProduct[0] /
                denseProduct[d]))
        {
            ++currentParent;

            pos.push_back(
                static_cast<int>(idxPosition)
            );
        }
    }

    return result;
}

std::vector<double> buildPackedValues(
    const std::vector<int>& coordinates,
    const std::vector<double>& values,
    const std::vector<int>& firstChanged,
    const std::vector<size_t>& sparseIdxCount,
    const std::vector<int>& previousSparse,
    const std::vector<size_t>& denseProduct,
    size_t uniqueCount,
    int dimensionCount,
    const std::vector<int>& dimensions,
    const std::vector<bool>& denseDimensions)
{
    const int lastDimension =
        dimensionCount - 1;

    if (!denseDimensions[lastDimension])
        return values;

    const int previous =
        previousSparse[lastDimension];

    size_t valueCountPacked;

    if (previous < 0)
    {
        valueCountPacked =
            denseProduct[0];
    }
    else
    {
        valueCountPacked =
            sparseIdxCount[previous] *
            (
                denseProduct[previous + 1] /
                denseProduct[dimensionCount]
                );
    }

    std::vector<double> packedValues(
        valueCountPacked,
        0.0
    );

    size_t sparseRank = 0;

    for (size_t row = 0; row < uniqueCount; ++row)
    {
        if (row > 0 &&
            previous >= 0 &&
            firstChanged[row] <= previous)
        {
            ++sparseRank;
        }

        size_t position = 0;

        if (previous >= 0)
        {
            for (int k = previous + 1;
                k < dimensionCount;
                ++k)
            {
                if (denseDimensions[k])
                {
                    position =
                        position *
                        static_cast<size_t>(
                            dimensions[k]
                            ) +
                        static_cast<size_t>(
                            coordinates[
                                row * dimensionCount + k
                            ]
                            );
                }
            }

            position +=
                sparseRank *
                (
                    denseProduct[previous + 1] /
                    denseProduct[dimensionCount]
                    );
        }
        else
        {
            for (int k = 0;
                k < dimensionCount;
                ++k)
            {
                if (denseDimensions[k])
                {
                    position =
                        position *
                        static_cast<size_t>(
                            dimensions[k]
                            ) +
                        static_cast<size_t>(
                            coordinates[
                                row * dimensionCount + k
                            ]
                            );
                }
            }
        }

        packedValues[position] =
            values[row];
    }

    return packedValues;
}

taco::Tensor<double> createTensorFromIndices(
    const ModeIndices& modeIndices,
    const std::vector<double>& packedValues,
    int dimensionCount,
    const std::vector<int>& dimensions,
    const std::vector<bool>& denseDimensions)
{
    std::vector<taco::ModeIndex> tacoModeIndices;

    tacoModeIndices.reserve(
        dimensionCount
    );

    for (int d = 0; d < dimensionCount; ++d)
    {
        if (denseDimensions[d])
        {
            tacoModeIndices.emplace_back(
                std::vector<taco::Array>{
                taco::makeArray(
                    modeIndices.allIdx[d]
                )
            }
            );
        }
        else
        {
            tacoModeIndices.emplace_back(
                std::vector<taco::Array>{
                taco::makeArray(
                    modeIndices.allPos[d]
                ),
                    taco::makeArray(
                        modeIndices.allIdx[d]
                    )
            }
            );
        }
    }

    taco::Format format =
        createFormat(
            dimensionCount,
            denseDimensions
        );

    taco::Index index(
        format,
        tacoModeIndices
    );

    taco::TensorStorage storage(
        taco::type<double>(),
        dimensions,
        format,
        taco::Literal(0.0)
    );

    storage.setIndex(index);

    storage.setValues(
        taco::makeArray(packedValues)
    );

    taco::Tensor<double> tensor(
        dimensions,
        format
    );

    tensor.setStorage(storage);

    return tensor;
}

// ================================================================
// Новый способ построения.
// ================================================================

taco::Tensor<double> buildTensorNew(
    const std::vector<DataPoint>& data,
    int dimensionCount,
    int elementsPerDimension,
    const std::vector<bool>& denseDimensions)
{
    std::vector<int> dimensions(
        dimensionCount,
        elementsPerDimension
    );

    std::vector<size_t> order = sortData(
        data,
        dimensionCount
    );

    std::vector<int> coordinates;
    std::vector<double> values;

    prepareSortedData(
        data,
        order,
        dimensionCount,
        coordinates,
        values
    );

    const size_t uniqueCount =
        values.size();

    std::vector<int> firstChanged =
        calculateFirstChanged(
            coordinates,
            uniqueCount,
            dimensionCount
        );

    std::vector<size_t> sparseIdxCount =
        calculateSparseIdxCount(
            firstChanged,
            uniqueCount,
            dimensionCount
        );

    DimensionStructure dimensionStructure =
        calculateDimensionStructure(
            dimensionCount,
            dimensions,
            denseDimensions
        );

    const auto& previousSparse =
        dimensionStructure.previousSparse;

    const auto& denseProduct =
        dimensionStructure.denseProduct;

    ModeIndices modeIndices =
        buildModeIndices(
            coordinates,
            firstChanged,
            sparseIdxCount,
            previousSparse,
            denseProduct,
            uniqueCount,
            dimensionCount,
            dimensions,
            denseDimensions
        );

    std::vector<double> packedValues =
        buildPackedValues(
            coordinates,
            values,
            firstChanged,
            sparseIdxCount,
            previousSparse,
            denseProduct,
            uniqueCount,
            dimensionCount,
            dimensions,
            denseDimensions
        );

    return createTensorFromIndices(
        modeIndices,
        packedValues,
        dimensionCount,
        dimensions,
        denseDimensions
    );
}


////////////////////////////////////////////////
/// Сравнение тензоров
////////////////////////////////////////////////
bool compareTensors(
    const taco::Tensor<double>& tensor1,
    const taco::Tensor<double>& tensor2
)
{
    std::string tensor1String;
    std::string tensor2String;

    {
        std::ostringstream stream;
        stream << tensor1;
        tensor1String = stream.str();
    }

    {
        std::ostringstream stream;
        stream << tensor2;
        tensor2String = stream.str();
    }

    const size_t firstNewline1 =
        tensor1String.find('\n');

    const size_t firstNewline2 =
        tensor2String.find('\n');

    if (firstNewline1 != std::string::npos &&
        firstNewline2 != std::string::npos)
    {
        tensor1String =
            tensor1String.substr(firstNewline1 + 1);

        tensor2String =
            tensor2String.substr(firstNewline2 + 1);
    }

    return tensor1String == tensor2String;
}

//Генерация данных в тензор любой размерности
void GenerateToTensor(
    taco::Tensor<double>& tensor,
    int valueCount,
    unsigned seed
)
{
    const int dimensionCount =
        tensor.getOrder();

    const std::vector<int> dimensions =
        tensor.getDimensions();

    if (dimensionCount == 0)
        throw std::runtime_error(
            "Тензор не содержит измерений"
        );

    if (dimensions.size() !=
        static_cast<size_t>(dimensionCount))
    {
        throw std::runtime_error(
            "Некорректные размеры тензора"
        );
    }

    for (int d = 0; d < dimensionCount; ++d)
    {
        if (dimensions[d] <= 0)
        {
            throw std::runtime_error(
                "Размер измерения должен быть положительным"
            );
        }
    }

    const auto& modeFormats =
        tensor.getFormat().getModeFormats();

    std::vector<bool> denseDimensions(
        dimensionCount
    );

    for (int d = 0; d < dimensionCount; ++d)
    {
        denseDimensions[d] =
            modeFormats[d] ==
            taco::ModeFormat::Dense;
    }

    std::vector<DataPoint> data;

    data.reserve(valueCount);

    std::mt19937 generator(seed);

    std::vector<
        std::uniform_int_distribution<int>
    > coordinatesDistribution;

    coordinatesDistribution.reserve(
        dimensionCount
    );

    for (int d = 0; d < dimensionCount; ++d)
    {
        coordinatesDistribution.emplace_back(
            0,
            dimensions[d] - 1
        );
    }

    std::uniform_int_distribution<int> value(
        1,
        1000
    );

    std::vector<int> coordinates(
        dimensionCount
    );

    for (int n = 0; n < valueCount; ++n)
    {
        for (int d = 0; d < dimensionCount; ++d)
        {
            coordinates[d] =
                coordinatesDistribution[d](
                    generator
                    );
        }

        data.push_back({
            coordinates,
            static_cast<double>(
                value(generator)
            )
            });
    }

    std::vector<size_t> order =
        sortData(
            data,
            dimensionCount
        );

    std::vector<int> sortedCoordinates;
    std::vector<double> values;

    prepareSortedData(
        data,
        order,
        dimensionCount,
        sortedCoordinates,
        values
    );

    const size_t uniqueCount =
        values.size();

    std::vector<int> firstChanged =
        calculateFirstChanged(
            sortedCoordinates,
            uniqueCount,
            dimensionCount
        );

    std::vector<size_t> sparseIdxCount =
        calculateSparseIdxCount(
            firstChanged,
            uniqueCount,
            dimensionCount
        );

    DimensionStructure dimensionStructure =
        calculateDimensionStructure(
            dimensionCount,
            dimensions,
            denseDimensions
        );

    const auto& previousSparse =
        dimensionStructure.previousSparse;

    const auto& denseProduct =
        dimensionStructure.denseProduct;

    ModeIndices modeIndices =
        buildModeIndices(
            sortedCoordinates,
            firstChanged,
            sparseIdxCount,
            previousSparse,
            denseProduct,
            uniqueCount,
            dimensionCount,
            dimensions,
            denseDimensions
        );

    std::vector<double> packedValues =
        buildPackedValues(
            sortedCoordinates,
            values,
            firstChanged,
            sparseIdxCount,
            previousSparse,
            denseProduct,
            uniqueCount,
            dimensionCount,
            dimensions,
            denseDimensions
        );

    taco::Format format =
        tensor.getFormat();

    std::vector<taco::ModeIndex> tacoModeIndices;

    tacoModeIndices.reserve(
        dimensionCount
    );

    for (int d = 0; d < dimensionCount; ++d)
    {
        if (denseDimensions[d])
        {
            tacoModeIndices.emplace_back(
                std::vector<taco::Array>{
                taco::makeArray(
                    modeIndices.allIdx[d]
                )
            }
            );
        }
        else
        {
            tacoModeIndices.emplace_back(
                std::vector<taco::Array>{
                taco::makeArray(
                    modeIndices.allPos[d]
                ),
                    taco::makeArray(
                        modeIndices.allIdx[d]
                    )
            }
            );
        }
    }

    taco::Index index(
        format,
        tacoModeIndices
    );

    taco::TensorStorage storage(
        taco::type<double>(),
        dimensions,
        format,
        taco::Literal(0.0)
    );

    storage.setIndex(index);

    storage.setValues(
        taco::makeArray(packedValues)
    );

    tensor.setStorage(storage);
}