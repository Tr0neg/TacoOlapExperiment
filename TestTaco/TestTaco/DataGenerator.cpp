#include "DataGenerator.h"

#include <random>

std::vector<DataPoint> generateData(
    int dimensionCount,
    int elementsPerDimension,
    int valueCount,
    unsigned seed
)
{
    std::vector<DataPoint> data;
    data.reserve(valueCount);

    std::mt19937 generator(seed);

    std::uniform_int_distribution<int> coord(
        0,
        elementsPerDimension - 1
    );

    std::uniform_int_distribution<int> value(1, 1000);

    std::vector<int> coordinates(dimensionCount);

    for (int n = 0; n < valueCount; n++)
    {
        for (int d = 0; d < dimensionCount; d++)
        {
            coordinates[d] = coord(generator);
        }

        data.push_back({
            coordinates,
            static_cast<double>(value(generator))
            });
    }

    return data;
}