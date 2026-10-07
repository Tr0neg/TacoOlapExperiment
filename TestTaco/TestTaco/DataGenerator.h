#pragma once

#include <vector>

struct DataPoint
{
    std::vector<int> coordinates;
    double value;
};

std::vector<DataPoint> generateData(
    int dimensionCount,
    int elementsPerDimension,
    int valueCount,
    unsigned seed
);