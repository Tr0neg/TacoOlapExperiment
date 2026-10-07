#pragma once

#include <vector>

#include <taco.h>

#include "DataGenerator.h"

taco::Tensor<double> buildTensorOld(
    const std::vector<DataPoint>& data,
    int dimensionCount,
    int elementsPerDimension,
    const std::vector<bool>& denseDimensions
);

taco::Tensor<double> buildTensorNew(
    const std::vector<DataPoint>& data,
    int dimensionCount,
    int elementsPerDimension,
    const std::vector<bool>& denseDimensions
);

bool compareTensors(
    const taco::Tensor<double>& tensor1,
    const taco::Tensor<double>& tensor2
);

void GenerateToTensor(
    taco::Tensor<double>& tensor,
    int valueCount,
    unsigned seed
);