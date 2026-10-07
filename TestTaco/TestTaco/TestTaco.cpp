#include <iostream>
#include <vector>

#include <taco.h>

#include "DataGenerator.h"
#include "TensorBuilder.h"

constexpr int DIMENSION_COUNT = 4;
constexpr int ELEMENTS_PER_DIMENSION = 20;
constexpr int VALUE_COUNT = 5'000;
constexpr unsigned RANDOM_SEED = 12345;

int main()
{   
    taco::Tensor<double> B_data({ 20,20, 10, 10}, taco::Format({ taco::ModeFormat::Sparse,taco::ModeFormat::Sparse,taco::ModeFormat::Sparse,taco::ModeFormat::Sparse }));
    taco::Tensor<double> A({ 20,20 }, taco::Format({ taco::ModeFormat::Sparse,taco::ModeFormat::Sparse }));
    taco::IndexVar d1, d2, d3, d4;

    std::cout << "Генерация данных...\n";

    GenerateToTensor(B_data, 500, 12345);

    std::cout << "Генерация завершена.\n";

    //Свёртка по измерениям (порядок индексов в источнике и приемнике должен совпадать)
    A(d1, d2) = B_data(d1, d2, d3, d4);

    A.evaluate();

    std::cout << "\nСвёртка по измерениям - Матрица A:\n";
    for (int i = 0; i < A.getDimension(0); ++i) {
        for (int j = 0; j < A.getDimension(1); ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << "\n";
    }

    /*
    //Перемножение двух срезов
    A(d1, d3) = B_data(d1, d3, d2, d4) * B_data(d2, d3, d1, d4);

    std::cout << "\nСвёртка по измерениям - Матрица A:\n";
    for (int i = 0; i < A.getDimension(0); ++i) {
        for (int j = 0; j < A.getDimension(1); ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << "\n";
    }
    */
    return 0;
}