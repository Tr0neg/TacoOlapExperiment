#include "TensorComparison.h"

#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>
#include <taco.h>

#include "DataGenerator.h"
#include "TensorBuilder.h"

namespace
{

    constexpr int DIMENSION_COUNT = 100;
    constexpr int ELEMENTS_PER_DIMENSION = 100;
    constexpr int VALUE_COUNT = 100'000;
    constexpr int THREAD_COUNT = 8;
    constexpr unsigned RANDOM_SEED = 12345;

}

void compareTensorBuilding()
{
    omp_set_num_threads(THREAD_COUNT);

    std::cout << "Параметры эксперимента:\n";
    std::cout << "Количество измерений: " << DIMENSION_COUNT << "\n";
    std::cout << "Элементов в каждом измерении: " << ELEMENTS_PER_DIMENSION << "\n";
    std::cout << "Количество исходных значений: " << VALUE_COUNT << "\n";
    std::cout << "Количество потоков: " << THREAD_COUNT << "\n";
    std::cout << "Начальное значение генератора: " << RANDOM_SEED << "\n";
    std::cout << std::flush;

    std::cout << "\nГенерация данных...\n" << std::flush;

    auto generationStart = std::chrono::steady_clock::now();

    std::vector<DataPoint> data = generateData(
        DIMENSION_COUNT,
        ELEMENTS_PER_DIMENSION,
        VALUE_COUNT,
        RANDOM_SEED
    );

    auto generationEnd = std::chrono::steady_clock::now();

    double generationTime = std::chrono::duration<double>(
        generationEnd - generationStart
    ).count();

    std::cout << "Генерация: " << generationTime << " сек\n" << std::flush;

    std::vector<std::pair<std::string, std::vector<bool>>> tests;

    tests.emplace_back(
        "1. Полностью разреженный",
        std::vector<bool>(DIMENSION_COUNT, false)
    );

    std::vector<bool> firstDense(DIMENSION_COUNT, false);
    firstDense[0] = true;

    tests.emplace_back(
        "2. Первое измерение плотное",
        firstDense
    );

    std::vector<bool> firstTwoDense(DIMENSION_COUNT, false);
    firstTwoDense[0] = true;
    firstTwoDense[1] = true;

    tests.emplace_back(
        "3. Первое и второе измерения плотные",
        firstTwoDense
    );

    std::vector<bool> secondDense(DIMENSION_COUNT, false);
    secondDense[1] = true;

    tests.emplace_back(
        "4. Только второе измерение плотное",
        secondDense
    );

    std::vector<bool> lastDense(DIMENSION_COUNT, false);
    lastDense[DIMENSION_COUNT - 1] = true;

    tests.emplace_back(
        "5. Последнее измерение плотное",
        lastDense
    );

    std::vector<bool> secondLastDense(DIMENSION_COUNT, false);
    secondLastDense[1] = true;
    secondLastDense[DIMENSION_COUNT - 1] = true;

    tests.emplace_back(
        "6. Второе и последнее измерения плотные",
        secondLastDense
    );

    std::vector<bool> firstLastDense(DIMENSION_COUNT, false);
    firstLastDense[0] = true;
    firstLastDense[DIMENSION_COUNT - 1] = true;

    tests.emplace_back(
        "7. Первое и последнее измерения плотные",
        firstLastDense
    );

    for (const auto& test : tests)
    {
        const auto& name = test.first;
        const auto& denseDimensions = test.second;

        std::cout << "\n=== " << name << " ===\n" << std::flush;

        auto oldStart = std::chrono::steady_clock::now();

        taco::Tensor<double> tensorOld = buildTensorOld(
            data,
            DIMENSION_COUNT,
            ELEMENTS_PER_DIMENSION,
            denseDimensions
        );

        auto oldEnd = std::chrono::steady_clock::now();

        double oldTime = std::chrono::duration<double>(
            oldEnd - oldStart
        ).count();

        auto newStart = std::chrono::steady_clock::now();

        taco::Tensor<double> tensorNew = buildTensorNew(
            data,
            DIMENSION_COUNT,
            ELEMENTS_PER_DIMENSION,
            denseDimensions
        );

        auto newEnd = std::chrono::steady_clock::now();

        double newTime = std::chrono::duration<double>(
            newEnd - newStart
        ).count();

        auto compareStart = std::chrono::steady_clock::now();

        bool tensorsEqual = compareTensors(
            tensorOld,
            tensorNew
        );

        auto compareEnd = std::chrono::steady_clock::now();

        double compareTime = std::chrono::duration<double>(
            compareEnd - compareStart
        ).count();

        std::cout << "Старый способ: " << oldTime << " сек\n";
        std::cout << "Новый способ: " << newTime << " сек\n";
        std::cout << "Тензоры совпадают: "
            << (tensorsEqual ? "да" : "НЕТ")
            << " (" << compareTime << " сек)\n"
            << std::flush;
    }
}