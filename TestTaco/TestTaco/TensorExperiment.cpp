#include <chrono>
#include <iostream>
#include <vector>

#include <omp.h>
#include <taco.h>

#include "DataGenerator.h"
#include "TensorBuilder.h"
#include "TensorExperiment.h"

constexpr int DIMENSION_COUNT = 100;
constexpr int ELEMENTS_PER_DIMENSION = 100;
constexpr int VALUE_COUNT = 1'000'000;
constexpr int THREAD_COUNT = 8;
constexpr unsigned RANDOM_SEED = 12345;

void runTensorExperiment()
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

    std::cout << "Генерация: " << generationTime << " сек\n"
        << std::flush;

    // ================================================================
    // Построение тензора новым методом.
    // ================================================================

    std::vector<bool> denseDimensions(      //Изначально все измерения разрежены (sparse)
        DIMENSION_COUNT,
        false
    );

    //Отдельно задаем плотные измерения (dense)
    denseDimensions[0] = true;
    //denseDimensions[4] = true;

    std::cout << "\nПостроение тензора новым методом...\n"
        << std::flush;

    auto buildStart = std::chrono::steady_clock::now();

    taco::Tensor<double> tensor = buildTensorNew(
        data,
        DIMENSION_COUNT,
        ELEMENTS_PER_DIMENSION,
        denseDimensions
    );

    auto buildEnd = std::chrono::steady_clock::now();

    double buildTime = std::chrono::duration<double>(
        buildEnd - buildStart
    ).count();

    std::cout << "Построение: " << buildTime << " сек\n"
        << std::flush;

    // ================================================================
    // СТАРЫЙ ЭКСПЕРИМЕНТ
    //
    // Свёртка:
    //
    // T(D0, D1, ..., D99) -> R(D0)
    //
    // R(D0) = SUM(D1, ..., D99) T(D0, D1, ..., D99)
    //
    // D0 сохраняется в результате.
    // D1...D99 являются редукционными измерениями.
    // ================================================================

    std::vector<taco::IndexVar> indexVars(DIMENSION_COUNT);

    std::vector<int> resultDimensions = {
        ELEMENTS_PER_DIMENSION
    };

    taco::Tensor<double> result(
        resultDimensions,
        taco::Format({
            taco::ModeFormat::Dense
            })
    );

    auto contractionStart = std::chrono::steady_clock::now();

    result(indexVars[0]) = tensor(indexVars);

    auto expressionEnd = std::chrono::steady_clock::now();

    result.compile();

    auto compileEnd = std::chrono::steady_clock::now();

    result.assemble();

    auto assembleEnd = std::chrono::steady_clock::now();

    result.compute();

    auto computeEnd = std::chrono::steady_clock::now();

    double expressionTime = std::chrono::duration<double>(
        expressionEnd - contractionStart
    ).count();

    double compileTime = std::chrono::duration<double>(
        compileEnd - expressionEnd
    ).count();

    double assembleTime = std::chrono::duration<double>(
        assembleEnd - compileEnd
    ).count();

    double computeTime = std::chrono::duration<double>(
        computeEnd - assembleEnd
    ).count();

    double contractionTime = std::chrono::duration<double>(
        computeEnd - contractionStart
    ).count();

    std::cout << "\nРезультаты свёртки:\n";
    std::cout << "Формирование выражения: " << expressionTime << " сек\n";
    std::cout << "Compile: " << compileTime << " сек\n";
    std::cout << "Assemble: " << assembleTime << " сек\n";
    std::cout << "Compute: " << computeTime << " сек\n";
    std::cout << "Вся свёртка: " << contractionTime << " сек\n"
        << std::flush;

    // ================================================================
    // НОВЫЙ ЭКСПЕРИМЕНТ: СВЁРТКА С ФИЛЬТРОМ
    //
    // Выполняем ту же свёртку:
    //
    // T(D0, D1, ..., D99) -> R(D0)
    //
    // но для измерения №10 используем только первые 50 элементов:
    //
    // D9 = 0..49
    //
    // То есть:
    //
    // R(D0) =
    //     SUM(D1...D8)
    //     SUM(D9 = 0..49)
    //     SUM(D10...D99)
    //     T(D0, ..., D99)
    //
    // Остальные измерения полностью участвуют в свёртке.
    // ================================================================

    std::cout << "\n========================================\n";
    std::cout << "Новый эксперимент: свёртка с фильтром\n";
    std::cout << "Измерение №10: элементы 0..49\n";
    std::cout << "========================================\n";

    taco::Tensor<double> filteredResult(
        resultDimensions,
        taco::Format({
            taco::ModeFormat::Dense
            })
    );

    // Для программного формирования доступа используем
    // IndexVarInterface, поскольку один из индексов будет
    // WindowedIndexVar.
    std::vector<std::shared_ptr<taco::IndexVarInterface>> filteredIndexVars;

    filteredIndexVars.reserve(DIMENSION_COUNT);

    for (int i = 0; i < DIMENSION_COUNT; ++i) {
        if (i == 9) {
            // Измерение №10: только элементы 0..49.
            filteredIndexVars.push_back(
                std::make_shared<taco::WindowedIndexVar>(
                    indexVars[i],
                    0,
                    50,
                    1
                )
            );
        }
        else {
            filteredIndexVars.push_back(
                std::make_shared<taco::IndexVar>(indexVars[i])
            );
        }
    }

    // TensorBase имеет специальный operator() для набора
    // IndexVar / WindowedIndexVar.
    taco::TensorBase& tensorBase = tensor;

    auto filteredContractionStart =
        std::chrono::steady_clock::now();

    filteredResult(indexVars[0]) =
        tensorBase(filteredIndexVars);

    auto filteredExpressionEnd =
        std::chrono::steady_clock::now();

    filteredResult.compile();

    auto filteredCompileEnd =
        std::chrono::steady_clock::now();

    filteredResult.assemble();

    auto filteredAssembleEnd =
        std::chrono::steady_clock::now();

    filteredResult.compute();

    auto filteredComputeEnd =
        std::chrono::steady_clock::now();

    double filteredExpressionTime =
        std::chrono::duration<double>(
            filteredExpressionEnd - filteredContractionStart
        ).count();

    double filteredCompileTime =
        std::chrono::duration<double>(
            filteredCompileEnd - filteredExpressionEnd
        ).count();

    double filteredAssembleTime =
        std::chrono::duration<double>(
            filteredAssembleEnd - filteredCompileEnd
        ).count();

    double filteredComputeTime =
        std::chrono::duration<double>(
            filteredComputeEnd - filteredAssembleEnd
        ).count();

    double filteredContractionTime =
        std::chrono::duration<double>(
            filteredComputeEnd - filteredContractionStart
        ).count();

    std::cout << "\nРезультаты свёртки с фильтром:\n";
    std::cout << "Формирование выражения: "
        << filteredExpressionTime << " сек\n";
    std::cout << "Compile: "
        << filteredCompileTime << " сек\n";
    std::cout << "Assemble: "
        << filteredAssembleTime << " сек\n";
    std::cout << "Compute: "
        << filteredComputeTime << " сек\n";
    std::cout << "Вся свёртка: "
        << filteredContractionTime << " сек\n"
        << std::flush;
}