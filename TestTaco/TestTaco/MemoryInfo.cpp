#include "MemoryInfo.h"

#include <sys/resource.h>

size_t getTensorStorageSize(
    const taco::Tensor<double>& tensor
)
{
    const taco::TensorStorage& storage =
        tensor.getStorage();

    size_t totalBytes = 0;

    // Значения тензора.
    totalBytes +=
        storage.getValues().getSize() *
        sizeof(double);

    const taco::Index& index =
        storage.getIndex();

    // Индексные массивы pos и idx.
    for (int dimension = 0;
        dimension < index.numModeIndices();
        ++dimension)
    {
        const taco::ModeIndex& mode =
            index.getModeIndex(dimension);

        for (int array = 0;
            array < mode.numIndexArrays();
            ++array)
        {
            const taco::Array& indexArray =
                mode.getIndexArray(array);

            totalBytes +=
                indexArray.getSize() *
                sizeof(int);
        }
    }

    return totalBytes;
}


size_t getPeakProcessMemory()
{
    struct rusage usage {};

    if (getrusage(RUSAGE_SELF, &usage) != 0)
    {
        return 0;
    }

    // В Linux ru_maxrss задаётся в КиБ.
    return static_cast<size_t>(usage.ru_maxrss) * 1024;
}