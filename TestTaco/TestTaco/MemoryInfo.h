#pragma once

#include <cstddef>

#include <taco.h>

// Возвращает объём памяти, занимаемый внутренним
// хранилищем TACO-тензора: values + pos + idx.
size_t getTensorStorageSize(
    const taco::Tensor<double>& tensor
);

// Возвращает максимальный объём резидентной памяти
// текущего процесса в байтах.
size_t getPeakProcessMemory();