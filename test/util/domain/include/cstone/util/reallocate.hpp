#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <variant>
#include <type_traits>

#include "util.hpp"

//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}