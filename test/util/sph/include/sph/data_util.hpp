#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <variant>
#include <type_traits>
#include <array>

#include "cstone/util/array.hpp"

namespace sphexa {
    //! @brief extract a vector of pointers to particle fields for file output
    template<class Dataset>
    auto getOutputArrays(Dataset& dataset)
    {
        auto fieldPointers = dataset.data();
        using FieldType    = std::variant<float*, double*, int*, unsigned*, uint64_t*,
            util::array<double, 14>*, util::array<double, 86>*, util::array<double, 87>*,
            util::array<float,  14>*, util::array<float,  86>*, util::array<float,  87>*>;

        std::vector<FieldType> outputFields;
        outputFields.reserve(dataset.outputFieldIndices.size());

        for (int i : dataset.outputFieldIndices)
        {
            if (!dataset.isAllocated(i))
            {
                throw std::runtime_error("Cannot output field " + std::string(dataset.fieldNames[i]) +
                                         ", because it is not active.");
            }
            std::visit([&outputFields](auto& arg) { outputFields.push_back(arg->data()); }, fieldPointers[i]);
        }
        return outputFields;
    }

    /*! @brief look up indices of field names
     *
     * @tparam     Array
     * @param[in]  allNames     array of strings with names of all fields
     * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
     * @return                  the indices of @p subsetNames in @p allNames
     */
    template<class Array>
    std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
    {
        std::vector<int> subsetIndices;
        subsetIndices.reserve(subsetNames.size());
        for (const auto& field : subsetNames)
        {
            auto it = std::find(allNames.begin(), allNames.end(), field);
            if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

            size_t fieldIndex = it - allNames.begin();
            subsetIndices.push_back(fieldIndex);
        }
        return subsetIndices;
    }
}
