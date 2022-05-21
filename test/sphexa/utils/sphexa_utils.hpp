#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

namespace sphexa {
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