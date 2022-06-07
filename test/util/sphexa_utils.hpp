#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <variant>
#include <type_traits>
#include <array>

#include <iostream>

namespace util {
	template<class T, int n>
	using array = std::array<T, n>;
}
template<class T, int n>
std::ostream& operator<<(std::ostream& os, const util::array<T, n>& Y);
namespace sphexa {
	namespace fileutils {
		/*! @brief write fields as columns to an ASCII file
		 *
		 * @tparam  T              field type
		 * @tparam  Separators
		 * @param   firstIndex     first field index to write
		 * @param   lastIndex      last field index to write
		 * @param   path           the file name to write to
		 * @param   append         append or overwrite if file already exists
		 * @param   fields         pointers to field array, each field is a column
		 * @param   separators     arbitrary number of separators to insert between columns, eg '\t', std::setw(n), ...
		 */
		template<class... T, class... Separators>
		void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, const std::vector<std::variant<T*...>>& fields, Separators&&... separators) {
		    std::ios_base::openmode mode;
		    if (append) { mode = std::ofstream::app; }
		    else { mode = std::ofstream::out; }

		    std::ofstream dumpFile(path, mode);

		    if (dumpFile.is_open())
		    {
		        for (size_t i = firstIndex; i < lastIndex; ++i)
		        {
		            for (auto field : fields)
		            {
		                [[maybe_unused]] std::initializer_list<int> list{(dumpFile << separators, 0)...};
		                std::visit([&dumpFile, i](auto& arg) { 
			                	dumpFile << arg[i];
			                }, field);
		            }
		            dumpFile << std::endl;
		        }
		    }
		    else { throw std::runtime_error("Can't open file at path: " + path); }

		    dumpFile.close();
		}
	}

	//! @brief extract a vector of pointers to particle fields for file output
	template<class Dataset>
	auto getOutputArrays(Dataset& dataset)
	{
	    auto fieldPointers = dataset.data();
	    using FieldType    = std::variant<float*, double*, int*, unsigned*, uint64_t*, uint8_t* /*,
	    	util::array<double, 14>*, util::array<double, 86>*, util::array<double, 87>*,
	    	util::array<float, 14>*,  util::array<float, 86>*,  util::array<float, 87>* */>;

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