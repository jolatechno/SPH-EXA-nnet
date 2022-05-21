#pragma once

#include <variant>
#include <vector>
#include <fstream>

#include "../nuclear-data.hpp"

namespace sphexa {
	namespace fileutils {
		/*! @brief write fields as columns to an ASCII file
		 *
		 * @tparam  T              "vector" type that can be indexed via the "[]" operator.
		 * @tparam  Separators
		 * @param   firstIndex     first field index to write
		 * @param   lastIndex      last field index to write
		 * @param   path           the file name to write to
		 * @param   append         append or overwrite if file already exists
		 * @param   fields         pointers to field array, each field is a column
		 * @param   separators     arbitrary number of separators to insert between columns, eg '\t', std::setw(n), ...
		 */
		template<class... T, class... Separators>
		void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, const std::vector<std::variant<T...>>& fields, Separators&&... separators) {
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
		                std::visit([&dumpFile, i](auto& arg) { dumpFile << (*arg)[i]; }, field);
		            }
		            dumpFile << std::endl;
		        }
		    }
		    else { throw std::runtime_error("Can't open file at path: " + path); }

		    dumpFile.close();
		}
	}

	//! @brief extract a vector of reference to nuclear particle fields for file output
	template<int n_species, typename Float=double>
	auto getOutputArrays(sphnnet::NuclearDataType<n_species, Float> &dataset) {
	    auto fieldPointers = dataset.data();
	    
	    decltype(fieldPointers) outputFields;
	    outputFields.reserve(dataset.outputFieldIndices.size());

	    for (int i : dataset.outputFieldIndices)
	    {
	        if (!dataset.isAllocated(i))
	        {
	            throw std::runtime_error("Cannot output field " + std::string(dataset.fieldNames[i]) +
	                                     ", because it is not active.");
	        }
	        std::visit([&outputFields](auto& arg) { outputFields.emplace_back(arg); }, fieldPointers[i]);
	    }
	    return outputFields;
	}
}