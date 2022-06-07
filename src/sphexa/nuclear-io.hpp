#pragma once

#include <vector>
#include <variant>
#include <tuple>

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
#endif

namespace sphexa::sphnnet::io {
	std::vector<int>         outputFieldIndices;
	std::vector<std::string> outputFieldNames;

	void setOutputFieldsNames(const int num_species) {
		outputFieldNames.resize(num_species);
		for (int i = 0; i < num_species; ++i) 
			outputFieldNames[i] = "Y(" + std::to_string(i) + ")";
	}

	void setOutputFieldsNames(const std::vector<std::string> &species_names) {
    	const int num_species = species_names.size();
		outputFieldNames.resize(num_species);
		for (int i = 0; i < num_species; ++i) 
			outputFieldNames[i] = "Y(" + species_names[i] + ")";
	}

	std::vector<std::string> setOutputFields(const std::vector<std::string>& outFields) {
		std::vector<std::string> nuclearOutFields, hydroOutFields = outFields;

		// separate nuclear and hydro vectors
		for (auto it = hydroOutFields.rbegin(); it != hydroOutFields.rend(); ++it)
			if (std::count(outputFieldNames.begin(), outputFieldNames.end(), *it)) {
				nuclearOutFields.insert(nuclearOutFields.begin(), *it);
				hydroOutFields.erase(it.base());
			}

		outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, nuclearOutFields);
		return hydroOutFields;
    }
}

/// overwritten output operator for nuclear array
/**
 * TODO
 */
template<typename T, size_t n>
std::ofstream& operator<<(std::ofstream& os, const std::array<T, n>& Y) {
	for (int idx : sphexa::sphnnet::io::outputFieldIndices)
		os << Y[idx] << " ";
    return os;
}