/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Utility function to interface nuclear networks with SPH-EXA IO module.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include <vector>
#include <variant>
#include <tuple>

#include "sph/data_util.hpp"

namespace sphexa::sphnnet::io {
	std::vector<int>         outputFieldIndices;
	std::vector<std::string> outputFieldNames;

	/*! @brief function used to set the outputnames of nuclear data
	 * 
	 * Simply names the ith species "i" and its abundance "Y(i)"
	 * 
	 * @param num_species  number of nuclear species
	 */
	void setOutputFieldsNames(const int num_species) {
		outputFieldNames.resize(num_species);
		for (int i = 0; i < num_species; ++i) 
			outputFieldNames[i] = "Y(" + std::to_string(i) + ")";
	}

	/*! @brief function used to set the outputnames of nuclear data
	 * 
	 * Simply names the abundance of a species named "X" "Y(X)"
	 * 
	 * @param species_names  vector of species name
	 */
	void setOutputFieldsNames(const std::vector<std::string> &species_names) {
    	const int num_species = species_names.size();
		outputFieldNames.resize(num_species);
		for (int i = 0; i < num_species; ++i) 
			outputFieldNames[i] = "Y(" + species_names[i] + ")";
	}

	/*! @brief function used to set the nuclear data to be outputed. Note that this list of species is global.
	 * 
	 * @param outFields  vector of species name to be outputed
	 */
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

    /*! @brief function used to right abundances as lines
     * 
     * Used un HDF5 part in SPH-EXA
     * 
     * @param Y            vector of array represneting abundances of particles
     * @param n_particles  number of particles for which the abundances should be righten
     * @param write_func   function used to right a line (taking a pointer to the floating point data, and a fieldname)
     * 
     * Returns the last output of write_func.
     */
	template<typename T, size_t n, class writter_function>
	auto write_lines(const util::array<T, n> *Y, size_t n_particles, const writter_function &write_func) {
		std::vector<T> buffer(n_particles);
		for (int i = 0;; ++i) {
			const         int  fieldIdx  = outputFieldIndices[i];
			const std::string& fieldName = outputFieldNames[  i];

			// move to buffer
			for (size_t j = 0; j < n_particles; ++j)
				buffer[j] = Y[j][fieldIdx]; 

			// write
			if (i >= outputFieldIndices.size() - 1)
				return write_func(fieldName, buffer.data());
			else
				write_func(fieldName, buffer.data());
		}
	}
}

// overwritten print operator for nuclear abundances array
template<typename T, size_t n>
std::ofstream& operator<<(std::ofstream& os, const util::array<T, n>& Y) {
	for (int idx : sphexa::sphnnet::io::outputFieldIndices)
		os << Y[idx] << " ";
    return os;
}