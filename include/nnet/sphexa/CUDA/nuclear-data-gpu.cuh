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
 * @brief Definition of CUDA GPU data.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include "../../../util/CUDA/cuda.inl"

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include "sph/traits.hpp"

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/concatenate.hpp"

#include "cstone/util/reallocate.hpp"
#include "cstone/util/array.hpp"

#include <thrust/device_vector.h>

namespace sphexa::sphnnet {
	/*! @brief device nuclear data class for n_species nuclear network */
	template<size_t n_species, typename Float, typename Int>
	class DeviceNuclearDataType : public cstone::FieldStates<DeviceNuclearDataType<n_species, Float, Int>> {
	public:
		//! maximum number of nuclear species
		static const int maxNumSpecies = 100;
		//! actual number of nuclear species
		int numSpecies = n_species;

		// types
		using RealType = Float;
    	using KeyType  = Int;
	    using Tmass    = float;
	    using XM1Type  = float;

		// could replace rho_m1 -> drho_dt
		//!  @brief hydro data
		thrust::device_vector<RealType> c, p, cv, dpdT, u, rho, temp, rho_m1;
		//!  @brief particle mass (lower precision)
		thrust::device_vector<Tmass> m;

		//!  @brief nuclear abundances (vector of vector)
		util::array<thrust::device_vector<RealType>, maxNumSpecies> Y;

		//! @brief timesteps
		thrust::device_vector<RealType> dt;

		//! solver buffer
		mutable thrust::device_vector<RealType> buffer;

		//!  resize the number of particules
		void resize(size_t size);


		//! base hydro fieldNames (every nuclear species is named "Yn")
		inline static constexpr auto fieldNames = concat(enumerateFieldNames<"Y", maxNumSpecies>(), std::array<const char*, 12>{
			"nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1",
		});

		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<
	    		thrust::device_vector<int>*,
	    		thrust::device_vector<unsigned>*,
	    		thrust::device_vector<uint64_t>*,
	    		thrust::device_vector<float>*,
	    		thrust::device_vector<double>*>;

			util::array<FieldType, fieldNames.size()> data;

			for (int i = 0; i < maxNumSpecies; ++i) 
				data[i] = &Y[i];

			data[maxNumSpecies]      = &node_id;
			data[maxNumSpecies + 1]  = &particle_id;
			data[maxNumSpecies + 2]  = &dt;
			data[maxNumSpecies + 3]  = &c;
			data[maxNumSpecies + 4]  = &p;
			data[maxNumSpecies + 5]  = &cv;
			data[maxNumSpecies + 6]  = &u;
			data[maxNumSpecies + 7]  = &dpdT;
			data[maxNumSpecies + 8]  = &m;
			data[maxNumSpecies + 9]  = &temp;
			data[maxNumSpecies + 10] = &rho;
			data[maxNumSpecies + 11] = &rho_m1;

			return data;
	    }
	 };
}