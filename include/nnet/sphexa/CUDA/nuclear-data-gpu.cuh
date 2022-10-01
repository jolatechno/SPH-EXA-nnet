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

#include "sph/data_util.hpp"
#include "sph/field_states.hpp"
#include "sph/traits.hpp"

#include "cstone/util/util.hpp"
#include "cstone/util/array.hpp"

#include <thrust/device_vector.h>

namespace sphexa::sphnnet {
	/*! @brief device nuclear data class for n_species nuclear network */
	template<size_t n_species, typename Float, typename Int>
	class DeviceNuclearDataType : public FieldStates<DeviceNuclearDataType<n_species, Float, Int>> {
	public:
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
		util::array<thrust::device_vector<RealType>, n_species> Y;

		//! @brief timesteps
		thrust::device_vector<RealType> dt;

		//! solver buffer
		mutable thrust::device_vector<RealType> buffer;

		//!  resize the number of particules
		void resize(size_t size);

		//! base hydro fieldNames (withoutnuclear species names)
		inline static constexpr std::array baseFieldNames {
			"nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1",
		};
		//! base hydro fieldNames (every nuclear species is named "Y")
		inline static constexpr auto fieldNames = []{
			std::array<std::string_view, baseFieldNames.size()+n_species> fieldNames;
			for (int i = 0; i < baseFieldNames.size(); ++i)
				fieldNames[i] = baseFieldNames[i];
			for (int i = baseFieldNames.size(); i < fieldNames.size(); ++i)
				fieldNames[i] = "Y";
		    return fieldNames;
		}();


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

			util::array<FieldType, fieldNames.size()> data = {
				(thrust::device_vector<int>*)nullptr, (thrust::device_vector<KeyType>*)nullptr,
				&dt, &c, &p, &cv, &u, &dpdT, &m, &temp, &rho, &rho_m1
			};

			for (int i = 0; i < n_species; ++i) 
				data[baseFieldNames.size() + i] = &Y[i];

			return data;
	    }
	 };
}