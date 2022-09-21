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

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include "sph/data_util.hpp"
#include "sph/field_states.hpp"
#include "sph/traits.hpp"

#include "cstone/util/util.hpp"
#include "cstone/util/array.hpp"

#include "../../CUDA/cuda.inl"
#include <thrust/device_vector.h>

namespace sphexa::sphnnet {
	/*! @brief device nuclear data class for n_species nuclear network */
	template<size_t n_species, typename Float, typename Int>
	class DeviceNuclearDataType : public FieldStates<DeviceNuclearDataType<n_species, Float, Int>> {
	public:
		// types
		using RealType = Float;

		//!  @brief hydro data
		thrust::device_vector<RealType> c, p, cv, dpdT, u, m, rho, temp, previous_rho;

		//!  @brief nuclear abundances (vector of vector)
		util::array<thrust::device_vector<RealType>, n_species> Y;
		//! device pointers to nuclear abundances vectors
		thrust::device_vector<RealType*> Y_dev_ptr;

		//! @brief timesteps
		thrust::device_vector<RealType> dt;

		//! solver buffer
		mutable thrust::device_vector<RealType> buffer;

		//!  resize the number of particules
		void resize(size_t size);

		//!  base fieldNames (without knowledge of nuclear species names)
		inline static constexpr std::array fieldNames {
			"nid", "pid",
			"dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho", "Y",
		};


		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<
		    	util::array<thrust::device_vector<RealType>, n_species>*,
		    	thrust::device_vector<RealType>*>;

			return util::array<FieldType, fieldNames.size()>{
				(thrust::device_vector<RealType>*)nullptr, (thrust::device_vector<RealType>*)nullptr,
				&dt, &c, &p, &cv, &u, &dpdT, &m, &temp, &rho, &previous_rho, &Y
			};
	    }
	 };
}