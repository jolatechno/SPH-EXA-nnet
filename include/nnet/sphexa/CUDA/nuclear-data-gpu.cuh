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
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float, typename Int>
	class DeviceNuclearDataType : public FieldStates<DeviceNuclearDataType<n_species, Float, Int>> {
	public:
		// types
		using RealType = Float;

		/// hydro data
		thrust::device_vector<RealType> c, p, cv, dpdT, u, m, rho, temp, previous_rho;

		/// nuclear abundances (vector of vector)
		thrust::device_vector<util::array<RealType, n_species>> Y;

		/// timesteps
		thrust::device_vector<RealType> dt;

		/// solver buffer
		mutable thrust::device_vector<RealType> buffer;

		/// resize the number of particules
		void resize(size_t size);


		/// base fieldNames (without knowledge of nuclear species names)
		inline static constexpr std::array fieldNames {
			"nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho", "Y",
		};


		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */

	    auto data() {
	    	using FieldType = std::variant<
		    	thrust::device_vector<util::array<RealType, n_species>>*,
		    	thrust::device_vector<RealType>*>;

			return util::array<FieldType, fieldNames.size()>{
				(thrust::device_vector<RealType>*)nullptr, (thrust::device_vector<RealType>*)nullptr,
				&dt, &c, &p, &cv, &u, &dpdT, &m, &temp, &rho, &previous_rho, &Y
			};
	    }
	 };
}