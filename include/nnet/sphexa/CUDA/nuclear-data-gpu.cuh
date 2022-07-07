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
#ifdef COMPILE_DEVICE
	#include <thrust/device_vector.h>
#endif

namespace sphexa::sphnnet {
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<class AccType, size_t n_species, typename Float>
	class DeviceNuclearDataType : public FieldStates<DeviceNuclearDataType<AccType, n_species, Float>> {
	public:
		// types
		using RealType = Float;
    	using AcceleratorType = AccType;

#ifdef COMPILE_DEVICE
		/// hydro data
		thrust::device_vector<Float> c, p, cv, dpdT, u, m, rho, temp, previous_rho;

		/// nuclear abundances (vector of vector)
		thrust::device_vector<util::array<Float, n_species>> Y;

		/// timesteps
		thrust::device_vector<Float> dt;

		/// solver buffer
		mutable thrust::device_vector<Float> buffer;
#endif

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
#ifdef COMPILE_DEVICE
	    	using FieldType = std::variant<
		    	thrust::device_vector<util::array<Float, n_species>>*,
		    	thrust::device_vector<Float>*>;

		    if constexpr (HaveGpu<AcceleratorType>{}) {
				return util::array<FieldType, fieldNames.size()>{
					(thrust::device_vector<Float>*)nullptr, (thrust::device_vector<Float>*)nullptr,
					&dt, &c, &p, &cv, &u, &dpdT, &m, &temp, &rho, &previous_rho, &Y
				};
			} else
				return util::array<FieldType, 0>{};
#else
			return util::array<std::variant<std::vector<int>*>, 0>{};
#endif
	    }
	 };
}