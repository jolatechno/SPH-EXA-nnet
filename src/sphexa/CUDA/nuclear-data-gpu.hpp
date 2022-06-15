#pragma once

#include <vector>
#include <array>
#include <memory>
#include <variant>

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
	#include "sph/field_states.hpp"
	#include "cstone/util/util.hpp"
#endif

#ifdef USE_CUDA
	#include <thrust/device_vector.h>
#endif

namespace sphexa::sphnnet {
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float=double>
	struct DeviceNuclearDataType : public FieldStates<DeviceNuclearDataType<n_species, Float>> {
	public:
		// types
		using RealType = Float;

#ifdef USE_CUDA
		/// hydro data
		thrust::device_vector<Float> c, p, cv, rho, temp, previous_rho; // drho_dt

		/// nuclear abundances (vector of vector)
		thrust::device_vector<util::array<Float, n_species>> Y;

		/// timesteps
		thrust::device_vector<Float> dt;
#endif

		/// resize the number of particules
		void resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) { 
	                	size_t previous_size = arg->size();
	                	reallocate(*arg, size, growthRate); 
	                }, data_[i]);
	            }
	        }

	        // devPtrs.resize(size);
	    }


		/// base fieldNames (without knowledge of nuclear species names)
		inline static constexpr std::array fieldNames {
			"dt", "c", "p", "cv", "temp", "rho", "previous_rho", "Y",
		};


		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */

	    auto data() {
#ifdef USE_CUDA

	    	using FieldType = std::variant<
	    		thrust::device_vector<util::array<Float, n_species>>*,
	    		thrust::device_vector<Float>*>;
	    	
			util::array<FieldType, 8> ret;

			ret[0] = &dt;
			ret[1] = &c;
			ret[2] = &p;
			ret[3] = &cv;
			ret[4] = &temp;
			ret[5] = &rho;
			ret[6] = &previous_rho;
			ret[7] = &Y;

			return ret;
#else
			return util::array<void*, 1>{nullptr};
#endif
	    }

	    DeviceNuclearDataType() {
#ifdef USE_CUDA
			/* TODO */
#endif
	    }

	    ~DeviceNuclearDataType() {
#ifdef USE_CUDA
			/* TODO */
#endif
	    }

	    template<class DataType>
		void transferToDevice(DataType& d, const std::vector<std::string>& fields) {
#ifdef USE_CUDA
			auto hostData   = d.data();
    		auto deviceData = d.devData.data();

			/* TODO */
#endif
		}

		template<class DataType>
		void transferToHost(DataType& d, const std::vector<std::string>& fields) {
#ifdef USE_CUDA
			auto hostData   = d.data();
    		auto deviceData = d.devData.data();

			/* TODO */
#endif
		}

	    void setOutputFields(const std::vector<std::string>& outFields) {
	        outputFieldNames = outFields;
	        outputFieldIndices = sphexa::fieldStringsToInt(fieldNames, outFields);
    	}

		//! @brief particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}