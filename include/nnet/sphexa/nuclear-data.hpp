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
 * @brief Definition of the main data class for nuclear networks, similar to the particuleData class in SPH-EXA.
 *
 * @author Joseph Touzet <joseph.touzet@ens-paris-saclay.fr>
 */


#pragma once

#include "../CUDA/cuda.inl"

#include <vector>
#include <array>
#include <memory>
#include <variant>

#if COMPILE_DEVICE
	#include "CUDA/nuclear-data-gpu.cuh"
#endif

#ifdef USE_MPI
	#include <mpi.h>
#endif
#include "mpi/mpi-wrapper.hpp"

#include "../nuclear-net.hpp"

#include "sph/data_util.hpp"
#include "sph/field_states.hpp"
#include "sph/traits.hpp"

#include "cstone/util/util.hpp"


namespace sphexa::sphnnet {
	template<size_t N, typename T, typename I>
	class DeviceNuclearDataType;

	/*! @brief nuclear data class for n_species nuclear network */
	template<size_t n_species, typename Float, typename Int, class AccType>
	struct NuclearDataType : public FieldStates<NuclearDataType<n_species, Float, Int, AccType>> {
	public:
    	template<class... Args>
    	using DeviceNuclearData_t = DeviceNuclearDataType<n_species, Args...>;

		// types
		using RealType = Float;
    	using KeyType  = Int;
    	using AcceleratorType = AccType;
		using DeviceData_t = typename sphexa::detail::AccelSwitchType<AcceleratorType, sphexa::DeviceDataFacade, DeviceNuclearData_t>::template type<Float, Int>;

    	DeviceData_t devData;

    	size_t iteration{0};
	    size_t numParticlesGlobal;
	    RealType ttot{0.0};
	    //! current and previous (global) time-steps
	    RealType minDt, minDt_m1;
	    //! @brief gravitational constant
	    RealType g{0.0};

		//! hydro data
		std::vector<RealType> c, p, cv, u, dpdT, m, rho, temp, previous_rho; // drho_dt

		//! nuclear abundances (vector of vector)
		util::array<std::vector<RealType>, n_species> Y;

		//! timesteps
		std::vector<RealType> dt;

		//! particle ID
		std::vector<int> node_id;
		//! node ID
		std::vector<KeyType> particle_id;

		//! mpi communicator
#ifdef USE_MPI
    	MPI_Comm comm=MPI_COMM_WORLD;
    	sphexa::mpi::mpi_partition partition;
#endif

		/*! @brief resize the number of particules
		 * 
		 * @param size  number of particle to be hold by the class
		 */
		void resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

			if constexpr (HaveGpu<AcceleratorType>{})
	        	devData.resize(size);

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) { 
	                	using T = decltype((*arg)[0]);
	                	size_t previous_size = arg->size();

                		// reallocate
                		reallocate(*arg, size, growthRate);

                		// fill node_id
		                if ((void*)arg == (void*)(&node_id)) {
	        				int rank = 0;
#ifdef USE_MPI
							MPI_Comm_rank(comm, &rank);
#endif
							std::fill(node_id.begin() + previous_size, node_id.end(), rank);
		                }

		                // fill particle ID
		                else if ((void*)arg == (void*)(&particle_id)) {
		                	std::iota(particle_id.begin() + previous_size, particle_id.end(), previous_size);
		                }

		                // fill rho or previous_rho
		                else if ((void*)arg == (void*)(&rho) || (void*)arg == (void*)(&previous_rho)) {
		                	std::fill(arg->begin() + previous_size, arg->end(), 0.);
		                }

		    			// fill dt
		                else if ((void*)arg == (void*)(&dt)) {
		                	std::fill(dt.begin() + previous_size, dt.end(), nnet::constants::initial_dt);
		                }
	                }, data_[i]);
	            }
	        }
	    }


		//! base hydro fieldNames (withoutnuclear species names)
		inline static constexpr std::array baseFieldNames {
			"nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "previous_rho",
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


		//! base fieldNames (contains individual species). Initialized by setOutputFieldsNames
		std::array<std::string, fieldNames.size()> outputableFieldNames = [] {
			std::array<std::string, fieldNames.size()> outputableFieldNames;
			for (int i = 0; i < baseFieldNames.size(); ++i)
				outputableFieldNames[i] = baseFieldNames[i];
			for (int i = 0; i < n_species; ++i) 
				outputableFieldNames[baseFieldNames.size() + i] = "Y(" + std::to_string(i) + ")";
			return outputableFieldNames;
		}();


		/*! @brief access and modifies "fieldNames" to account for nuclear species names
	     *
	     * @param species_names  vector of species name
		 */
		void setOutputFieldsNames(const std::vector<std::string> &species_names) {
			for (int i = 0; i < n_species; ++i) 
				outputableFieldNames[baseFieldNames.size() + i] = "Y(" + species_names[i] + ")";
		}


		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<
	    		std::vector<int>*,
	    		std::vector<KeyType>*,
	    		std::vector<RealType>*>;
	    	
			util::array<FieldType, fieldNames.size()> data;

			data[0]  = &node_id;
			data[1]  = &particle_id;
			data[2]  = &dt;
			data[3]  = &c;
			data[4]  = &p;
			data[5]  = &cv;
			data[6]  = &u;
			data[7]  = &dpdT;
			data[8]  = &m;
			data[9]  = &temp;
			data[10] = &rho;
			data[11] = &previous_rho;

			for (int i = 0; i < n_species; ++i) 
				data[baseFieldNames.size() + i] = &Y[i];

			return data;
	    }

	    /*! @brief sets the field to be outputed
	     * 
	     * @param outFields  vector of the names of fields to be outputed (including abundances names "Y(i)" for the ith species)
	     */
	    void setOutputFields(const std::vector<std::string>& outFields) {
	        outputFieldNames = outFields;
	        outputFieldIndices = sphexa::fieldStringsToInt(outputableFieldNames, outFields);
    	}

		// particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}