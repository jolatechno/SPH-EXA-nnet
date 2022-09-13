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

#include "nuclear-io.hpp"

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

	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
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

		/// hydro data
		std::vector<RealType> c, p, cv, u, dpdT, m, rho, temp, previous_rho; // drho_dt

		/// nuclear abundances (vector of vector)
		std::vector<util::array<RealType, n_species>> Y;

		/// timesteps
		std::vector<RealType> dt;

		// particle ID and nodeID
		std::vector<int> node_id;
		std::vector<KeyType> particle_id;

		/// mpi communicator
#ifdef USE_MPI
    	MPI_Comm comm=MPI_COMM_WORLD;
    	sphexa::mpi::mpi_partition partition;
#endif

		/// resize the number of particules
		void resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

			if constexpr (HaveGpu<AcceleratorType>{})
	        	devData.resize(size);

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) { 
	                	size_t previous_size = arg->size();
	                	reallocate(*arg, size, growthRate); 

	                	using T = decltype((*arg)[0]);
	                	if constexpr (std::is_convertible<T, int>::value) {
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
			            }
	                }, data_[i]);
	            }
	        }
	    }


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
	    		std::vector<util::array<RealType, n_species>>*,
	    		std::vector<int>*,
	    		std::vector<KeyType>*,
	    		std::vector<RealType>*>;
	    	
			return util::array<FieldType, fieldNames.size()>{
				&node_id, &particle_id, &dt, &c, &p, &cv, &u, &dpdT, &m, &temp, &rho, &previous_rho, &Y
			};
	    }

	    void setOutputFields(const std::vector<std::string>& outFields) {
	    	int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif

	        outputFieldNames = outFields;

			// separate nuclear fields from hydro fields
			io::setOutputFieldsNames(n_species);
			std::vector<std::string> hydroOutFields = io::setOutputFields(outFields);

			// add output field
			bool print_nuclear = hydroOutFields.size() < outFields.size();
	        if (print_nuclear && !std::count(hydroOutFields.begin(), hydroOutFields.end(), "Y"))
	        	hydroOutFields.push_back("Y");

	        outputFieldIndices = sphexa::fieldStringsToInt(fieldNames, hydroOutFields);
    	}

    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
    		outputFieldNames = outFields;

			// separate nuclear fields from hydro fields
			io::setOutputFieldsNames(species_names);
	        std::vector<std::string> hydroOutFields = io::setOutputFields(outFields);

	        // add output field
			bool print_nuclear = hydroOutFields.size() < outFields.size();
	        if (print_nuclear && !std::count(hydroOutFields.begin(), hydroOutFields.end(), "Y"))
	        	hydroOutFields.push_back("Y");

	        outputFieldIndices = sphexa::fieldStringsToInt(fieldNames, hydroOutFields);
    	}

		//! @brief particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}