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

#include "../../util/CUDA/cuda.inl"

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

#include "sph/traits.hpp"

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/concatenate.hpp"

#include "cstone/util/reallocate.hpp"


namespace sphexa::sphnnet {
	template<typename T, typename I>
	class DeviceNuclearDataType;

	/*! @brief nuclear data class for nuclear network */
	template<typename Float, typename Int, class AccType>
	struct NuclearDataType : public cstone::FieldStates<NuclearDataType<Float, Int, AccType>> {
	public:
		//! maximum number of nuclear species
		static const int maxNumSpecies = 100;
		//! actual number of nuclear species
		int numSpecies = 0;

    	template<class... Args>
    	using DeviceNuclearData_t = DeviceNuclearDataType<Args...>;

		// types
		using RealType        = Float;
    	using KeyType         = Int;
	    using Tmass           = float;
	    using XM1Type         = float;
    	using AcceleratorType = AccType;
		using DeviceData_t    = typename sphexa::detail::AccelSwitchType<AcceleratorType, sphexa::DeviceDataFacade, DeviceNuclearData_t>::template type<Float, Int>;

    	DeviceData_t devData;

    	size_t iteration{0};
	    size_t numParticlesGlobal;
	    RealType ttot{0.0};
	    //! current and previous (global) time-steps
	    RealType minDt, minDt_m1;
	    //! @brief gravitational constant
	    RealType g{0.0};

		// could replace rho_m1 -> drho_dt
		//!  @brief hydro data
		std::vector<RealType> c, p, cv, u, dpdT, rho, temp, rho_m1;
		//!  @brief particle mass (lower precision)
		std::vector<Tmass> m;

		//! nuclear abundances (vector of vector)
		util::array<std::vector<RealType>, maxNumSpecies> Y;

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

		                // fill rho or rho_m1
		                else if ((void*)arg == (void*)(&rho) || (void*)arg == (void*)(&rho_m1)) {
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
	    		std::vector<int>*,
	    		std::vector<unsigned>*,
	    		std::vector<uint64_t>*,
	    		std::vector<float>*,
	    		std::vector<double>*>;

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

	    /*! @brief sets the field to be outputed
	     * 
	     * @param outFields  vector of the names of fields to be outputed (including abundances names "Y(i)" for the ith species)
	     */
	    void setOutputFields(const std::vector<std::string>& outFields) {
	        outputFieldNames = outFields;
	        outputFieldIndices = cstone::fieldStringsToInt(outFields, fieldNames);
    	}

		// particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}