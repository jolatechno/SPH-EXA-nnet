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

// can't compile CUDA GPU data for now !
#if COMPILE_DEVICE && 0
	#include "CUDA/nuclear-data-gpu.cuh"
#endif

#ifdef USE_MPI
	#include <mpi.h>
#endif
#include "mpi/mpi-wrapper.hpp"

#include "../nuclear-net.hpp"

#include "cstone/tree/accel_switch.hpp"

#include "cstone/fields/field_states.hpp"
#include "cstone/fields/data_util.hpp"
#include "cstone/fields/enumerate.hpp"
#include "cstone/fields/concatenate.hpp"

#include "cstone/util/reallocate.hpp"


namespace sphexa::sphnnet {
	/*! @brief nuclear data device facade */
	template<typename T, typename I>
	class DeviceNuclearDataFacade {
	public:
		void resize(size_t) {}

        template<class... Ts>
        void setConserved(Ts...) {}

        template<class... Ts>
        void setDependent(Ts...) {}

        template<class... Ts>
        void release(Ts...) {}

        template<class... Ts>
        void acquire(Ts...) {}

        inline static constexpr std::array fieldNames{0};
	};

	/*! @brief Forward declaration of DeviceNuclearDataType, defined to DeviceNuclearDataFacade for now ! */
	template<typename T, typename I>
	class DeviceNuclearDataType : public DeviceNuclearDataFacade<T, I> {};
	// class DeviceNuclearDataType;

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

	    template<class ValueType>
	    using FieldVector = std::vector<ValueType, std::allocator<ValueType>>;

		// types
		using RealType        = Float;
    	using KeyType         = Int;
	    using Tmass           = float;
	    using XM1Type         = float;
    	using AcceleratorType = AccType;
		using DeviceData_t    = typename cstone::AccelSwitchType<AcceleratorType, DeviceNuclearDataFacade, DeviceNuclearData_t>::template type<Float, Int>;

    	DeviceData_t devData;

    	size_t iteration{0};
	    size_t numParticlesGlobal;
	    RealType ttot{0.0};
	    //! current and previous (global) time-steps
	    RealType minDt, minDt_m1;
	    //! @brief gravitational constant
	    RealType g{0.0};

		//!  @brief hydro data
		FieldVector<RealType> c;                             // speed of sound
		FieldVector<RealType> p;                             // pressure
		FieldVector<RealType> cv;                            // cv
		FieldVector<RealType> u;                             // internal energy
		FieldVector<RealType> dpdT;                          // dP/dT
		FieldVector<RealType> rho;                           // density
		FieldVector<RealType> temp;                          // temperature
		FieldVector<RealType> rho_m1;                        // previous density
		FieldVector<Tmass> m;                                // mass
		FieldVector<RealType> dt;                            // timesteps
		util::array<FieldVector<RealType>, maxNumSpecies> Y; // vector of nuclear abundances
		FieldVector<int> node_id;                            // node ids
		FieldVector<KeyType> particle_id;                    // particle id

		//! mpi communicator
#ifdef USE_MPI
    	MPI_Comm comm=MPI_COMM_WORLD;
    	sphexa::mpi::mpi_partition partition;
#endif

		//! base hydro fieldNames (every nuclear species is named "Yn")
		inline static constexpr auto fieldNames = concat(enumerateFieldNames<"Y", maxNumSpecies>(), std::array<const char*, 12>{
			"nid", "pid", "dt", "c", "p", "cv", "u", "dpdT", "m", "temp", "rho", "rho_m1",
		});

		/*! @brief return a tuple of field references
	     *
	     * Note: this needs to be in the same order as listed in fieldNames
	     */
		auto dataTuple() { 
			return std::tuple_cat(
				dataTuple_helper(std::make_index_sequence<maxNumSpecies>{}),
				std::tie(node_id, particle_id, dt, c, p, cv, u, dpdT, m, temp, rho, rho_m1)
			); 
		}

		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	        using FieldType =
	            std::variant<FieldVector<float>*, FieldVector<double>*, FieldVector<unsigned>*, FieldVector<int>*, FieldVector<uint64_t>*>;

	        return std::apply([](auto&... fields) { return std::array<FieldType, sizeof...(fields)>{&fields...}; },
	                          dataTuple());
	    }

	    /*! @brief sets the field to be outputed
	     * 
	     * @param outFields  vector of the names of fields to be outputed (including abundances names "Y(i)" for the ith species)
	     */
	    void setOutputFields(const std::vector<std::string>& outFields) {
	        outputFieldNames = outFields;
	        outputFieldIndices = cstone::fieldStringsToInt(outFields, fieldNames);
    	}

		/*! @brief resize the number of particules
		 * 
		 * @param size  number of particle to be hold by the class
		 */
		void resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

			if constexpr (cstone::HaveGpu<AcceleratorType>{})
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

		// particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;

private:
	    template<size_t... Is>
	    auto dataTuple_helper(std::index_sequence<Is...>) {
	        return std::tie(Y[Is]...);
	    }
	};
}