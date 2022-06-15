#pragma once

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include "../nuclear-net.hpp"
#include "nuclear-io.hpp"

#ifdef USE_MPI
	#include <mpi.h>
	#include "mpi/mpi-wrapper.hpp"
#endif

#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
	#include "sph/field_states.hpp"
	#include "cstone/util/util.hpp"
#endif

#include "CUDA/nuclear-data-gpu.hpp"

namespace sphexa::sphnnet {
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float=double>
	struct NuclearDataType : public FieldStates<NuclearDataType<n_species, Float>> {
	public:
		// types
		using RealType = Float;
    	using KeyType  = size_t;

    	DeviceNuclearDataType<n_species, Float> devData;

    	size_t iteration{0};
	    size_t numParticlesGlobal;
	    Float ttot{0.0};
	    //! current and previous (global) time-steps
	    Float minDt, minDt_m1;
	    //! @brief gravitational constant
	    Float g{0.0};

		/// hydro data
		std::vector<Float> c, p, cv, rho, temp, previous_rho; // drho_dt

		/// nuclear abundances (vector of vector)
		std::vector<util::array<Float, n_species>> Y;

		/// timesteps
		std::vector<Float> dt;

		// particle ID and nodeID
		std::vector<int> node_id;
		std::vector<size_t> particle_id;

		/// mpi communicator
#ifdef USE_MPI
    	MPI_Comm comm=MPI_COMM_WORLD;
    	sphexa::mpi::mpi_partition partition;
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

	        // devPtrs.resize(size);
	    }


		/// base fieldNames (without knowledge of nuclear species names)
		inline static constexpr std::array fieldNames {
			"nid", "pid", "dt", "c", "p", "cv", "temp", "rho", "previous_rho", "Y",
		};


		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<
	    		std::vector<util::array<Float, n_species>>*,
	    		std::vector<int>*,
	    		std::vector<size_t>*,
	    		std::vector<Float>*>;
	    	
			util::array<FieldType, 10> ret;

			ret[0] = &node_id;
			ret[1] = &particle_id;
			ret[2] = &dt;
			ret[3] = &c;
			ret[4] = &p;
			ret[5] = &cv;
			ret[6] = &temp;
			ret[7] = &rho;
			ret[8] = &previous_rho;
			ret[9] = &Y;

			return ret;
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