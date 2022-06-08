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
#endif

namespace sphexa::sphnnet {
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<size_t n_species, typename Float=double>
	struct NuclearDataType {
	public:
		// types
		using RealType = Float;
    	using KeyType  = size_t;

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
		void resize(const size_t N) {
			rho.resize(N, 0.);
			previous_rho.resize(N, 0.);
			temp.resize(N);
			c.resize(N);
			p.resize(N);
			cv.resize(N);

			Y.resize(N);

			dt.resize(N, nnet::constants::initial_dt);

			int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif
			node_id.resize(N);
			std::fill(node_id.begin(), node_id.end(), rank);

			particle_id.resize(N);
			std::iota(particle_id.begin(), particle_id.end(), 0);
		}

		/// base fieldNames (without knowledge of nuclear species names)
		const std::vector<std::string> fieldNames = []() {
			std::vector<std::string> fieldNames_(9);
			fieldNames_[0] = "nid";
	        fieldNames_[1] = "pid";
	        fieldNames_[2] = "dt";
	        fieldNames_[3] = "c";
	        fieldNames_[4] = "p";
	        fieldNames_[5] = "cv";
	        fieldNames_[6] = "temp";
	        fieldNames_[7] = "rho";
	        fieldNames_[8] = "previous_rho";

			return fieldNames_;
		}();


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

	    bool isAllocated(int i) const {
	    	return true;
	    }

	    void setOutputFields(const std::vector<std::string>& outFields) {
	    	int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif

	        outputFieldNames = fieldNames;

			// separate nuclear fields from hydro fields
			io::setOutputFieldsNames(n_species);
			std::vector<std::string> hydroOutFields = io::setOutputFields(outFields);
			bool print_nuclear = hydroOutFields.size() < outFields.size();

	        outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, hydroOutFields);
	        if (print_nuclear)
	        	outputFieldIndices.push_back(9);
    	}

    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
	    	int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif

    		outputFieldNames = fieldNames;

			// separate nuclear fields from hydro fields
			io::setOutputFieldsNames(species_names);
	        std::vector<std::string> hydroOutFields = io::setOutputFields(outFields);
			bool print_nuclear = hydroOutFields.size() < outFields.size();

	        outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, hydroOutFields);
	        if (print_nuclear)
	        	outputFieldIndices.push_back(9);
    	}

		//! @brief particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}