#pragma once

#include <vector>
#include <array>
#include <memory>
#include <variant>

#ifdef USE_MPI
	#include <mpi.h>
	#include "mpi/mpi-wrapper.hpp"
#endif

#include "util/data_util.hpp"
#ifndef NOT_FROM_SPHEXA
	#include "sph/data_util.hpp"
#endif

namespace sphexa::sphnnet {
	/// nuclear data class for n_species nuclear network
	/**
	 * TODO
	 */
	template<int n_species, typename Float=double>
	struct NuclearDataType {
	public:
		/// hydro data
		std::vector<Float> c, p, u, rho, temp, previous_rho; // drho_dt

		/// nuclear abundances (vector of vector)
		std::vector<util::array<Float, n_species>> Y;

		/// timesteps
		std::vector<Float> dt;

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
			u.resize(N);

			Y.resize(N);

			dt.resize(N, nnet::constants::initial_dt);
		}

		/// base fieldNames (without knowledge of nuclear species names)
		const std::vector<std::string> fieldNames = []() {
			std::vector<std::string> fieldNames_(n_species + 10);
			fieldNames_[0] = "nid";
	        fieldNames_[1] = "pid";
	        fieldNames_[2] = "dt";
	        fieldNames_[3] = "c";
	        fieldNames_[4] = "p";
	        fieldNames_[5] = "u";
	        fieldNames_[6] = "temp";
	        fieldNames_[7] = "rho";
	        fieldNames_[8] = "previous_rho";

	        for (int i = 0; i < n_species; ++i)
				fieldNames_[i + 9] = "Y(" + std::to_string(i) + ")";

			return fieldNames_;
		}();

		/// io field to print out node_id for safety
		const_vector<int> node_id;
		/// io field to print out nuclear_particle_id for safety
		iota_vector<size_t> nuclear_particle_id;


		/// nuclear abundances "transpose" vector for IO
		util::array<nuclear_IO_vector<n_species, Float>, n_species> Y_io = [&]{
			util::array<nuclear_IO_vector<n_species, Float>, n_species> Y_io_;
			for (int i = 0; i < n_species; ++i)
				Y_io_[i] = nuclear_IO_vector<n_species, Float>(Y, i);
			return Y_io_;
		}();

		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<
	    		/*iota_vector<size_t>*,
	    		const_vector<int>*,
	    		nuclear_IO_vector<n_species, Float>*,*/
	    		std::vector<Float>*,
	    		std::vector<uint8_t/*bool*/>*>;
	    	
			util::array<FieldType, n_species + 9> ret;

			ret[0] = (std::vector<Float>*)nullptr; //&node_id;
			ret[1] = (std::vector<Float>*)nullptr; //&nuclear_particle_id;
			ret[2] = &dt;
			ret[3] = &c;
			ret[4] = &p;
			ret[5] = &u;
			ret[6] = &temp;
			ret[7] = &rho;
			ret[8] = &previous_rho;

			for (int i = 0; i < n_species; ++i)
				ret[i + 9] = (std::vector<Float>*)nullptr; //&Y_io[i];

			return ret;
	    }

	    bool isAllocated(int i) const {
	    	if (i <= 1 || i >= 9)
	    		return false;
	    	return true;
	    }

	    void setOutputFields(const std::vector<std::string>& outFields) {
	    	int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif

			// initialize node_id and nuclear_particle_id
			node_id             = const_vector<int>(rank);
			nuclear_particle_id = iota_vector<size_t>(0);

	        outputFieldNames = fieldNames;
			outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
    	}

    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
	    	int rank = 0;
#ifdef USE_MPI
			MPI_Comm_rank(comm, &rank);
#endif

			// initialize node_id and nuclear_particle_id
			node_id             = const_vector<int>(rank);
			nuclear_particle_id = iota_vector<size_t>(0);

			// initialize outputFieldNames with the right names
    		outputFieldNames.resize(n_species + 9);
	        outputFieldNames[0] = "nid";
	        outputFieldNames[1] = "pid";
	        outputFieldNames[2] = "dt";
	        outputFieldNames[3] = "c";
	        outputFieldNames[4] = "c";
	        outputFieldNames[5] = "p";
	        outputFieldNames[6] = "temp";
	        outputFieldNames[7] = "rho";
	        outputFieldNames[8] = "previous_rho";

	        for (int i = 0; i < n_species; ++i)
				outputFieldNames[i + 9] = "Y(" + species_names[i] + ")";

	        outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
    	}

		//! @brief particle fields selected for file output
		std::vector<int>         outputFieldIndices;
		std::vector<std::string> outputFieldNames;
	};
}