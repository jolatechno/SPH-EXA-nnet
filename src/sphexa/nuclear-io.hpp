#pragma once

#include <variant>

#include "mpi-wrapper.hpp"
#include "nuclear-data.hpp"

namespace sphexa::sphnnet {
	/// vector for nuclear IO
	/**
	 * vector for nuclear IO,
	 * allows access by reference of nuclear data
	 * therby allowing "splitting" the vector of vector of nuclear species (Y) multiple vector, without memory overhead.
	 */
	template <int n_species, typename Float=double>
	class nuclear_IO_vector {
	private:
		const std::vector<NuclearAbundances<n_species, Float>> *Y;
		const int species;

	public:
		// constructor
		nuclear_IO_vector(const std::vector<NuclearAbundances<n_species, Float>> &Y_, const int species_) : Y(&Y_), species(species_) {
			if (species >= n_species) {
				std::cerr << "species out of bound in nuclear_IO_vector!\n";
				throw;
			}
		}

		template<typename Int=size_t>
		Float operator[](const Int i) const {
			return (*Y)[i][species];
		}

		template<typename Int=size_t>
		Float at(const Int i) const {
			return Y->at(i)[species];
		}
	};

	/// simple "iota" vector:
	/**
	 * "virtual" vector that respects the equation vect[i] = i + i0
	 */
	template<typename T>
	class iota_vector {
	private:
		size_t i0;

	public:
		// constructor
		template<typename Int=size_t>
		iota_vector(Int const i0_=0) : i0(i0_) {}

		template<typename Int=size_t>
		T operator[](const Int i) const {
			return i + i0;
		}
	};

	/// simple "constant" vector:
	/**
	 * "virtual" vector that respects the equation vect[i] = v0
	 */
	template<typename T>
	class const_vector {
	private:
		size_t v0;

	public:
		// constructor
		template<typename Int=size_t>
		const_vector(T const v0_=0) : v0(v0_) {}

		template<typename Int=size_t>
		T operator[](const Int i) const {
			return v0;
		}
	};


	/// class for IO
	/**
	 * contains all the relevent data fields to save nuclear abundances
	 */
	template<int n_species, typename Float=double>
	class NuclearIoDataSet {
	private:
		const_vector<int> node_id;
		iota_vector<size_t> nuclear_particle_id;
		std::vector<nuclear_IO_vector<n_species, Float>> Y = {};
		const std::vector<Float> *T, *rho, *previous_rho;

	public:
		NuclearIoDataSet(const NuclearDataType<n_species, Float> &n) : T(&n.T), rho(&n.rho), previous_rho(&n.previous_rho) {
			int rank;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);

			// initialize
			node_id = const_vector<int>(rank);
			nuclear_particle_id = iota_vector<size_t>(0);

			for (int species = 0; species < n_species; ++species)
				Y.push_back(nuclear_IO_vector(n.Y, species));
		}

		/*! @brief return a vector of pointers to field vectors
	     *
	     * We implement this by returning an rvalue to prevent having to store pointers and avoid
	     * non-trivial copy/move constructors.
	     */
	    auto data() {
	    	using FieldType = std::variant<iota_vector<size_t>*, const_vector<int>*, const nuclear_IO_vector<n_species, Float>*, const std::vector<Float>*>;
			std::array<FieldType, 5 + n_species> ret;

			ret[0] = &node_id;
			ret[1] = &nuclear_particle_id;
			ret[2] = T;
			ret[3] = rho;
			ret[4] = previous_rho;

			for (int i = 0; i < n_species; ++i)
				ret[i + 5] = &Y[i];

			return ret;
	    }

	    void setOutputFields(const std::vector<std::string>& outFields) {
	        outputFieldNames[0] = "node_id";
	        outputFieldNames[1] = "nuclear_particle_id";
	        outputFieldNames[2] = "T";
	        outputFieldNames[3] = "rho";
	        outputFieldNames[4] = "previous_rho";

	        for (int i = 0; i < n_species; ++i)
				outputFieldNames[i + 5] = "Y(" + std::to_string(i) + ")";

			// outputFieldIndices = fieldStringsToInt(fieldNames, outFields);
    	}

    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
	        outputFieldNames[0] = "node_id";
	        outputFieldNames[1] = "nuclear_particle_id";
	        outputFieldNames[2] = "T";
	        outputFieldNames[3] = "rho";
	        outputFieldNames[4] = "previous_rho";

	        for (int i = 0; i < n_species; ++i)
				outputFieldNames[i + 5] = "Y(" + species_names[i] + ")";

	        // outputFieldIndices = fieldStringsToInt(fieldNames, outFields);
    	}

		//! @brief particle fields selected for file output
		std::array<int, 5 + n_species>         outputFieldIndices;
		std::array<std::string, 5 + n_species> outputFieldNames;
	};
}