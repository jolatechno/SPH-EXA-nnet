#pragma once

#include <vector>
#include <array>
#include <memory>
#include <variant>

#include <mpi.h>

namespace sphexa {
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

	namespace sphnnet {
		/// nuclear abundances type, that is integrated into NuclearData, or should be integrated into ParticlesData
		template <int n_species, typename Float=double>
		class NuclearAbundances {
		private:
			Float data[n_species];
		public:
			NuclearAbundances(Float x=0) {
				for (int i = 0; i < n_species; ++i)
					data[i] = x;
			}

			size_t size() const {
				return n_species;
			}

			Float &operator[](const int i) {
				return data[i];
			}

			const Float &operator[](const int i) const {
				return data[i];
			}

			template<class Vector>
			NuclearAbundances &operator=(const Vector &other) {
				for (int i = 0; i < n_species; ++i)
					data[i] = other[i];

				return *this;
			}
		};

		/// vector for nuclear IO
		/**
		 * vector for nuclear IO,
		 * allows access by reference of nuclear data
		 * therby allowing "splitting" the vector of vector of nuclear species (Y) multiple vector, without memory overhead.
		 */
		template<int n_species, typename Float=double>
		class nuclear_IO_vector {
		private:
			std::vector<NuclearAbundances<n_species, Float>> *Y;
			int species;

		public:
			// constructors
			nuclear_IO_vector() {};
			nuclear_IO_vector(std::vector<NuclearAbundances<n_species, Float>> &Y_, const int species_) : Y(&Y_), species(species_) {}

			template<typename Int=size_t>
			auto operator[](const Int i) const {
				return (*Y)[i][species];
			}

			template<typename Int=size_t>
			auto at(const Int i) const {
				return Y->at(i)[species];
			}
		};
		

		/// nuclear data class for n_species nuclear network
		/**
		 * TODO
		 */
		template<int n_species, typename Float=double>
		struct NuclearDataType {
		public:
			/// hydro data
			std::vector<Float> rho, T, previous_rho; // drho_dt

			/// nuclear abundances (vector of vector)
			std::vector<NuclearAbundances<n_species, Float>> Y;

			/// timesteps
			std::vector<Float> dt;

			/// mpi communicator
			MPI_Comm comm=MPI_COMM_WORLD;

			/// resize the number of particules
			void resize(const size_t N) {
				rho.resize(N);
				previous_rho.resize(N); //drho_dt.resize(N);
				T.resize(N);

				Y.resize(N);

				dt.resize(N, 1e-12);
			}

			/// base fieldNames (without knowledge of nuclear species names)
			const std::vector<std::string> fieldNames = []() {
				std::vector<std::string> fieldNames_(5 + n_species);

				fieldNames_[0] = "node_id";
		        fieldNames_[1] = "nuclear_particle_id";
		        fieldNames_[2] = "T";
		        fieldNames_[3] = "rho";
		        fieldNames_[4] = "previous_rho";

		        for (int i = 0; i < n_species; ++i)
					fieldNames_[i + 5] = "Y(" + std::to_string(i) + ")";

				return fieldNames_;
			}();

			/// io field to print out node_id for safety
			const_vector<int> node_id;
			/// io field to print out nuclear_particle_id for safety
			iota_vector<size_t> nuclear_particle_id;


			/// nuclear abundances "transpose" vector for IO
			std::array<nuclear_IO_vector<n_species, Float>, n_species> Y_io = [&]{
				std::array<nuclear_IO_vector<n_species, Float>, n_species> Y_io_;
				for (int i = 0; i < n_species; ++i)
					Y_io_[i] = nuclear_IO_vector(Y, i);
				return Y_io_;
			}();

			/*! @brief return a vector of pointers to field vectors
		     *
		     * We implement this by returning an rvalue to prevent having to store pointers and avoid
		     * non-trivial copy/move constructors.
		     */
		    auto data() {
		    	using FieldType = std::variant<
		    		iota_vector<size_t>*,
		    		const_vector<int>*,
		    		nuclear_IO_vector<n_species, Float>*,
		    		std::vector<Float>*>;
				std::array<FieldType, n_species + 5> ret;

				ret[0] = &node_id;
				ret[1] = &nuclear_particle_id;
				ret[2] = &T;
				ret[3] = &rho;
				ret[4] = &previous_rho;

				for (int i = 0; i < n_species; ++i)
					ret[i + 5] = &Y_io[i];

				return ret;
		    }

		    bool isAllocated(int i) const {
		    	/* TODO */
		    	return true;
		    }

		    void setOutputFields(const std::vector<std::string>& outFields) {
		    	int rank;
				MPI_Comm_rank(comm, &rank);

				// initialize node_id and nuclear_particle_id
				node_id             = const_vector<int>(rank);
				nuclear_particle_id = iota_vector<size_t>(0);

		        outputFieldNames = fieldNames;
				outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
	    	}

	    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
	    		int rank;
				MPI_Comm_rank(comm, &rank);

				// initialize node_id and nuclear_particle_id
				node_id             = const_vector<int>(rank);
				nuclear_particle_id = iota_vector<size_t>(0);

				// initialize outputFieldNames with the right names
	    		outputFieldNames.resize(n_species + 5);
		        outputFieldNames[0] = "node_id";
		        outputFieldNames[1] = "nuclear_particle_id";
		        outputFieldNames[2] = "T";
		        outputFieldNames[3] = "rho";
		        outputFieldNames[4] = "previous_rho";

		        for (int i = 0; i < n_species; ++i)
					outputFieldNames[i + 5] = "Y(" + species_names[i] + ")";

		        outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
	    	}

			//! @brief particle fields selected for file output
			std::vector<int>         outputFieldIndices;
			std::vector<std::string> outputFieldNames;
		};
	}
}