#pragma once

#include <variant>
#include <memory>

#include "mpi-wrapper.hpp"
#include "nuclear-data.hpp"

namespace sphexa {
	namespace sphnnet {
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
			std::shared_ptr<const Float[]> T, rho, previous_rho;

		public:
			MPI_Comm comm;
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

			NuclearIoDataSet(const NuclearDataType<n_species, Float> &n, MPI_Comm comm_=MPI_COMM_WORLD) : 
				T(           n.T.data(),            /*void deleter:*/[](const Float*) {}),
				rho(         n.rho.data(),          /*void deleter:*/[](const Float*) {}),
				previous_rho(n.previous_rho.data(), /*void deleter:*/[](const Float*) {}),
				comm(comm_)
			{
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
		    	using FieldType = std::variant<iota_vector<size_t>*, const_vector<int>*, const nuclear_IO_vector<n_species, Float>*, std::shared_ptr<const Float[]>*>;
				std::array<FieldType, n_species + 5> ret;

				ret[0] = &node_id;
				ret[1] = &nuclear_particle_id;
				ret[2] = &T;
				ret[3] = &rho;
				ret[4] = &previous_rho;

				for (int i = 0; i < n_species; ++i)
					ret[i + 5] = &Y[i];

				return ret;
		    }

		    bool isAllocated(int i) const {
		    	/* TODO */
		    	return true;
		    }

		    void setOutputFields(const std::vector<std::string>& outFields) {
		        outputFieldNames = fieldNames;
				outputFieldIndices = sphexa::fieldStringsToInt(outputFieldNames, outFields);
	    	}

	    	void setOutputFields(const std::vector<std::string>& outFields, const std::vector<std::string> &species_names) {
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
	
	//! @brief extract a vector of reference to nuclear particle fields for file output
	template<int n_species, typename Float=double>
	auto getOutputArrays(sphexa::sphnnet::NuclearIoDataSet<n_species, Float>& dataset) {
	    auto fieldPointers = dataset.data();
	    using FieldType = std::variant<sphexa::sphnnet::iota_vector<size_t>, sphexa::sphnnet::const_vector<int>, sphexa::sphnnet::nuclear_IO_vector<n_species, Float>, std::shared_ptr<const Float[]>>;

	    std::vector<FieldType> outputFields;
	    outputFields.reserve(dataset.outputFieldIndices.size());

	    for (int i : dataset.outputFieldIndices)
	    {
	        if (!dataset.isAllocated(i))
	        {
	            throw std::runtime_error("Cannot output field " + std::string(dataset.fieldNames[i]) +
	                                     ", because it is not active.");
	        }
	        std::visit([&outputFields](auto& arg) { outputFields.emplace_back(*arg); }, fieldPointers[i]);
	    }
	    return outputFields;
	}

	namespace fileutils {
		/*! @brief read input data from an ASCII file
		 *
		 * @tparam T         an elementary type or a std::variant thereof
		 * @param  path      the input file to read from
		 * @param  numLines  number of lines/elements per field to read
		 * @param  fields    the data containers to read into
		 *
		 *  Each data container will get one column of the input file.
		 *  The number of rows to read is determined by the data container size.
		 */
		template<class... itType, class... Separators>
		void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, const std::vector<std::variant<itType...>>& fields, Separators&&... separators) {
		    std::ios_base::openmode mode;
		    if (append) { mode = std::ofstream::app; }
		    else { mode = std::ofstream::out; }

		    std::ofstream dumpFile(path, mode);

		    if (dumpFile.is_open())
		    {
		        for (size_t i = firstIndex; i < lastIndex; ++i)
		        {
		            for (auto field : fields)
		            {
		                [[maybe_unused]] std::initializer_list<int> list{(dumpFile << separators, 0)...};
		                std::visit([&dumpFile, i](auto& arg) { dumpFile << arg[i]; }, field);
		            }
		            dumpFile << std::endl;
		        }
		    }
		    else { throw std::runtime_error("Can't open file at path: " + path); }

		    dumpFile.close();
		}
	}
}