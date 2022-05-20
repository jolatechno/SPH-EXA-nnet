#pragma once

#include <variant>

#include "mpi-wrapper.hpp"
#include "nuclear-data.hpp"

#ifndef IMPORTED_FROM_SPHEXA
namespace sphexa {
	/*! @brief look up indices of field names
	 *
	 * @tparam     Array
	 * @param[in]  allNames     array of strings with names of all fields
	 * @param[in]  subsetNames  array of strings of field names to look up in @p allNames
	 * @return                  the indices of @p subsetNames in @p allNames
	 */
	template<class Array>
	std::vector<int> fieldStringsToInt(const Array& allNames, const std::vector<std::string>& subsetNames)
	{
	    std::vector<int> subsetIndices;
	    subsetIndices.reserve(subsetNames.size());
	    for (const auto& field : subsetNames)
	    {
	        auto it = std::find(allNames.begin(), allNames.end(), field);
	        if (it == allNames.end()) { throw std::runtime_error("Field " + field + " does not exist\n"); }

	        size_t fieldIndex = it - allNames.begin();
	        subsetIndices.push_back(fieldIndex);
	    }
	    return subsetIndices;
	}
}
#endif

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

	/// simple "reference" vector:
	/**
	 * "virtual" vector that access another vector. Allows pass-by-value
	 */
	template<typename T>
	class ref_vector {
	private:
		const std::vector<T> *ref;

	public:
		// constructor
		template<typename Int=size_t>
		ref_vector(const std::vector<T> &ref_) : ref(&ref_) {}

		template<typename Int=size_t>
		T operator[](const Int i) const {
			return (*ref)[i];
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
		const ref_vector<Float> T, rho, previous_rho;

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
			T(           ref_vector(n.T)),
			rho(         ref_vector(n.rho)),
			previous_rho(ref_vector(n.previous_rho)),
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
	    	using FieldType = std::variant<iota_vector<size_t>*, const_vector<int>*, const nuclear_IO_vector<n_species, Float>*, const ref_vector<Float>*>;
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

namespace sphexa {
	template<int n_species, typename Float=double>
	auto getOutputArrays(sphexa::sphnnet::NuclearIoDataSet<n_species, Float>& dataset) {
	    auto fieldPointers = dataset.data();
	    using FieldType = std::variant<sphexa::sphnnet::iota_vector<size_t>, sphexa::sphnnet::const_vector<int>, sphexa::sphnnet::nuclear_IO_vector<n_species, Float>, sphexa::sphnnet::ref_vector<Float>>;

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
}