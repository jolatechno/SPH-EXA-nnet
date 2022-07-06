#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <variant>
#include <type_traits>
#include <array>

#include <iostream>

#if COMPILE_DEVICE
    #include <thrust/device_vector.h>
#endif

namespace util {
	template<class T, long unsigned int n>
	using array = std::array<T, n>;
}

template<typename T, size_t n>
std::ofstream& operator<<(std::ofstream& os, const util::array<T, n>& Y);

namespace cstone {
    struct CpuTag
    {
    };
    struct GpuTag
    {
    };
}

namespace sphexa {

    template<class AccType>
    struct HaveGpu : 
        public std::integral_constant<int, std::is_same_v<AccType, cstone::GpuTag>>
    {
    };

	namespace fileutils {
		/*! @brief write fields as columns to an ASCII file
		 *
		 * @tparam  T              field type
		 * @tparam  Separators
		 * @param   firstIndex     first field index to write
		 * @param   lastIndex      last field index to write
		 * @param   path           the file name to write to
		 * @param   append         append or overwrite if file already exists
		 * @param   fields         pointers to field array, each field is a column
		 * @param   separators     arbitrary number of separators to insert between columns, eg '\t', std::setw(n), ...
		 */
		template<class... T, class... Separators>
		void writeAscii(size_t firstIndex, size_t lastIndex, const std::string& path, bool append, const std::vector<std::variant<T*...>>& fields, Separators&&... separators) {
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
		                std::visit([&dumpFile, i](auto& arg) { 
			                	dumpFile << arg[i];
			                }, field);
		            }
		            dumpFile << std::endl;
		        }
		    }
		    else { throw std::runtime_error("Can't open file at path: " + path); }

		    dumpFile.close();
		}
	}

	//! @brief extract a vector of pointers to particle fields for file output
	template<class Dataset>
	auto getOutputArrays(Dataset& dataset)
	{
	    auto fieldPointers = dataset.data();
	    using FieldType    = std::variant<float*, double*, int*, unsigned*, uint64_t*,
	    	util::array<double, 14>*, util::array<double, 86>*, util::array<double, 87>*,
	    	util::array<float,  14>*, util::array<float,  86>*, util::array<float,  87>*>;

	    std::vector<FieldType> outputFields;
	    outputFields.reserve(dataset.outputFieldIndices.size());

	    for (int i : dataset.outputFieldIndices)
	    {
	        if (!dataset.isAllocated(i))
	        {
	            throw std::runtime_error("Cannot output field " + std::string(dataset.fieldNames[i]) +
	                                     ", because it is not active.");
	        }
	        std::visit([&outputFields](auto& arg) { outputFields.push_back(arg->data()); }, fieldPointers[i]);
	    }
	    return outputFields;
	}

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







//! @brief resizes a vector with a determined growth rate upon reallocation
template<class Vector>
void reallocate(Vector& vector, size_t size, double growthRate)
{
    size_t current_capacity = vector.capacity();

    if (size > current_capacity)
    {
        size_t reserve_size = double(size) * growthRate;
        vector.reserve(reserve_size);
    }
    vector.resize(size);
}









namespace sphexa
{

/*! @brief Helper class to keep track of field states
 *
 * @tparam DataType  the array with the particles fields with a static array "fieldNames" and a data()
 *                   function returning variant pointers to the fields
 *
 * -Conserved fields always stay allocated and their state cannot be modified by release/acquire
 *
 * -Dependent fields are also allocated, but the list of dependent fields can be changed at any time
 *  by release/acquire
 *
 * -release and acquire do NOT deallocate or allocate memory, they just pass on existing memory from
 *  one field to another.
 *
 *  This class guarantees that:
 *      -conserved fields are not modified
 *      -only dependent fields can be released
 *      -a field can be acquired only if it is unused and a suitable released field is available
 *
 * It remains the programmers responsibility to not access unused fields or fields outside their allocated bounds.
 */
template<class DataType>
class FieldStates
{
public:
    template<class... Fields>
    void setConserved(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(setState(fields, State::conserved), 0)...};
    }

    template<class... Fields>
    void setDependent(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(setState(fields, State::dependent), 0)...};
    }

    bool isAllocated(size_t fieldIdx) const { return fieldStates_[fieldIdx] != State::unused; }

    //! @brief indicate that @p fields are currently not required
    template<class... Fields>
    void release(const Fields&... fields)
    {
        [[maybe_unused]] std::initializer_list<int> list{(releaseOne(fields), 0)...};
    }

    //! @brief try to acquire memory for @p fields from previously released fields
    template<class... Fields>
    void acquire(const Fields&... fields)
    {
        auto data_ = static_cast<DataType*>(this)->data();

        [[maybe_unused]] std::initializer_list<int> list{(acquireOne(data_, fields), 0)...};
    }

private:
    /*! @brief private constructor to ensure that only class X that is derived from FieldStates<X> can instantiate
     *
     * This prohibits the following:
     *
     * class Y : public FieldStates<X>
     * {};
     */
    FieldStates()
        : fieldStates_(DataType::fieldNames.size(), State::unused)
    {
    }

    friend DataType;

    enum class State
    {
        unused,
        conserved,
        dependent,
        released
    };

    void releaseOne(const std::string& field)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        releaseOne(fieldIdx);
    }

    void releaseOne(int idx)
    {
        if (fieldStates_[idx] != State::dependent)
        {
            throw std::runtime_error("The following field could not be released due to wrong state: " +
                                     std::string(DataType::fieldNames[idx]));
        }

        fieldStates_[idx] = State::released;
    }

    template<class FieldPointers>
    void acquireOne(FieldPointers& data_, const std::string& field)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        acquireOne(data_, fieldIdx);
    }

    template<class FieldPointers>
    void acquireOne(FieldPointers& data_, int fieldIdx)
    {
        if (fieldStates_[fieldIdx] != State::unused)
        {
            throw std::runtime_error("The following field could not be acquired because already in use: " +
                                     std::string(DataType::fieldNames[fieldIdx]));
        }

        auto checkTypesMatch = [](const auto* var1, const auto* var2)
        {
            using Type1 = std::decay_t<decltype(*var1)>;
            using Type2 = std::decay_t<decltype(*var2)>;
            return std::is_same_v<Type1, Type2>;
        };

        auto swapFields = [](auto* varPtr1, auto* varPtr2)
        {
            using Type1 = std::decay_t<decltype(*varPtr1)>;
            using Type2 = std::decay_t<decltype(*varPtr2)>;
            if constexpr (std::is_same_v<Type1, Type2>) { swap(*varPtr1, *varPtr2); }
        };

        for (size_t i = 0; i < fieldStates_.size(); ++i)
        {
            if (fieldStates_[i] == State::released)
            {
                bool typesMatch = std::visit(checkTypesMatch, data_[i], data_[fieldIdx]);
                if (typesMatch)
                {
                    std::visit(swapFields, data_[i], data_[fieldIdx]);
                    fieldStates_[i]        = State::unused;
                    fieldStates_[fieldIdx] = State::dependent;
                    return;
                }
            }
        }
        throw std::runtime_error("Could not acquire field " + std::string(DataType::fieldNames[fieldIdx]) +
                                 ". No suitable field available");
    }

    void setState(const std::string& field, State state)
    {
        int idx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        fieldStates_[idx] = state;
    }

    //! @brief current state of each field
    std::vector<State> fieldStates_;
};




#if COMPILE_DEVICE
template<class ThrustVec>
typename ThrustVec::value_type* rawPtr(ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}

template<class ThrustVec>
const typename ThrustVec::value_type* rawPtr(const ThrustVec& p)
{
    assert(p.size() && "cannot get pointer to unallocated device vector memory");
    return thrust::raw_pointer_cast(p.data());
}
#endif



template<class Dataset, std::enable_if_t<not HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
void transferToDevice(Dataset&, size_t, size_t, const std::vector<std::string>&)
{
}

template<class Dataset, std::enable_if_t<not HaveGpu<typename Dataset::AcceleratorType>{}, int> = 0>
void transferToHost(Dataset&, size_t, size_t, const std::vector<std::string>&)
{
}



template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferToDevice(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
#if COMPILE_DEVICE
    auto hostData = d.data();
    auto deviceData = d.devData.data();

    auto launchTransfer = [first, last](const auto* hostField, auto* deviceField)
    {
        using Type1 = std::decay_t<decltype(*hostField)>;
        using Type2 = std::decay_t<decltype(*deviceField)>;
        if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>)
        {
            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
            size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
            // CHECK_CUDA_ERR(
            gpuErrchk(
                cudaMemcpy(
                rawPtr(*deviceField) + first, hostField->data() + first, transferSize, cudaMemcpyHostToDevice));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
#endif
}

template<class DataType, std::enable_if_t<HaveGpu<typename DataType::AcceleratorType>{}, int> = 0>
void transferToHost(DataType& d, size_t first, size_t last, const std::vector<std::string>& fields)
{
#if COMPILE_DEVICE
    auto hostData   = d.data();
    auto deviceData = d.devData.data();

    auto launchTransfer = [first, last](auto* hostField, const auto* deviceField)
    {
        using Type1 = std::decay_t<decltype(*hostField)>;
        using Type2 = std::decay_t<decltype(*deviceField)>;
        if constexpr (std::is_same_v<typename Type1::value_type, typename Type2::value_type>)
        {
            assert(hostField->size() > 0);
            assert(deviceField->size() > 0);
            size_t transferSize = (last - first) * sizeof(typename Type1::value_type);
            // CHECK_CUDA_ERR(
            gpuErrchk(
                cudaMemcpy(
                hostField->data() + first, rawPtr(*deviceField) + first, transferSize, cudaMemcpyDeviceToHost));
        }
        else { throw std::runtime_error("Field type mismatch between CPU and GPU in copy to device");
        }
    };

    for (const auto& field : fields)
    {
        int fieldIdx =
            std::find(DataType::fieldNames.begin(), DataType::fieldNames.end(), field) - DataType::fieldNames.begin();
        std::visit(launchTransfer, hostData[fieldIdx], deviceData[fieldIdx]);
    }
#endif
}

} // namespace sphexa