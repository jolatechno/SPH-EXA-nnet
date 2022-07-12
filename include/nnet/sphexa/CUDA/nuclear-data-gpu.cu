#include "nuclear-data-gpu.cuh"

namespace sphexa {
	namespace sphnnet {
		template<size_t n_species, typename Float, typename Int>
		void DeviceNuclearDataType<n_species, Float, Int>::resize(size_t size) {
	        double growthRate = 1;
	        auto   data_      = data();

	        for (size_t i = 0; i < data_.size(); ++i) {
	            if (this->isAllocated(i)) {
	            	// actually resize
	                std::visit([&](auto& arg) { 
	                	size_t previous_size = arg->size();
	                	reallocate(*arg, size, growthRate); 
	                }, data_[i]);
	            }
	        }
		}

	    template class DeviceNuclearDataType<14, double, size_t>;
	    template class DeviceNuclearDataType<86, double, size_t>;
	    template class DeviceNuclearDataType<87, double, size_t>;

	    template class DeviceNuclearDataType<14, float, size_t>;
	    template class DeviceNuclearDataType<86, float, size_t>;
	    template class DeviceNuclearDataType<87, float, size_t>;
	}
}