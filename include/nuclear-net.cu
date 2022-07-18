#include "nnet/net87/net87.hpp"
#include "nnet/net86/net86.hpp"
#include "nnet/net14/net14.hpp"

#include "nnet/eos/helmholtz.hpp"
#include "nnet/eos/ideal_gas.hpp"



bool test = []() {
	size_t size; void *ptr;

	gpuErrchk(cudaGetSymbolAddress(&ptr, nnet::eos::helmholtz_constants::dev_d));
	gpuErrchk(cudaGetSymbolSize(&size, nnet::eos::helmholtz_constants::dev_d));
	std::cerr << "\n\t(nulcear-net.cu main): dev_d = " << std::hex << (size_t)ptr << std::dec << ", size = " << size << " (" << nnet::eos::helmholtz_constants::imax*sizeof(double) << ")\n";

	gpuErrchk(cudaGetSymbolAddress(&ptr, nnet::eos::helmholtz_constants::dev_f));
	gpuErrchk(cudaGetSymbolSize(&size, nnet::eos::helmholtz_constants::dev_f));
	std::cerr << "\t(nulcear-net.cu main): dev_f = " << std::hex << (size_t)ptr << std::dec << ", size = " << size << " (" << nnet::eos::helmholtz_constants::imax*nnet::eos::helmholtz_constants::jmax*sizeof(double) << ")\n\n";

	return true;
}();



namespace nnet {
	namespace net87 {
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace net86 {
		bool debug = false;
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace net14 {
		bool debug = false;
		compute_reaction_rates_functor compute_reaction_rates;
	}
	namespace eos {
		bool debug = false;
	}
}