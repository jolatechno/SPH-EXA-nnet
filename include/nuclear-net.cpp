#include "nnet/net87/net87.cuh"
#include "nnet/net14/net14.cuh"
#include "nnet/eos/helmholtz.cuh"

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