#pragma once

#include "../CUDA/cuda.inl"

namespace nnet::eos {
	namespace ideal_gas_constants {
		const static double Kb = 1.380658e-16;
		const static double Na = 6.022137e23;
		const static double R  = 8.317e7;
	}

	/// helmholtz eos structure
	/**
	 * TODO
	 */
	template<typename Float>
	struct ideal_gas_eos_output {
		HOST_DEVICE_FUN ideal_gas_eos_output() {}
		HOST_DEVICE_FUN ~ideal_gas_eos_output() {}

		Float cv, dpdT, p;
		Float c, u;

		Float dudYe = 0;
	};

	/// helmholtz eos functor
	/**
	*...TODO
	 */
	class ideal_gas_functor {
	private:
		double mu;

	public:
		ideal_gas_functor(double mu_) : mu(mu_) {}
		~ideal_gas_functor() {}

		template<typename Float>
		HOST_DEVICE_FUN ideal_gas_eos_output<Float> inline operator()(const Float *Y, const Float T, const Float rho) const {
			ideal_gas_eos_output<Float> res;

			const Float dmy  = ideal_gas_constants::R / mu;
		            res.cv   = 1.5 * dmy;
		    		res.u    = T * res.cv;
		    		res.p    = rho * T * dmy;
		    		res.c    = std::sqrt(res.p / rho);
		    		res.dpdT = rho * dmy;

			return res;
		}
	};
}