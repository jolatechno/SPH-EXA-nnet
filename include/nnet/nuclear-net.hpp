#pragma once

#include "CUDA/cuda.inl"

#include "eigen/eigen.hpp"

#include <iostream>

#include <cmath> // factorial

#include <vector>
#include <tuple>
#include <memory>

namespace nnet {
/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
constants :
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */
	namespace constants {
		/// initial nuclear timestep
		const double initial_dt = 1e-5;

		/// theta for the implicit method
		const double theta = 0.8;

		/// minimum temperature at which we compute the nuclear network
		const double min_temp = 1e8;
		/// minimum density at which we compute the nuclear network
		const double min_rho = 1e5;

		/// maximum timestep
		const double max_dt = 1e-2;
		/// maximum timestep evolution
		const double max_dt_step = 2;
		/// maximum negative timestep evolution
		const double min_dt_step = 1e-2;
		/// timestep jump when a nan is in the solution
		const double nan_dt_step = 2e-1;

		/// relative temperature variation target of the implicit solver
		const double dT_T_target = 4e-3;
		/// relative temperature variation tolerance of the implicit solver
		const double dT_T_tol = 4;

		/// the value that is considered null inside a system
		const double epsilon_system = 1e-40;
		/// the value that is considered null inside a state
		const double epsilon_vector = 1e-16;

		namespace NR {
			/// maximum timestep
			const double max_dt = 1e-2;

			/// relative temperature variation target of the implicit solver
			const double dT_T_target = 1e-2;
			/// relative temperature variation tolerance of the implicit solver
			const double dT_T_tol = 4;

			/// minimum number of newton raphson iterations
			const int min_it = 1;
			/// maximum number of newton raphson iterations
			const int max_it = 11;
			/// tolerance for the correction to break out of the newton raphson loop
			const double it_tol = 1e-7;
		}

		namespace substep {
			/// timestep tolerance for substepping
			const double dt_tol = 1e-6;

			/// ratio of the nuclear timestep and "super timestep" to jump to NSE
			const double dt_nse_tol = 0; //1e-8; // !!!! useless for now
		}
	}


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

	


	/// reaction class
	/**
	 * ...TODO
	 */
	struct reaction {
		/// class representing a product or reactant
		struct reactant_product {
			int species_id, n_consumed = 1;
		};

		std::vector<reactant_product> reactants, products;

		/// reaction class print operator
		friend std::ostream& operator<<(std::ostream& os, const reaction& r);
	};





/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction reference class (referenced from a bigger vector):
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */




	/// class referencing a reaction
	struct reaction_reference {
		HOST_DEVICE_FUN reaction_reference() {}

		/// class simulating a vector from a pointer
		template<class T>
		class vector_reference {
		private:
			const T *ptr = nullptr;
			size_t size_ = 0;
		public:
			HOST_DEVICE_FUN vector_reference() {}
			HOST_DEVICE_FUN vector_reference(const T *ptr_, size_t size) : ptr(ptr_), size_(size) {}
			size_t inline size() const {
				return size_;
			}
			HOST_DEVICE_FUN const T *begin() const {
				return ptr;
			}
			HOST_DEVICE_FUN const T *end() const {
				return ptr + size_;
			}
			HOST_DEVICE_FUN const T &operator[](int i) {
				return ptr[i];
			}
		};

		vector_reference<reaction::reactant_product> reactants, products;

		/// reaction class print operator
		friend std::ostream& operator<<(std::ostream& os, const reaction_reference& r);
	};





/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) reaction list class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



	/// reaction pointer class (pointer to contigous buffers rather then a vector of reaction)
	/**
	 * ...TODO
	 */
	class reaction_list {
	private:
		// pointer to each reaction
		std::vector<int> reactant_begin = {0};
		std::vector<int> product_begin  = {};

		// actual vectors
		std::vector<reaction::reactant_product> reactant_product = {};

		friend class ptr_reaction_list;

	public:

		reaction_list () {}
		reaction_list(std::vector<reaction> const &reactions) {
			for (auto &Reaction : reactions)
				push_back(Reaction);
		}

		void inline push_back(reaction const &Reaction) {
			reactant_product.insert(reactant_product.end(), Reaction.reactants.begin(), Reaction.reactants.end());
			reactant_product.insert(reactant_product.end(), Reaction.products.begin(),  Reaction.products.end());

			product_begin .push_back(reactant_begin.back() + Reaction.reactants.size());
			reactant_begin.push_back(product_begin.back()  + Reaction.products.size());
		}

		reaction_reference inline operator[](int i) const {
			reaction_reference Reaction;

			Reaction.reactants = reaction_reference::vector_reference(reactant_product.data() + reactant_begin[i], product_begin [i    ] - reactant_begin[i]);
			Reaction.products  = reaction_reference::vector_reference(reactant_product.data() + product_begin [i], reactant_begin[i + 1] - product_begin[i]);

			return Reaction;
		}

		size_t inline size() const {
			return product_begin.size();
		}
	};




/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
(nuclear) offloadable reaction list class:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



	class gpu_reaction_list;

	/// reaction pointer class (pointer to contigous buffers rather then a vector of reaction)
	/**
	 * ...TODO
	 */
	class ptr_reaction_list {
	protected:
		// pointer to each reaction
		const int *reactant_begin, *product_begin;
		const reaction::reactant_product *reactant_product;
		int num_reactions;

		// forward declaration
		friend gpu_reaction_list;
		friend gpu_reaction_list move_to_gpu(const ptr_reaction_list &reactions);
		friend void inline free(gpu_reaction_list &reactions);
		
	public:
		ptr_reaction_list() {}
		ptr_reaction_list(reaction_list const &other) {
			num_reactions    = other.size();

			reactant_begin   = other.reactant_begin.data();
			product_begin    = other.product_begin.data();
			reactant_product = other.reactant_product.data();
		}

		HOST_DEVICE_FUN reaction_reference inline operator[](int i) const {
			reaction_reference Reaction;

			Reaction.reactants = reaction_reference::vector_reference(reactant_product + reactant_begin[i], product_begin [i    ] - reactant_begin[i]);
			Reaction.products  = reaction_reference::vector_reference(reactant_product + product_begin [i], reactant_begin[i + 1] - product_begin[i]);

			return Reaction;
		}

		HOST_DEVICE_FUN size_t inline size() const {
			return num_reactions;
		}
	};




/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print functions:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



	std::ostream inline &operator<<(std::ostream& os, const reaction_reference& r) {
		// print reactant
		for (auto [reactant_id, n_reactant_consumed] : r.reactants)
			os << n_reactant_consumed << "*[" << reactant_id << "] ";

		os << " ->  ";

		// print products
		for (auto [product_id, n_product_produced] : r.products)
			os << n_product_produced << "*[" << product_id << "] ";
	    return os;
	}


	std::ostream inline &operator<<(std::ostream& os, const reaction& r) {
		// print reactant
		for (auto [reactant_id, n_reactant_consumed] : r.reactants)
			os << n_reactant_consumed << "*[" << reactant_id << "] ";

		os << " ->  ";

		// print products
		for (auto [product_id, n_product_produced] : r.products)
			os << n_product_produced << "*[" << product_id << "] ";
	    return os;
	}




/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
utils functions:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

	namespace util {
		/// clip the values in a Vector
		/**
		 * clip the values in a Vector, to make 0 any negative value, or values smaller than a tolerance epsilon
		 * ...TODO
		 */
		template<typename Float>
		HOST_DEVICE_FUN void inline clip(Float *X, const int dimension, const Float epsilon) {
			for (int i = 0; i < dimension; ++i)
				if (X[i] <= epsilon) //if (std::abs(X(i)) <= epsilon)
					X[i] = 0;
		}




		/// function to check if there is a nan in both temperature and abundances
		/**
		 * ...TODO
		 */
		template<typename Float>
		HOST_DEVICE_FUN bool inline contain_nan(const Float T, const Float *Y, const int dimension) {
			if (std::isnan(T))
				return true;

			for (int i = 0; i < dimension; ++i)
				if (std::isnan(Y[i]))
					return true;

			return false;
		}




		/// create a first order system from a list of reaction.
		/**
		 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
		 * ...TODO
		 */
		template<typename Float>
		HOST_DEVICE_FUN void inline derivatives_from_reactions(const ptr_reaction_list &reactions, const Float *rates, const Float rho, const Float *Y, Float *dY, const int dimension) {
			// fill with zero
			for (int i = 0; i < dimension; ++i)
				dY[i] = 0.;

			const int num_reactions = reactions.size();
			for (int i = 0; i < num_reactions; ++i) {
				const auto &Reaction = reactions[i];
				Float rate = rates[i];

				// compute rate and order
				for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants) {
					// divide by factorial
					if (n_reactant_consumed != 1)
						rate /= std::tgamma(n_reactant_consumed + 1);

					// multiply by abundance
					rate *= std::pow(Y[reactant_id]*rho, n_reactant_consumed);
				}

				// correct for rho
				rate /= rho;

				// insert consumption rates
				for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants)
					dY[reactant_id] -= rate*n_reactant_consumed;

				// insert production rates
				for (const auto [product_id, n_product_produced] : Reaction.products)
					dY[product_id] += rate*n_product_produced;
			}
		}




		/// create a first order system from a list of reaction.
		/**
		 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
		 * ...TODO
		 */
		template<typename Float>
		HOST_DEVICE_FUN void inline order_1_dY_from_reactions(const ptr_reaction_list &reactions, const Float *rates, const Float rho,
			Float const *Y, Float *M, const int dimension)
		{
			// fill matrix with zero
			for (int i = 0; i < dimension; ++i)
				for (int j = 0; j < dimension; ++j)
					M[(i + 1) + (dimension + 1)*(j + 1)] = 0.;

			const int num_reactions = reactions.size();
			for (int i = 0; i < num_reactions; ++i) {
				const auto &Reaction = reactions[i];
				Float rate = rates[i];

				// compute rate
				for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
					// divide by factorial
					if (n_reactant_consumed != 1)
						rate /= std::tgamma(n_reactant_consumed + 1);

					// multiply by abundance
					rate *= std::pow(Y[reactant_id]*rho, n_reactant_consumed - 1);
				}

				if (rate > constants::epsilon_system)
					for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
						// compute rate
						Float this_rate = rate;
						for (auto &[other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
							// multiply by abundance
							if (other_reactant_id != reactant_id)
								this_rate *= Y[other_reactant_id]*rho;

						// insert consumption rates
						for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
							M[(other_reactant_id + 1) + (dimension + 1)*(reactant_id + 1)] -= this_rate*other_n_reactant_consumed;
							

						// insert production rates
						for (auto const [product_id, n_product_produced] : Reaction.products)
							M[(product_id + 1)        + (dimension + 1)*(reactant_id + 1)] += this_rate*n_product_produced;
					}
			}
		}
	}



/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
First simple direct solver:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



	/// generate the system to be solve (with rates computed at a specific "guess") 
	/**
	 * TODO
	 */
	template<class eos_type, class func_type, typename Float>
	HOST_DEVICE_FUN void inline prepare_system_from_guess(const int dimension, Float *Mp, Float *RHS, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, 
		const Float *Y, const Float T, const Float *Y_guess, const Float T_guess,
		const Float rho, const Float drho_dt,
		const eos_type &eos_struct, const Float dt)
	{
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

 	<=>                               D{T, Y} = Dt*(M + theta*dM/dT*DT)*{T_in,Y_in} + theta*Dt*Myy*D{T,Y})
 	<=> (I - Dt*theta*(Myy + dM/dT*Y))*D{T, Y} = Dt*M*{T_in,Y_in}

 		Energy equation:

		dT/dt*cv = value_1*T + (dY/dt).BE
	<=> DT*cv = value_1*(T + theta*DT) + DY.BE
	<=> DT*(cv - theta*value_1) - DY.BE = value_1*T
		------------------- */
#if !DEVICE_CODE
		if (dt == 0) {
			std::string error = "Zero timestep in nuclear network\n";
			error += "\tT=" + std::to_string(T) + ",\tTguess=" + std::to_string(T_guess) + "\n";
			error += "\trho=" + std::to_string(rho) + "\tdrho/dt=" + std::to_string(drho_dt) + "\n";
			error += "\tdP/dT=" + std::to_string(eos_struct.dpdT) + ", cv=" + std::to_string(eos_struct.cv) + "\n";
			error += "\tY=";
			for (int i = 0; i < dimension; ++i)
				error += std::to_string(Y[i]) + " ";
			error += "\n\tYguess=";
			for (int i = 0; i < dimension; ++i)
				error += std::to_string(Y_guess[i]) + " ";
			
			throw std::runtime_error(error);
		}
#endif

		Float *BE        = RHS + 1;
		Float *drates_dT = Mp  + dimension + 1;

		// construct BE and rates in plance
		construct_rates_BE(Y_guess, T_guess, rho, eos_struct, BE, rates, drates_dT);
		// include rate derivative
		util::derivatives_from_reactions(reactions, drates_dT, rho, Y_guess, &Mp[1], dimension);
		// swap
		for (int i = 0; i < dimension; ++i)
			Mp[0 + (dimension + 1)*(i + 1)] = -BE[i]/eos_struct.cv;

		// compute RHS
		util::derivatives_from_reactions(reactions, rates, rho, Y_guess, &RHS[1], dimension);
		for (int i = 0; i < dimension; ++i)
			RHS[i + 1] *= dt;
		// correct RHS based on the derivative of rates
		for (int i = 0; i < dimension; ++i) {
			//               __*Dt = __*(next_T - T_guess) = __*(next_T - T + T - T_guess) = __*(next_T - T) - __*(T_guess - T)
			// <=> -__*theta*dt*Dt = ... - __*theta*dt*(T_guess - T)
			RHS[i + 1]  += -constants::theta*dt*Mp[(i + 1) + 0]*(T_guess - T);

			// correct rate derivative
			Mp[(i + 1) + 0] *= -constants::theta*dt;
		}

		// main matrix part
		util::order_1_dY_from_reactions(reactions, rates, rho, Y_guess, Mp, dimension);
		for (int i = 0; i < dimension; ++i) {
			//     dY = ... + theta*dt*Mp*(next_Y - Y_guess) = ... + theta*dt*Mp*(next_Y - Y + Y - Y_guess) = ... + theta*dt*Mp*dY - theta*dt*Mp*(Y_guess - Y)
			// <=> dY*(I - theta*dt*Mp) = ... - theta*Mp*dt*(Y_guess - Y)
			Float RHS_correction = 0;
			for (int j = 0; j < dimension; ++j)
				RHS_correction += Mp[(i + 1) + (dimension + 1)*(j + 1)]*(Y_guess[j] - Y[j]);
			RHS[i + 1] += -constants::theta*dt*RHS_correction;
		}
		for (int i = 0; i < dimension; ++i) {
			// diagonal terms
			Mp[(i + 1) + (dimension + 1)*(i + 1)] = 1.      -constants::theta*dt*Mp[(i + 1) + (dimension + 1)*(i + 1)];

			// other terms
			for (int j = 0; j < dimension; ++j)
				if (i != j)
					Mp[(i + 1) + (dimension + 1)*(j + 1)] = -constants::theta*dt*Mp[(i + 1) + (dimension + 1)*(j + 1)];
		}

		// compute value1
		const Float drho = drho_dt*dt;
		const Float value_1 = eos_struct.dpdT*drho/(rho*rho);

		// energy equation
		RHS[0] = T*value_1/eos_struct.cv;
		Mp[0] = 1 - constants::theta*value_1/eos_struct.cv;
	}




	/// second part after solving the system (generated in "prepare_system_from_guess")
	/**
	 * TODO
	 */
	template<typename Float>
	HOST_DEVICE_FUN void inline finalize_system(const int dimension, const Float *Y, const Float T, Float *next_Y, Float &next_T, const Float *DY_T) {
		// increment values
		for (int i = 0; i < dimension; ++i)
			next_Y[i] = Y[i] + DY_T[i + 1];

		// update temperature
		next_T = T + DY_T[0];
	}



	/* actual solver: */
	/// solves a system non-iteratively (with rates computed at a specific "guess").
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M (computed at a specific "guess").
	 * ...TODO
	 */
	template<class func_type, class eos_type, typename Float>
	void inline solve_system_from_guess(const int dimension,
		Float *Mp, Float *RHS, Float *DY_T, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, 
		const Float *Y, const Float T, const Float *Y_guess, const Float T_guess, Float *next_Y, Float &next_T,
		const Float rho, const Float drho_dt,
		const eos_type &eos_struct, const Float dt)
	{
		if (rho < constants::min_rho || T < constants::min_temp) {
			for (int i = 0; i < dimension; ++i)
				next_Y[i] = Y[i];
			next_T = T;
		} else {
			// generate system
			prepare_system_from_guess(dimension,
				Mp, RHS, DY_T, rates,
				reactions, construct_rates_BE,
				Y, T, Y_guess, T_guess,
				rho, drho_dt, eos_struct, dt);

			// solve M*D{T, Y} = RHS
			eigen::solve(Mp, RHS, DY_T, dimension + 1, constants::epsilon_system);

			// finalize
			return finalize_system(dimension, Y, T, next_Y, next_T, DY_T);
		}
	}




	/* actual solver: */
	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class func_type, typename Float>
	void inline solve_system(
		const int dimension,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE,
		const Float *Y, const Float T, Float *next_Y, Float &next_T,
		const Float cv, const Float rho, const Float value_1, const Float dt)
	{

		eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
		eigen::Vector<Float> RHS(dimension + 1);
		eigen::Vector<Float> DY_T(dimension + 1);
		std::vector<Float> rates(dimension*dimension);

		solve_system_from_guess(dimension,
			Mp.data(), RHS.data(), DY_T.data(), rates.data(),
			reactions, construct_rates_BE,
			Y, T, Y, T, next_Y, next_T,
			cv, rho, value_1, dt);
	}




/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Iterative solver:
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */



	/// generate the system to be solve for the iterative solver
	/**
	 * TODO
	 */
	template<class func_type, class func_eos, typename Float>
	HOST_DEVICE_FUN void inline prepare_system_NR(const int dimension, 
		Float *Mp, Float *RHS, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		const Float *Y, Float T, Float *final_Y, Float final_T, 
		const Float rho, const Float drho_dt,
		Float &dt,  const int i)
	{
		// copy if first iteration
		if (i <= 1) {
			for (int j = 0; j < dimension; ++j)
				final_Y[j] = Y[j];
			final_T = T;
		}

		// compute n+theta values
		Float T_theta  = (1 - constants::theta)*T    + constants::theta*final_T;
		for (int j = 0; j < dimension; ++j)
			final_Y[j] = (1 - constants::theta)*Y[j] + constants::theta*final_Y[j];

		// compute eos
		auto eos_struct = eos(final_Y, T_theta, rho);

		// generate system
		prepare_system_from_guess(dimension,
			Mp, RHS, rates,
			reactions, construct_rates_BE,
			Y, T, final_Y, T_theta,
			rho, drho_dt, eos_struct, dt);
	}




	/// second part after solving the system (generated in "prepare_system_NR")
	/**
	 * TODO
	 */
	template<typename Float>
	HOST_DEVICE_FUN bool inline finalize_system_NR(const int dimension,
		const Float *Y, const Float T,
		Float *final_Y, Float &final_T,
		const Float *DY_T, Float &dt, Float &used_dt, int &i)
	{

		Float last_T = final_T;
		finalize_system(dimension, Y, T, final_Y, final_T, DY_T);

		// check for garbage 
		if (util::contain_nan(final_T, final_Y, dimension) || final_T < 0) {
			// set timestep
			dt *= constants::nan_dt_step;

			// jump back
			i = 0;
			used_dt = 0.;
			return false;
		}

		// break condition
		Float dT_T = std::abs((final_T - T)/final_T);
		if (i >= constants::NR::min_it && dT_T > constants::NR::dT_T_target*constants::NR::dT_T_tol) {
			// set timestep
			dt *= constants::NR::dT_T_target/dT_T;

			// jump back
			i = 0;
			used_dt = 0.;
			for (int j = 0; j < dimension; ++j)
				final_Y[j] = Y[j];
			final_T = T;
			return false;
		}

		// cleanup Vector
		util::clip(final_Y, dimension, (Float)nnet::constants::epsilon_vector);
		
		// return condition
		Float correction = std::abs((final_T - last_T)/final_T);
		if ((i >= constants::NR::min_it && correction < constants::NR::it_tol) ||
			i >= constants::NR::max_it)
		{
			// mass and temperature variation
			Float dT_T = std::abs((final_T - T)/final_T);

			// timestep tweeking
			used_dt = dt;
			dt = (dT_T == 0 ? constants::max_dt_step : constants::NR::dT_T_target/dT_T)*used_dt;
			dt = std::min(dt, used_dt*(Float)constants::max_dt_step);
			dt = std::max(dt, used_dt*(Float)constants::min_dt_step);
			dt = std::min(dt,         (Float)constants::NR::max_dt);

			return true;
		}

		// continue the loop
		used_dt = 0.;
		return false;
	}





	/* actual solver: */
	/// solve with  newton raphson
	/**
	 * iterative solver.
	 * ...TODO
	 */
	template<class func_type, class func_eos, typename Float>
	Float inline solve_system_NR(const int dimension,
		Float *Mp, Float *RHS, Float *DY_T, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		const Float *Y, Float T, Float *final_Y, Float &final_T, 
		const Float rho, const Float drho_dt, Float &dt)
	{
		// check for non-burning particles
		if (rho < constants::min_rho || T < constants::min_temp) {
			dt = constants::max_dt;
			return dt;
		}

		// actual solving
		Float timestep = 0;
		for (int i = 1;; ++i) {
			// generate system
			prepare_system_NR(dimension, 
				Mp, RHS, rates,
				reactions, construct_rates_BE, eos,
				Y, T, final_Y, final_T, 
				rho, drho_dt, dt, i);

			// solve M*D{T, Y} = RHS
			eigen::solve(Mp, RHS, DY_T, dimension + 1, constants::epsilon_system);

			// finalize
			if(finalize_system_NR(
				dimension,
				Y, T,
				final_Y, final_T,
				DY_T, dt, timestep, i))
			{
				return timestep;
			}
		}
	}



	/* actual solver: */
	/// solve with  newton raphson
	/**
	 * iterative solver.
	 * ...TODO
	 */
	template<class func_type, class func_eos, typename Float>
	Float inline solve_system_NR(const int dimension,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		const Float *Y, Float T, Float *final_Y, Float &final_T, 
		const Float rho, const Float drho_dt, Float &dt)
	{
		std::vector<Float> rates(reactions.size());
		eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
		eigen::Vector<Float> RHS(dimension + 1);
		eigen::Vector<Float> DY_T(dimension + 1);

		return solve_system_NR(dimension,
			Mp.data(), RHS.data(), DY_T.data(), rates.data(),
			reactions, construct_rates_BE,eos,
			Y, T, final_Y, final_T, 
			rho, drho_dt, dt);
	}



/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Substeping solver
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */




	/// generate the system to be solve for the substepping solver
	/**
	 * TODO
	 */
	template<class func_type, class func_eos, typename Float, class nseFunction=void*>
	HOST_DEVICE_FUN void inline prepare_system_substep(const int dimension,
		Float *Mp, Float *RHS, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		const Float *final_Y, Float final_T, Float *next_Y, Float &next_T, 
		const Float final_rho, const Float drho_dt,
		const Float dt_tot, Float &elapsed_time, Float &dt, const int i,
		const nseFunction jumpToNse=NULL)
	{
		// compute rho
		Float rho = final_rho - drho_dt*(dt_tot - elapsed_time);

#ifndef COMPILE_DEVICE
		// timejump if needed
		if constexpr (std::is_invocable<std::remove_pointer<nseFunction>>())
		if (dt < dt_tot*constants::substep::dt_nse_tol) {
			dt = constants::max_dt;
			elapsed_time = dt_tot;

			(*jumpToNse)(reactions, construct_rates_BE, eos,
				final_Y, final_T, rho, drho_dt);
		}
#endif
		
		// insure convergence to the right time
		Float used_dt = dt;
		if (dt_tot - elapsed_time < dt)
			used_dt = dt_tot - elapsed_time;

		// prepare system
		prepare_system_NR(dimension, 
			Mp, RHS, rates,
			reactions, construct_rates_BE, eos,
			final_Y, final_T, next_Y, next_T, 
			rho, drho_dt, used_dt, i);
	}




	/// second part after solving the system (generated in "prepare_system_from_guess")
	/**
	 * TODO
	 */
	template<typename Float>
	HOST_DEVICE_FUN bool inline finalize_system_substep(const int dimension,
		Float *final_Y, Float &final_T,
		Float *next_Y, Float &next_T,
		const Float *DY_T, const Float dt_tot, Float &elapsed_time,
		Float &dt, int &i)
	{
		// insure convergence to the right time
		Float timestep, used_dt = dt;
		if (dt_tot - elapsed_time < dt)
			used_dt = dt_tot - elapsed_time;

		// finalize system
		bool converged = finalize_system_NR(dimension,
			final_Y, final_T,
			next_Y, next_T,
			DY_T, used_dt, timestep, i);

		// update timestep
		if (dt_tot - elapsed_time < dt) {
			if (used_dt < dt_tot - elapsed_time)
				dt = used_dt;
		} else
			dt = used_dt;

		if (converged) {
			// update state
			for (int j = 0; j < dimension; ++j)
				final_Y[j] = next_Y[j];
			final_T = next_T;

			// jump back, increment time
			i = 0;
			elapsed_time += timestep;

			// check exit condition
			if ((dt_tot - elapsed_time)/dt_tot < constants::substep::dt_tol)
				return true;
		}

		return false;
	}



	/* actual substepping solver: */
	/// function to supperstep (can include jumping to NSE)
	/**
	 * Superstepping using solve_system_NR, might move it to SPH-EXA
	 * ...TODO
	 */
	template<class func_type, class func_eos, typename Float, class nseFunction=void*>
	HOST_DEVICE_FUN void inline solve_system_substep(const int dimension,
		Float *Mp, Float *RHS, Float *DY_T, Float *rates,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		Float *final_Y, Float &final_T, Float *Y_buffer,
		const Float final_rho, const Float drho_dt, Float const dt_tot, Float &dt,
		const nseFunction jumpToNse=NULL)
	{
		// check for non-burning particles
		if (final_rho < constants::min_rho || final_T < constants::min_temp)
			return;

		// actual solving
		Float elapsed_time = 0;
		Float T_buffer;
		for (int i = 1;; ++i) {
			// generate system
			prepare_system_substep(dimension,
				Mp, RHS, rates,
				reactions, construct_rates_BE, eos,
				final_Y, final_T, Y_buffer, T_buffer,
				final_rho, drho_dt,
				dt_tot, elapsed_time, dt, i,
				jumpToNse);

			// solve M*D{T, Y} = RHS
			eigen::solve(Mp, RHS, DY_T, dimension + 1, constants::epsilon_system);

			// finalize
			if(finalize_system_substep(dimension,
				final_Y, final_T,
				Y_buffer, T_buffer,
				DY_T, dt_tot, elapsed_time,
				dt, i))
			{
				break;
			}
		}
	}




	/* actual substepping solver: */
	/// function to supperstep (can include jumping to NSE)
	/**
	 * Superstepping using solve_system_NR, might move it to SPH-EXA
	 * ...TODO
	 */
	template<class func_type, class func_eos, typename Float, class nseFunction=void*>
	void inline solve_system_substep(const int dimension,
		const ptr_reaction_list &reactions, const func_type &construct_rates_BE, const func_eos &eos,
		Float *final_Y, Float &final_T,
		const Float final_rho, const Float drho_dt, Float const dt_tot, Float &dt,
		const nseFunction jumpToNse=NULL)
	{
		std::vector<Float> rates(reactions.size());
		eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);
		eigen::Vector<Float> RHS(dimension + 1), DY_T(dimension + 1), Y_buffer(dimension);

		solve_system_substep(dimension,
			Mp.data(), RHS.data(), DY_T.data(), rates.data(),
			reactions, construct_rates_BE, eos,
			final_Y, final_T, Y_buffer.data(),
			final_rho, drho_dt, dt_tot, dt,
			jumpToNse);
	}
}
