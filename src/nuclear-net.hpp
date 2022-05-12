#pragma once

#include "eigen.hpp"

#include <cmath> // factorial
//#include <ranges> // drop

#include <vector>
#include <tuple>


namespace nnet {
	/* !!!!!!!!!!!!
	debuging :
	!!!!!!!!!!!! */
	bool debug = false;

	namespace constants {
		/// theta for the implicit method
		double theta = 1;

		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;
		/// timestep jump when a nan is in the solution
		double nan_dt_step = 1e-1;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 4e-3;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 10;

		/// the value that is considered null inside a system
		double epsilon_system = 1e-200;
		/// the value that is considered null inside a state
		double epsilon_vector = 1e-16;

		namespace NR {
			/// maximum timestep
			double max_dt = 1e-2;
			/// maximum timestep evolution
			double max_dt_step = 1.5;

			/// relative temperature variation target of the implicit solver
			double dT_T_target = 2e-2;
			/// relative temperature variation tolerance of the implicit solver
			double dT_T_tol = 5;

			/// minimum number of newton raphson iterations
			uint min_it = 2;
			/// maximum number of newton raphson iterations
			uint max_it = 10;
			/// tolerance for the correction to break out of the newton raphson loop
			double it_tol = 1e-5;
		}

		namespace superstep {
			/// timestep tolerance for superstepping
			double dt_tol = 1e-5;

			/// ratio of the nuclear timestep and "super timestep" to jump to NSE
			double dt_nse_tol = 1e-8;
		}
	}


	/// reaction class
	/**
	 * ...TODO
	 */
	struct reaction {
		friend std::ostream& operator<<(std::ostream& os, const reaction& r);

		struct reactant {
			int reactant_id, n_reactant_consumed = 1;
		};
		struct product {
			int product_id, n_product_produced = 1;
		};
		std::vector<reactant> reactants;
		std::vector<product> products;
	};

	/// reaction class print operator
	std::ostream& operator<<(std::ostream& os, const reaction& r) {
		// print reactant
		for (auto [reactant_id, n_reactant_consumed] : r.reactants)
			os << n_reactant_consumed << "*[" << reactant_id << "] ";

		os << " ->  ";

		// print products
		for (auto [product_id, n_product_produced] : r.products)
			os << n_product_produced << "*[" << product_id << "] ";
	    return os;
	}




	namespace utils {
		/// clip the values in a Vector
		/**
		 * clip the values in a Vector, to make 0 any negative value, or values smaller than a tolerance epsilon
		 * ...TODO
		 */
		template<typename Float, class Vector>
		void clip(Vector &X, const Float epsilon) {
			const int dimension = X.size();

			for (int i = 0; i < dimension; ++i)
				if (X[i] <= epsilon) //if (std::abs(X(i)) <= epsilon)
					X[i] = 0;
		}




		/// function to check if there is a nan in both temperature and abundances
		/**
		 * ...TODO
		 */
		template<typename Float, class Vector>
		bool contain_nan(const Vector &Y, const Float T) {
			if (std::isnan(T))
				return true;

			const int dimension = Y.size();
			for (int i = 0; i < dimension; ++i)
				if (std::isnan(Y[i]))
					return true;

			return false;
		}
	}



	/// create a first order system from a list of reaction.
	/**
	 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
	 * ...TODO
	 */
	template<typename Float, class Vector>
	Vector derivatives_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, Vector const &Y, const Float rho) {
		const int dimension = Y.size();

		Vector dY(dimension);

		const int num_reactions = reactions.size();
		if (num_reactions != rates.size()) {
			std::cerr << "number of reaction and rates don't match !\n";
			throw;
		}

		for (int i = 0; i < num_reactions; ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute rate and order
			int order = 0;
			for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				// divide by factorial
				rate /= std::tgamma(n_reactant_consumed + 1);

				// multiply by abundance
				rate *= std::pow(Y[reactant_id], n_reactant_consumed);

				// increment order
				order += n_reactant_consumed;
			}

			// correct for rho
			rate *= std::pow(rho, order - 1);

			// insert consumption rates
			for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants)
				dY[reactant_id] -= rate*n_reactant_consumed;

			// insert production rates
			for (auto const [product_id, n_product_produced] : Reaction.products)
				dY[product_id] += rate*n_product_produced;
		}

		return dY;
	}




	/// create a first order system from a list of reaction.
	/**
	 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
	 * ...TODO
	 */
	template<typename Float, class Vector>
	eigen::matrix<Float> order_1_dY_from_reactions(const std::vector<reaction> &reactions, const Vector &rates,
		Vector const &Y,
		const Float rho) {
		const int dimension = Y.size();

		eigen::matrix<Float> M(dimension, dimension);

		const int num_reactions = reactions.size();
		if (num_reactions != rates.size()) {
			std::cerr << "number of reaction and rates don't match !\n";
			throw;
		}

		for (int i = 0; i < num_reactions; ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute rate and order
			int order = 0;
			for (auto const [_, n_reactant_consumed] : Reaction.reactants) {
				// divide by factorial
				if (n_reactant_consumed != 1)
					rate /= std::tgamma(n_reactant_consumed + 1);

				// increment order
				order += n_reactant_consumed;
			}

			// correct for rho
			rate *= std::pow(rho, order - 1);

			for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				// compute rate
				Float this_rate = rate;
				this_rate *= std::pow(Y[reactant_id], n_reactant_consumed - 1);
				for (auto &[other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
					// multiply by abundance
					if (other_reactant_id != reactant_id)
						this_rate *= std::pow(Y[other_reactant_id], other_n_reactant_consumed);

				// insert consumption rates
				for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
					M(other_reactant_id, reactant_id) -= this_rate*other_n_reactant_consumed;

				// insert production rates
				for (auto const [product_id, n_product_produced] : Reaction.products)
					M(product_id, reactant_id) += this_rate*n_product_produced;
			}
		}

		return M;
	}




	/// solves a system non-iteratively (with rates computed at a specific "guess").
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M (computed at a specific "guess").
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::pair<Vector, Float> solve_system_from_guess(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT, const Vector &BE, 
		const Vector &Y, const Float T, const Vector &Y_guess, const Float T_guess,
		const Float cv, const Float rho, const Float value_1, const Float dt) {
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

 	<=>                               D{T, Y} = Dt*(M + theta*dM/dT*DT)*{T_in,Y_in} + theta*Dt*Myy*D{T,Y})
 	<=> (I - Dt*theta*(Myy + dM/dT*Y))*D{T, Y} = Dt*M*{T_in,Y_in}

 		Energy equation:

		dT/dt*cv = value_1*T + (dY/dt).BE
	<=> DT*cv = value_1*(T + theta*DT) + DY.BE
	<=> DT*(cv - theta*value_1) - DY.BE = value_1*T
		------------------- */
		const int dimension = Y.size();

		eigen::matrix<Float> Mp(dimension + 1, dimension + 1);
		Vector next_Y(dimension);

		// right hand side
		Vector RHS(dimension + 1), dY_dt = derivatives_from_reactions(reactions, rates, Y_guess, rho);
		for (int i = 1; i <= dimension; ++i)
			RHS[i] = dY_dt[i - 1]*dt;

		// main matrix part
		eigen::matrix<Float> MpYY = order_1_dY_from_reactions(reactions, rates, Y_guess, rho);
		for (int i = 0; i < dimension; ++i) {
			// diagonal terms
			Mp(i + 1, i + 1) = 1.      -constants::theta*dt*MpYY(i, i);

			// other terms
			for (int j = 0; j < dimension; ++j)
				if (i != j)
					Mp(i + 1, j + 1) = -constants::theta*dt*MpYY(i, j);

			//     dY = ... + theta*dt*Mp*(next_Y - Y_guess) = ... + theta*dt*Mp*(next_Y - Y + Y - Y_guess) = ... + theta*dt*Mp*dY - theta*dt*Mp*(Y_guess - Y)
			// <=> dY*(I - theta*dt*Mp) = ... - theta*Mp*dt*(Y_guess - Y)
			Float RHS_correction = 0;
			for (int j = 0; j < dimension; ++j)
				RHS_correction += MpYY(i, j)*(Y_guess[j] - Y[j]);
			RHS[i + 1] += -constants::theta*dt*RHS_correction;
		}

		// energy equation
		RHS[0] = T*value_1/cv;
		Mp(0, 0) = 1 - constants::theta*value_1/cv;
		for (int i = 0; i < dimension; ++i)
			Mp(0, i + 1) = -BE[i]/cv;

		// include rate derivative
		Vector dY_dT = derivatives_from_reactions(reactions, drates_dT, Y_guess, rho);
		for (int i = 0; i < dimension; ++i) {
			Mp(i + 1, 0) = -constants::theta*dt*dY_dT[i];

			// *Dt = -__*(next_T - T_guess) = -__*(next_T - T + T - T_guess) = -__*(next_T - T) - __*(T - T_guess)
			RHS[i + 1]  += -constants::theta*dt*dY_dT[i]*(T_guess - T);
		}



		// !!!!!!!!!!
		// debuging:
		if (debug) {
			std::cout << "BE=";
			for (int i = 0; i < dimension; ++i)
				std::cout << "\t" << BE[i];

			std::cout << "\nRHS=";
			for (int i = 0; i <= dimension; ++i)
				std::cout << "\t" << RHS[i];

			std::cout << "\nM=";
			for (int i = 0; i <= dimension; ++i) {
				if (i > 0)
					std::cout << "  ";
				for (int j = 0; j <= dimension; ++j)
					std::cout << "\t" << Mp(i, j);
				std::cout << "\n";
			}
		}



		// now solve M*D{T, Y} = RHS
		Vector DY_T = eigen::solve(Mp, RHS);

		// increment values
		for (int i = 0; i < dimension; ++i)
			next_Y[i] = Y[i] + DY_T[i + 1];

		// update temperature
		Float next_T = T + DY_T[0];

		return {next_Y, next_T};
	}





	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::pair<Vector, Float> solve_system(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT, const Vector &BE,
		const Vector &Y, const Float T,
		const Float cv, const Float rho, const Float value_1, const Float dt) {
		return solve_system_from_guess(reactions, rates, drates_dT, 
			BE,
			Y, T, Y, T,
			cv, rho, value_1, dt);
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::tuple<Vector, Float, Float> solve_system_var_timestep(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT, const Vector &BE,
		const Vector &Y, const Float T,
		const Float cv, const Float rho, const Float value_1, Float &dt) {

		// actual solving
		while (true) {
			// solve the system
			auto [next_Y, next_T] = solve_system(reactions, rates, drates_dT, BE, Y, T, cv, rho, value_1, dt);

			// cleanup Vector
			utils::clip(next_Y, nnet::constants::epsilon_vector);

			// mass and temperature variation
			Float dT_T = std::abs((next_T - T)/T);

			// timestep tweeking
			Float previous_dt = dt;
			dt = (dT_T == 0 ? (Float)constants::max_dt_step : constants::dT_T_target/dT_T)*previous_dt;
			dt = std::min(dt, previous_dt*constants::max_dt_step);
			dt = std::min(dt, (Float)constants::max_dt);

			// exit on condition
			if (dT_T <= constants::dT_T_target*constants::dT_T_tol)
				return {next_Y, next_T, previous_dt};
		}
	}




	/// solve with  newton raphson
	/**
	 * Superstepping using solve_system_var_timestep, might move it to SPH-EXA
	 * ...TODO
	 */
	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float>
	std::tuple<Vector, Float, Float> solve_system_NR(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, Float T, const Float rho, const Float drho_dt, Float &dt) {
		const int dimension = Y.size();

		while (true) {
			Vector Y_theta(dimension), final_Y = Y;
			Float T_theta, final_T = T;

			// actual solving
			for (int i = 0; i < constants::NR::max_it; ++i) {
				if (dt == 0) {
					std::cerr << "zero timestep !\n";
					throw;
				}

				// compute n+theta values
				T_theta =        (1 - constants::theta)*T    + constants::theta*final_T;
				for (int j = 0; j < dimension; ++j)
					Y_theta[j] = (1 - constants::theta)*Y[j] + constants::theta*final_Y[j];

				// compute rate
				auto [rates, drates_dT] = construct_rates(         T_theta, rho);
				auto BE                 = construct_BE   (Y_theta, T_theta, rho);
				auto eos_struct         = eos            (Y_theta, T_theta, rho);

				// compute value_1
				double value_1 = drho_dt/(rho*rho)*eos_struct.dP_dT;

				// solve the system
				Float last_T = final_T;
				std::tie(final_Y, final_T) = solve_system_from_guess(
					reactions, rates, drates_dT, BE,
					Y, T, Y_theta, T_theta,
					eos_struct.cv, rho, value_1, dt);

				// check for nan
				if (utils::contain_nan(final_Y, final_T)) {
					// set timestep
					dt *= constants::nan_dt_step;

					// jump back
					final_Y = Y;
					final_T = T;
					i = 0;
				} else {
					// cleanup Vector
					utils::clip(final_Y, nnet::constants::epsilon_vector);

					// exit loop on condition
					Float correction = std::abs((final_T - last_T)/final_T);
					if (i >= constants::NR::min_it && correction < constants::NR::it_tol)
						break;
				}
			}

			// mass and temperature variation
			Float dT_T = std::abs((final_T - T)/final_T);

			// timestep tweeking
			Float previous_dt = dt;
			dt = (dT_T == 0 ? (Float)constants::NR::max_dt_step : constants::NR::dT_T_target/dT_T)*previous_dt;
			dt = std::min(dt, previous_dt*constants::NR::max_dt_step);
			dt = std::min(dt,      (Float)constants::NR::max_dt);

			// exit on condition
			if (dT_T <= constants::NR::dT_T_target*constants::NR::dT_T_tol)
				return {final_Y, final_T, previous_dt};
		}
	}



	/// function to supperstep
	/**
	 * Superstepping using solve_system_var_timestep, might move it to SPH-EXA
	 * ...TODO
	 */
	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float>
	std::tuple<Vector, Float> solve_system_superstep(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, const Float T, const Float rho, const Float drho_dt, Float const dt_tot, Float &dt);



	/// jump to Nuclear Statistical Equilibrium
	/**
	 * Used inside of solve_system_superstep
	 * STUPID IMPLEMENTATION, TODO
	 * ...TODO
	 */
	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float>
	std::tuple<Vector, Float> find_nse(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, const Float T, const Float rho, const Float drho_dt) {

		/* TODO: real implementation
		CURRENT: arbitrary time jump (unefficient) */
		Float dt_tot = 1e-5, used_dt = 1e-12;
		return solve_system_superstep(reactions, construct_rates, construct_BE, eos,
			Y, T, rho, drho_dt, dt_tot, used_dt);
	}



	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float>
	std::tuple<Vector, Float> solve_system_superstep(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, const Float T, const Float rho, const Float drho_dt, Float const dt_tot, Float &dt) {

		Float elapsed_t = 0;
		Float used_dt = dt;

		Vector final_Y = Y;
		Float final_T = T;

		while (true) {
			// insure convergence to the right time
			bool update_dt = (dt_tot - elapsed_t) > used_dt;
			if (!update_dt)
				used_dt = dt_tot - elapsed_t;

			// solve system
			auto [next_Y, next_T, this_dt] = solve_system_NR(reactions, construct_rates, construct_BE, eos,
				final_Y, final_T, rho, drho_dt, used_dt);
			elapsed_t += this_dt;
			final_Y = next_Y;
			final_T = next_T;

			// update dt
			if (update_dt)
				dt = used_dt;

			// exit condition
			if ((dt_tot - elapsed_t)/dt_tot < constants::superstep::dt_tol)
				return {final_Y, final_T};

			// timejump if needed
			if (dt < dt_tot*constants::superstep::dt_nse_tol) {
				dt = constants::max_dt;
				return find_nse(reactions, construct_rates, construct_BE, eos,
					final_Y, final_T, rho, drho_dt);
			}
		} 
	}
}
