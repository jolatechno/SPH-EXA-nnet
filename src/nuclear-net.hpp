#pragma once

#include "eigen.hpp"

#include <cmath> // factorial
//#include <ranges> // drop

#include <vector>
#include <tuple>

/* !!!!!!!!!!!!
debuging :
!!!!!!!!!!!! */
bool net14_debug = false;

namespace nnet {
	namespace constants {
		/// theta for the implicit method
		double theta = 1;

		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 5e-3;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 10; //1e-1;

		/// the value that is considered null inside a system
		double epsilon_system = 1e-200;
		/// the value that is considered null inside a state
		double epsilon_vector = 1e-16;

		/// minimum number of newton raphson iterations
		uint min_NR_it = 2;
		/// maximum number of newton raphson iterations
		uint max_NR_it = 10;
		/// tolerance for the correction to break out of the newton raphson loop
		double NR_tol = 1e-8;

		/// timestep tolerance for superstepping
		double dt_tol = 1e-5;
	}




	/// reaction class
	/**
	 * ...TODO
	 */
	struct reaction {
		struct reactant {
			int reactant_id, n_reactant_consumed = 1;
		};
		struct product {
			int product_id, n_product_produced = 1;
		};
		std::vector<reactant> reactants;
		std::vector<product> products;
	};




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

		for (int i = 0; i < reactions.size() && i < rates.size(); ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute rate and order
			int order = 0;
			for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				rate *= std::pow(Y[reactant_id], n_reactant_consumed);
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
		Vector const &A, Vector const &Y,
		const Float rho) {
		const int dimension = Y.size();

		eigen::matrix<Float> M(dimension, dimension);

		for (int i = 0; i < reactions.size() && i < rates.size(); ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute order and correct for rho
			int order = 0;
			for (auto const [_, n_reactant_consumed] : Reaction.reactants)
				order += n_reactant_consumed;
			rate *= std::pow(rho, order - 1);

			for (auto const [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				// compute rate
				Float this_rate = rate;
				this_rate *= std::pow(Y[reactant_id], n_reactant_consumed - 1);
				for (auto &[other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
					/* debugging */if (other_reactant_id != reactant_id)/* debugging */
					this_rate *= std::pow(Y[other_reactant_id], other_n_reactant_consumed);

				// insert consumption rates
				for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
					if (other_reactant_id != reactant_id)
						M(other_reactant_id, reactant_id) -= this_rate*other_n_reactant_consumed;

				// insert production rates
				for (auto const [product_id, n_product_produced] : Reaction.products)
					M(product_id, reactant_id) += this_rate*n_product_produced;
			}
		}

		// correct mass gain rate
		for (int i = 0; i < dimension; ++i) {
			// compute the discripency
			Float mass_gain_rate = 0;
			for (int j = 0; j < dimension; ++j)
				mass_gain_rate += M(j, i)*A[j];

			// correct it
			M(i, i) -= mass_gain_rate/A[i];
		}

		return M;
	}




	/// solves a system non-iteratively (with rates computed at a specific "guess").
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M (computed at a specific "guess").
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::pair<Vector, Float> solve_system_from_guess(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const Vector &BE, const Vector &A, 
		const Vector &Y, const Float T, const Vector &Y_guess, const Float T_guess,
		const Float cv, const Float rho, const Float value_1, const Float dt) {
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		                              D{T, Y} = Dt*(M'  + theta*dM/dT*DT)*{T_in+theta*DT, Y_in+theta*DY}
 	<=>                               D{T, Y} = Dt*((M' + theta*dM/dT*DT)*{T_in,Y_in}  + theta*M'*D{T,Y})
 	<=> (I - Dt*theta*(M' + dM/dT*Y))*D{T, Y} = Dt*M'*{T_in,Y_in}

 		Energy equation:

		dT/dt*cv = value_1*T + (dY/dt).BE
	<=> DT*cv = value_1*(T + theta*DT) + DY.BE
	<=> DT*(cv - theta*value_1) - DY.BE = value_1*T
		------------------- */
		const int dimension = Y.size();

		eigen::matrix<Float> Mp(dimension + 1, dimension + 1);
		Vector next_Y(dimension);

		// main matrix part
		eigen::matrix<Float> MpYY = order_1_dY_from_reactions(reactions, rates, A, Y, rho);
		for (int i = 1; i <= dimension; ++i) {
			// diagonal terms
			Mp(i, i) = 1.     -constants::theta*dt*MpYY(i - 1, i - 1);

			// other terms
			for (int j = 1; j <= dimension; ++j)
				if (i != j)
					Mp(i, j) = -constants::theta*dt*MpYY(i - 1, j - 1);
		}

		// right hand side
		Vector RHS(dimension + 1), dY_dt = derivatives_from_reactions(reactions, rates, Y, rho);
		for (int i = 1; i <= dimension; ++i)
			RHS[i] = dY_dt[i - 1]*dt;

		// energy equation
		RHS[0] = T*value_1;
		Mp(0, 0) = cv - constants::theta*value_1;
		for (int i = 1; i <= dimension; ++i)
			Mp(0, i) = -BE[i - 1];

		// include rate derivative
		Vector dY_dT = derivatives_from_reactions(reactions, drates_dT, Y, rho);
		for (int i = 1; i <= dimension; ++i)
			Mp(i, 0) = -constants::theta*dt*dY_dT[i - 1];



		// !!!!!!!!!!
		// debuging:
		if (net14_debug) {
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
		for (int i = 1; i <= dimension; ++i)
			next_Y[i - 1] = Y[i - 1] + DY_T[i];

		// update temperature
		// Float next_T = T + DY_T[0];
		Float next_T = T;
		for (int i = 0; i < dimension; ++i)
			next_T += DY_T[i + 1]*BE[i]/cv;

		return {next_Y, next_T};
	}





	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::pair<Vector, Float> solve_system(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const Vector &BE, const Vector &A, const Vector &Y,
		const Float T, const Float cv, const Float rho, const Float value_1, const Float dt) {
		return solve_system_from_guess(reactions, rates, drates_dT, 
			BE, A, 
			Y, T, Y, T,
			cv, rho, value_1, dt);
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::tuple<Vector, Float, Float> solve_system_var_timestep(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const Vector &BE, const Vector &A, const Vector &Y,
		const Float T, const Float cv, const Float rho, const Float value_1, Float &dt) {
		const Float m_in = eigen::dot(Y, A);

		// actual solving
		while (true) {
			// solve the system
			auto [next_Y, next_T] = solve_system(reactions, rates, drates_dT, BE, A, Y, T, cv, rho, value_1, dt);

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
			if (dT_T <= constants::dT_T_tol)
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
		const Vector &A, const Vector &Y, const Float T, Float &dt) {

		Vector final_Y = Y;
		Float final_T = T;

		while (true) {
			// actual solving
			for (int i = 0; i < constants::max_NR_it; ++i) {
				/* TODO */

				// compute rate
				auto [rates, drates_dT] = construct_rates(final_T);
				auto BE = construct_BE(final_Y, final_T);
				auto [cv, rho, value_1] = eos(final_Y, final_T);

				// solve the system
				std::tie(final_Y, final_T) = solve_system_from_guess(reactions, rates, drates_dT, 
					BE, A, 
					Y, T, final_Y, final_T,
					cv, rho, value_1, dt);

				// cleanup Vector
				utils::clip(final_Y, nnet::constants::epsilon_vector);

				Float correction = 0;

				if (i >= constants::min_NR_it && correction < constants::NR_tol)
					break;
			}

			// mass and temperature variation
			Float dT_T = std::abs((final_T - T)/T);

			// timestep tweeking
			Float previous_dt = dt;
			dt = (dT_T == 0 ? (Float)constants::max_dt_step : constants::dT_T_target/dT_T)*previous_dt;
			dt = std::min(dt, previous_dt*constants::max_dt_step);
			dt = std::min(dt, (Float)constants::max_dt);

			// exit on condition
			if (dT_T <= constants::dT_T_tol)
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
		const Vector &A, const Vector &Y, const Float T, Float const dt_tot, Float &dt) {

		Float elapsed_t = 0;
		Float used_dt = dt;

		Vector final_Y = Y;
		Float final_T = T;

		while (true) {
			// insure convergence to the right time
			bool update_dt = (dt_tot - elapsed_t) > used_dt;
			if (!update_dt)
				used_dt = dt_tot - elapsed_t;

			// compute rate
			auto [rates, drates_dT] = construct_rates(final_T);
			auto BE = construct_BE(final_Y, final_T);
			auto [cv, rho, value_1] = eos(final_Y, final_T);

			// solve system
			auto [next_Y, next_T, this_dt] = solve_system_var_timestep(reactions, rates, drates_dT,
				BE, A, final_Y,
				final_T, cv, rho, value_1, used_dt);
			elapsed_t += this_dt;
			final_Y = next_Y;
			final_T = next_T;

			// update dt
			if (update_dt)
				dt = used_dt;

			// exit condition
			if ((dt_tot - elapsed_t)/dt_tot < constants::dt_tol)
				return {final_Y, final_T};
		} 
	}
}
