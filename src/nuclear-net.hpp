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
		double theta = 0.7;

		/// minimum timestep
		double min_dt = 1e-20;
		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;
		/// minimum timestep evolution
		double min_dt_step = 1e-2;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 5e-3;
		/// relative mass conservation target of the implicit solver
		double dm_m_target = 1e-5;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 1e-2;
		/// relative mass conservation tolerance of the implicit solver
		double dm_m_tol = 1e-4;

		/// the value that is considered null inside a system
		double epsilon_system = 1e-200;
		/// the value that is considered null inside a state
		double epsilon_vector = 1e-16;
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




	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class Vector, typename Float>
	std::pair<Vector, Float> solve_system(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const Vector &BE, const Vector &A, const Vector &Y,
		const Float T, const Float cv, const Float rho, const Float value_1, const Float dt) {
		/* -------------------
		Solves d{Y, T}/dt = M'*Y using eigen:

		                              D{T, Y} = Dt*(M'  + theta*dM/dT*DT)*{T_in+theta*DT, Y_in+theta*DY}
 	<=>                               D{T, Y} = Dt*((M' + theta*dM/dT*DT)*{T_in,Y_in}  + theta*M'*D{T,Y})
 	<=> (I - Dt*theta*(M' + dM/dT*Y))*D{T, Y} = Dt*M'*{T_in,Y_in}

 		To include temperature:

		dY/dt = (M + dM/dT*dT)*Y + dM*dY
		dT/dt = value_1/cv*T + (dY/dt).Be/cv = value_1/cv*T + (M*Y).Be = value_1/cv*T + (M.T*Be)*Y
		------------------- */
		const int dimension = Y.size();

		eigen::matrix<Float> Mp(dimension + 1, dimension + 1);
		Vector next_Y(dimension);

		// main matrix part
		eigen::matrix<Float> MpYY = order_1_dY_from_reactions(reactions, rates, A, Y, rho);
		for (int i = 1; i <= dimension; ++i)
			for (int j = 1; j <= dimension; ++j)
				Mp(i, j) = MpYY(i - 1, j - 1);

		// right hand side
		Vector RHS(dimension + 1), dY_dt = derivatives_from_reactions(reactions, rates, Y, rho);
		for (int i = 1; i <= dimension; ++i)
			RHS[i] = dY_dt[i - 1]*dt;
		RHS[0] = (T*value_1 + eigen::dot(dY_dt, BE))/cv*dt;

		// include Y -> T terms
		for (int i = 1; i <= dimension; ++i)
			Mp(0, i) = BE[i - 1]/cv;
		Mp(0, 0) = value_1/cv;

		// include rate derivative
		Vector dY_dT = derivatives_from_reactions(reactions, drates_dT, Y, rho);
		for (int i = 1; i <= dimension; ++i)
			Mp(i, 0) = dY_dT[i - 1];

		// construct M
		for (int i = 0; i <= dimension; ++i)
			for (int j = 0; j <= dimension; ++j)
				if (i == j) {
					// diagonal terms
					Mp(i, i) = 1. - constants::theta*dt*Mp(i, i);
				} else
					Mp(i, j) = 	   -constants::theta*dt*Mp(i, j);

		// now solve M*D{T, Y} = RHS
		Vector DY_T = eigen::solve(Mp, RHS);

		// add values
		Float next_T = T + DY_T[0];
		for (int i = 1; i <= dimension; ++i)
			next_Y[i - 1] = Y[i - 1] + DY_T[i];
		return {next_Y, next_T};
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class Vector, class Vector_int, typename Float>
	std::tuple<Vector, Float, Float> solve_system_var_timestep(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const Vector &BE, const Vector_int &A, const Vector &Y,
		const Float T, const Float cv, const Float rho, const Float value_1, Float &dt) {
		const Float m_in = eigen::dot(Y, A);

		// actual solving
		while (true) {
			// solve the system
			auto [next_Y, next_T] = solve_system(reactions, rates, drates_dT, BE, A, Y, T, cv, rho, value_1, dt);

			// cleanup Vector
			utils::clip(next_Y, nnet::constants::epsilon_vector);

			// mass and temperature variation
			Float dm_m = std::abs(1 - eigen::dot(next_Y, A)/m_in);
			Float dT_T = std::abs((next_T - T)/T);

			// timestep tweeking
			Float actual_dt = dt;
			Float dt_multiplier = 
				std::min(
					dT_T == 0 ? (Float)constants::max_dt_step : constants::dT_T_target/dT_T,
					dm_m == 0 ? (Float)constants::max_dt_step : constants::dm_m_target/dm_m
				);
			dt_multiplier = 
				std::min(
					std::max(
						(Float)constants::min_dt_step, 
						dt_multiplier
					),
					(Float)constants::max_dt_step
				);
			dt = std::min(
				(Float)constants::max_dt,
				std::max(
					(Float)constants::min_dt,
					dt*dt_multiplier
				));

			// exit on condition
			if (actual_dt <= constants::min_dt ||
			(dm_m <= constants::dm_m_tol && dT_T <= constants::dT_T_tol))
				return {next_Y, next_T, actual_dt};
		}
	}
}
