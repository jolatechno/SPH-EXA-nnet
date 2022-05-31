#pragma once

#include "eigen.hpp"

#include <iostream>

#include <cmath> // factorial

#include <vector>
#include <tuple>


namespace nnet {
	/* !!!!!!!!!!!!
	debuging :
	!!!!!!!!!!!! */
	bool debug = false;

	namespace constants {
		/// theta for the implicit method
		double theta = 0.8;

		/// minimum temperature at which we compute the nuclear network
		double min_temp = 1e8;
		/// minimum density at which we compute the nuclear network
		double min_rho = 1e5;

		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 2;
		/// maximum negative timestep evolution
		double min_dt_step = 1e-2;
		/// timestep jump when a nan is in the solution
		double nan_dt_step = 2e-1;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 4e-3;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 4;

		/// the value that is considered null inside a system
		double epsilon_system = 1e-100;
		/// the value that is considered null inside a state
		double epsilon_vector = 1e-16;

		namespace NR {
			/// maximum timestep
			double max_dt = 1e-2;

			/// relative temperature variation target of the implicit solver
			double dT_T_target = 2e-2;
			/// relative temperature variation tolerance of the implicit solver
			double dT_T_tol = 4;

			/// minimum number of newton raphson iterations
			uint min_it = 1;
			/// maximum number of newton raphson iterations
			uint max_it = 11;
			/// tolerance for the correction to break out of the newton raphson loop
			double it_tol = 1e-7;
		}

		namespace substep {
			/// timestep tolerance for substepping
			double dt_tol = 1e-6;

			/// ratio of the nuclear timestep and "super timestep" to jump to NSE
			double dt_nse_tol = 0; //1e-8; // !!!! useless for now
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
#ifdef USE_EIGEN
	eigen::Vector<Float>
#else
	Vector
#endif
	derivatives_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, Vector const &Y, const Float rho) {
		const int dimension = Y.size();

#ifdef USE_EIGEN
		eigen::Vector<Float> dY(dimension);
#else
		Vector dY = Y;
#endif
		
		for (int i = 0; i < dimension; ++i)
			dY[i] = 0.;

		const int num_reactions = reactions.size();
		if (num_reactions != rates.size())
			throw std::runtime_error("Number of reaction and rates don't match !\n");

		for (int i = 0; i < num_reactions; ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute rate and order
			int order = 0;
			for (const auto [reactant_id, n_reactant_consumed] : Reaction.reactants) {
				// divide by factorial
				if (n_reactant_consumed != 1)
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
	eigen::Matrix<Float> order_1_dY_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates,
		Vector const &Y,
		const Float rho)
	{
		const int dimension = Y.size();
		eigen::Matrix<Float> M(dimension, dimension);

		const int num_reactions = reactions.size();
		if (num_reactions != rates.size())
			throw std::runtime_error("Number of reaction and rates don't match !\n");

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

			if (rate > constants::epsilon_system)
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
	template<class Vector1, class Vector2, typename Float>
	std::pair<Vector1, Float> solve_system_from_guess(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT, const Vector2 &BE, 
		const Vector1 &Y, const Float T, const Vector1 &Y_guess, const Float T_guess,
		const Float cv, const Float rho, const Float value_1, const Float dt)
	{
		if (rho < constants::min_rho || T < constants::min_temp)
			return {Y, T};

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

		Vector1 next_Y = Y;
		eigen::Matrix<Float> Mp(dimension + 1, dimension + 1);

		// right hand side
#ifdef USE_EIGEN
		eigen::Vector<Float> RHS(dimension + 1);
#else
		std::vector<Float> RHS(dimension + 1);
#endif
		auto dY_dt = derivatives_from_reactions(reactions, rates, Y_guess, rho);
		for (int i = 0; i < dimension; ++i)
			RHS[i + 1] = dY_dt[i]*dt;

		// main matrix part
		eigen::Matrix<Float> MpYY = order_1_dY_from_reactions(reactions, rates, Y_guess, rho);
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
		auto dY_dT = derivatives_from_reactions(reactions, drates_dT, Y_guess, rho);
		for (int i = 0; i < dimension; ++i) {
			Mp(i + 1, 0) = -constants::theta*dt*dY_dT[i];

			//               __*Dt = __*(next_T - T_guess) = __*(next_T - T + T - T_guess) = __*(next_T - T) - __*(T_guess - T)
			// <=> -__*theta*dt*Dt = ... - __*theta*dt*(T_guess - T)
			RHS[i + 1]  += -constants::theta*dt*dY_dT[i]*(T_guess - T);
		}

		// now solve M*D{T, Y} = RHS
		auto DY_T = eigen::solve(Mp, RHS, constants::epsilon_system);

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
		const Float cv, const Float rho, const Float value_1, const Float dt)
	{
		return solve_system_from_guess(reactions, rates, drates_dT, 
			BE,
			Y, T, Y, T,
			cv, rho, value_1, dt);
	}





	/// solve with  newton raphson
	/**
	 * iterative solver.
	 * ...TODO
	 */
	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float=double>
	std::tuple<Vector, Float, Float> solve_system_NR(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, Float T, const Float rho, const Float drho_dt, Float &dt)
	{
		if (rho < constants::min_rho || T < constants::min_temp)
			return {Y, T, dt};

		const int dimension = Y.size();

		Vector Y_theta = Y, final_Y = Y;
		Float T_theta = T, final_T = T;

		// actual solving
		for (int i = 0; i < constants::NR::max_it; ++i) {
			if (dt == 0) {
				std::string error = "Zero timestep in nuclear network\n";
				error += "\tT=" + std::to_string(T) + "\n";
				error += "\trho=" + std::to_string(rho) + "\n";
				error += "\tdrho_dt=" + std::to_string(drho_dt) + "\n";
				error += "\tY=";
				for (auto y : Y)
					error += std::to_string(y) + " ";
				error += "\n";
				
				throw std::runtime_error(error);
			}

			// compute n+theta values
			T_theta =        (1 - constants::theta)*T    + constants::theta*final_T;
			for (int j = 0; j < dimension; ++j)
				Y_theta[j] = (1 - constants::theta)*Y[j] + constants::theta*final_Y[j];

			// compute rate
			auto [rates, drates_dT] = construct_rates(         T_theta, rho);
			auto BE                 = construct_BE   (         T_theta, rho);
			auto eos_struct         = eos            (Y_theta, T_theta, rho);

			// compute value_1
			const double drho = drho_dt*dt;
			double value_1 = eos_struct.dP_dT*drho/(rho*rho);

			// solve the system
			Float last_T = final_T;
			std::tie(final_Y, final_T) = solve_system_from_guess(
				reactions, rates, drates_dT, BE,
				Y, T, Y_theta, T_theta,
				eos_struct.cv, rho, value_1, dt);

			// check for garbage 
			if (utils::contain_nan(final_Y, final_T) || final_T < 0) {
				// set timestep
				dt *= constants::nan_dt_step;

				// jump back
				final_Y = Y;
				final_T = T;
				i = -1;
				continue;
			}

			// break condition
			Float dT_T = std::abs((final_T - T)/final_T);
			if (i >= constants::NR::min_it && dT_T > constants::NR::dT_T_target*constants::NR::dT_T_tol) {
				// set timestep
				dt *= constants::NR::dT_T_target/dT_T;

				// jump back
				final_Y = Y;
				final_T = T;
				i = -1;
				continue;
			}

			// cleanup Vector
			utils::clip(final_Y, nnet::constants::epsilon_vector);
			
			// return condition
			Float correction = std::abs((final_T - last_T)/final_T);
			if (i >= constants::NR::min_it && correction < constants::NR::it_tol)
				break;
		}

		// mass and temperature variation
		Float dT_T = std::abs((final_T - T)/final_T);

		// timestep tweeking
		Float previous_dt = dt;
		dt = (dT_T == 0 ? (Float)constants::max_dt_step : constants::NR::dT_T_target/dT_T)*previous_dt;
		dt = std::min(dt, previous_dt*constants::max_dt_step);
		dt = std::max(dt, previous_dt*constants::min_dt_step);
		dt = std::min(dt,      (Float)constants::NR::max_dt);

		return {final_Y, final_T, previous_dt};
	}



	/// function to supperstep (can include jumping to NSE)
	/**
	 * Superstepping using solve_system_NR, might move it to SPH-EXA
	 * ...TODO
	 */
	template<class Vector, class func_rate, class func_BE, class func_eos, typename Float=double, class nseFunction=void*>
	std::tuple<Vector, Float> solve_system_substep(const std::vector<reaction> &reactions, const func_rate construct_rates, const func_BE construct_BE, const func_eos eos,
		const Vector &Y, const Float T,
		const Float rho, const Float drho_dt, Float const dt_tot, Float &dt,
		const nseFunction jumpToNse=NULL)
	{
		if (rho < constants::min_rho || T < constants::min_temp)
			return {Y, T};

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
			if ((dt_tot - elapsed_t)/dt_tot < constants::substep::dt_tol)
				return {final_Y, final_T};

			// timejump if needed
			if (jumpToNse != NULL)
				if (dt < dt_tot*constants::substep::dt_nse_tol) {
					dt = constants::max_dt;
					return jumpToNse(reactions, construct_rates, construct_BE, eos,
						final_Y, final_T,
						rho, drho_dt);
				}
		} 
	}
}
