#pragma once


#include <cmath> // factorial
//#include <ranges> // drop

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// solvers
#include <Eigen/SparseLU>
#ifndef SOLVER
	#define SOLVER Eigen::BiCGSTAB<Eigen::SparseMatrix<Float>>
	// #define SOLVER Eigen::SparseLU<Eigen::SparseMatrix<Float>, Eigen::COLAMDOrdering<int>> 
#endif



/* !!!!!!!!!!!!
debuging :
!!!!!!!!!!!! */
bool net14_debug = false;

namespace nnet {
	namespace constants {
		/// theta for the implicit method
		double theta = 0.7;
		/// tolerance of the implicit solver
		double implicit_tol = 1e-9;

		/// minimum timestep
		double min_dt = 1e-20;
		/// maximum timestep
		double max_dt = 1e-2;
		/// maximum timestep evolution
		double max_dt_step = 1.5;
		/// minimum timestep evolution
		double min_dt_step = 1e-2;

		/// relative temperature variation target of the implicit solver
		double dT_T_target = 4e-4;
		/// relative mass conservation target of the implicit solver
		double dm_m_target = 1e-5;
		/// relative temperature variation tolerance of the implicit solver
		double dT_T_tol = 1e-3;
		/// relative mass conservation tolerance of the implicit solver
		double dm_m_tol = 1e-3;



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
		/// sparcify a matrix.
		/**
		 * retursn a "sparsified" version of Min, with non-zero elements having a higher absolute value thab epsilon.
		 * ...TODO
		 */
		template<typename Float, int n>
		Eigen::SparseMatrix<Float> sparsify(const Eigen::Matrix<Float, n, n> &Min, const Float epsilon) {
			std::vector<Eigen::Triplet<Float>> coefs;

			for (int i = 0; i < Min.cols(); ++i)
				for (int j = 0; j < Min.rows(); ++j)
					if (std::abs(Min(i, j)) > epsilon)
						coefs.push_back(Eigen::Triplet<Float>(i, j, Min(i, j)));

			Eigen::SparseMatrix<Float> Mout(Min.cols(), Min.rows());
			Mout.setFromTriplets(coefs.begin(), coefs.end());
			return Mout;
		}




		/// normalizes a system.
		/**
		 * normalizes each equation of a system represented by a matrix (M) and a Right Hand Side (RHS).
		 * ...TODO
		 */
		template<typename Float, int n, int m>
		void normalize(Eigen::Matrix<Float, n, n> &M, Eigen::Vector<Float, m> &RHS) {
			const int dimension = RHS.size();

			for (int i = 0; i < dimension; ++i) {

				// find maximum
				Float max_ = std::abs(M(0, i));
				for (int j = 1; j < dimension; ++j)
					max_ = std::max(max_, std::abs(M(j, i)));

				// normalize
				RHS[i] /= max_;
				for (int j = 0; j < dimension; ++j)
					M(j, i) /= max_;
			}
		}




		/// solves a system
		/**
		 * solves a system represented by a matrix (M) and a Right Hand Side (RHS).
		 * ...TODO
		 */
		template<typename Float, int n, int m>
		Eigen::Vector<Float, m> solve(Eigen::Matrix<Float, n, n> &M, Eigen::Vector<Float, m> &RHS, const Float epsilon) {
			// sparcify M
			auto sparse_M = utils::sparsify(M, epsilon);

			// now solve M*X = RHS
			SOLVER solver;
			solver.compute(sparse_M);
			return solver.solve(RHS);
		}




		/// clip the values in a vector
		/**
		 * clip the values in a vector, to make 0 any negative value, or values smaller than a tolerance epsilon
		 * ...TODO
		 */
		template<typename Float, int n>
		void clip(Eigen::Vector<Float, n> &X, const Float epsilon) {
			const int dimension = X.size();

			for (int i = 0; i < dimension; ++i)
				if (std::abs(X(i)) <= epsilon)
					X(i) = 0;
		}
	}






	/// create a first order system from a list of reaction.
	/**
	 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
	 * ...TODO
	 */
	template<typename Float>
	Eigen::Matrix<Float, -1, -1> order_0_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, Eigen::Vector<Float, -1> const &Y, const Float rho) {
		const int dimension = Y.size();

		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

		for (int i = 0; i < reactions.size() && i < rates.size(); ++i) {
			const reaction &Reaction = reactions[i];
			Float rate = rates[i];

			// compute order and correct for rho
			int order = 0;
			for (auto const [_, n_reactant_consumed] : Reaction.reactants)
				order += n_reactant_consumed;
			rate *= std::pow(rho, order - 1);

			auto const [reactant_id, n_reactant_consumed] = Reaction.reactants[0];
			// compute rate
			Float this_rate = rate;
			this_rate *= std::pow(Y(reactant_id), n_reactant_consumed - 1);
			for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
				if (other_reactant_id != reactant_id)
					this_rate *= std::pow(Y(other_reactant_id), other_n_reactant_consumed);

			// insert consumption rates
			for (const auto [other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
				M(other_reactant_id, reactant_id) -= this_rate*other_n_reactant_consumed;

			// insert production rates
			for (auto const [product_id, n_product_produced] : Reaction.products)
				M(product_id, reactant_id) += this_rate*n_product_produced;
		}

		return M;
	}





	/// create a first order system from a list of reaction.
	/**
	 * creates a first order system from a list of reactions represented by a matrix M such that dY/dt = M*Y.
	 * ...TODO
	 */
	template<typename Float>
	Eigen::Matrix<Float, -1, -1> order_1_dY_from_reactions(const std::vector<reaction> &reactions, const std::vector<Float> &rates, Eigen::Vector<Float, -1> const &Y, const Float rho) {
		const int dimension = Y.size();

		Eigen::Matrix<Float, -1, -1> M = Eigen::Matrix<Float, -1, -1>::Zero(dimension, dimension);

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
				this_rate *= std::pow(Y(reactant_id), n_reactant_consumed - 1);
				for (auto &[other_reactant_id, other_n_reactant_consumed] : Reaction.reactants)
					if (other_reactant_id != reactant_id)
						this_rate *= std::pow(Y(other_reactant_id), other_n_reactant_consumed);

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




	/// solves a system non-iteratively.
	/**
	 *  solves non-iteratively and partialy implicitly the system represented by M.
	 * ...TODO
	 */
	template<class vector, typename Float>
	std::tuple<vector, Float, /* debugging */Float/* debugging */> solve_system(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const vector &BE, const vector &Y,                         /* debugging */const vector &A/* debugging */, 
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

		// construct matrix
		Eigen::Matrix<Float, -1, -1> Mp = Eigen::Matrix<Float, -1, -1>::Zero(dimension + 1, dimension + 1);

		// main matrix part
		auto MpYY = order_1_dY_from_reactions(reactions, rates, Y, rho);
		Mp(Eigen::seq(1,dimension), Eigen::seq(1,dimension)) = MpYY;

		// right hand side
		auto M = order_0_from_reactions(reactions, rates, Y, rho);
		vector RHS(dimension + 1), dY_dt = M*Y;
		RHS(Eigen::seq(1, dimension)) = dY_dt*dt;
		RHS(0) = (T*value_1 + BE.dot(dY_dt))/cv*dt;

		// include Y -> T terms
		Mp(0, Eigen::seq(1, dimension)) = BE/cv;
		Mp(0, 0) = value_1/cv;

		// include rate derivative
		Eigen::Matrix<Float, -1, -1> dM_dT = order_0_from_reactions(reactions, drates_dT, Y, rho);
		Mp(Eigen::seq(1, dimension), 0) = dM_dT*Y;


		// !!!!!!!!!!!!!!!!!!
		// debug:
		double tol = 1e-4;
		Float d1 = (MpYY.transpose()*A).sum();
		Float d2 = (M.transpose()*A).sum();
		Float d3 = (dM_dT.transpose()*A).sum();
		if (std::abs(d1) > tol || std::abs(d2) > tol || std::abs(d3) > tol)
			std::cout << "Mp.T*A=" << d1 << ", M.T*A=" << d2 << ", (dM/dT).T*A=" << d3 << "\n";

		if (std::abs(d1) > 1 || std::abs(d2) > 1 || std::abs(d3) > 1) {
			std::cout << "\n";

			for (int i = 0; i < reactions.size(); ++i)
				std::cout << "rate=" << rates[i] << ", drate/dt=" << drates_dT[i] << "\n";

			std::cout << "\nT=" << T << ", dt=" << dt << "\n";
			std::cout << "Y=" << Y.transpose() << "\n\n";

			std::cout << "Mp=\n" << Mp << "\n\nM=\n" << M << "\n\ndM_dT=\n" << dM_dT << "\n\n";
			std::cout << "Mp.T*A=" << (MpYY.transpose()*A).transpose() << "\nM.T*A=" << (M.transpose()*A).transpose() << "\n(dM/dT).T*A=" << (dM_dT.transpose()*A).transpose() << "\n"; 
			std::cout << "(dM/dT)*Y=" << (dM_dT*Y).transpose() << "\n\n";

			throw;
		}

		// !!!!!!!!!!!!!!!!!!
		// debug:
		if (net14_debug) {
			std::cout << "Mp=\n" << Mp << "\n\nM=\n" << M << "\n\ndM_dT=\n" << dM_dT << "\n\n";
			std::cout << "Mp.T*A=" << (MpYY.transpose()*A).transpose() << "\nM.T*A=" << (M.transpose()*A).transpose() << "\n(dM/dT).T*A=" << (dM_dT.transpose()*A).transpose() << "\n"; 
			std::cout << "(dM/dT)*Y=" << (dM_dT*Y).transpose() << "\n\n";
		}


		// construct M
		Eigen::Matrix<Float, -1, -1> M_sys = Eigen::Matrix<Float, -1, -1>::Identity(dimension + 1, dimension + 1) - constants::theta*Mp*dt;

		// normalize
		//utils::normalize(M_sys, RHS);

		// now solve M*D{T, Y} = RHS
		vector DY_T = utils::solve(M_sys, RHS, constants::epsilon_system);

		// add values
		vector next_Y = Y + DY_T(Eigen::seq(1, dimension));
		Float next_T = T + DY_T(0);
		return {next_Y, next_T, /* debugging */DY_T(Eigen::seq(1, dimension)).dot(A)/* debugging */};
	}




	/// fully solves a system, with timestep tweeking
	/**
	 *  solves iteratively and fully implicitly a single iteration of the system constructed by construct_system, with added timestep tweeking
	 * ...TODO
	 */
	template<class vector, class vector_int, typename Float>
	std::tuple<vector, Float, Float, /* debugging */Float/* debugging */> solve_system_var_timestep(const std::vector<reaction> &reactions, const std::vector<Float> &rates, const std::vector<Float> &drates_dT,
		const vector &BE, const vector_int &A, const vector &Y,
		const Float T, const Float cv, const Float rho, const Float value_1, Float &dt) {
		const Float m_in = Y.dot(A);

		// actual solving
		while (true) {
			// solve the system
			auto [next_Y, next_T, /* debugging */dm/* debugging */] = solve_system(reactions, rates, drates_dT, BE, Y, /* debugging */A/* debugging */, T, cv, rho, value_1, dt);

			// minimum value
			Float min_coef = next_Y.minCoeff();

			// !!!!!!!!!!!!!!!!!!
			// debug:
			// if (min_coef < -constants::epsilon_vector) std::cout << min_coef << "=min_coef\n";

			// !!!!!!!!!!!!!!!!!!
			// debug: (mass variation)
			// dm = 0;
			/* if (min_coef < 0) 
				for (int i = 0; i < Y.size(); ++i) {
					Float ya = next_Y(i)*A(i);
					if (ya < 0)
						dm -= ya;
				}/**/

			// cleanup vector
			utils::clip(next_Y, nnet::constants::epsilon_vector);

			// mass and temperature variation
			Float dm_m = std::abs(1 - next_Y.dot(A)/m_in);
			Float dT_T = std::abs((next_T - T)/((1 - constants::theta)*T + constants::theta*next_T));

			// timestep tweeking
			Float actual_dt = dt;
			Float dt_multiplier = 
				std::min(
					std::min(
						dT_T ==                                 0 ? (Float)constants::max_dt_step : constants::dT_T_target/dT_T,
						dm_m ==                                 0 ? (Float)constants::max_dt_step : constants::dm_m_target/dm_m
					),
					min_coef >= -(Float)constants::epsilon_vector ? (Float)constants::max_dt_step : -constants::epsilon_vector/min_coef
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
			(dm_m <= constants::dm_m_tol && dT_T <= constants::dT_T_tol && min_coef >= -nnet::constants::epsilon_vector))
				return {next_Y, next_T, actual_dt, /* debugging */dm/* debugging */};
		}
	}
}
