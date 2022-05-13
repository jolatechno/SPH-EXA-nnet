#pragma once

#include <vector>

namespace eigen {
	/// custom matrix type
	template<typename Type>
	class matrix {
	private:
		std::vector<Type> weights;

	public:
		int n, m;

		matrix(int n_, int m_) : n(n_), m(m_) {
			weights.resize(n*m, 0);
		}

		Type &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type operator()(int i, int j) const {
			return weights[i + j*n];
		}
	};



	/// custom fixed-size matrix type
	template<typename Type, int n, int m>
	class fixed_size_matrix {
	private:
		std::vector<Type> weights;

	public:
		fixed_size_matrix() {
			weights.resize(n*m, 0);
		}

		Type &operator()(int i, int j) {
			return weights[i + j*n];
		}

		Type operator()(int i, int j) const {
			return weights[i + j*n];
		}
	};



	/// dot product function
	template<class Vector1, class Vector2>
	double dot(Vector1 const &X, Vector2 const &Y) {
		double res = 0;
		const int dimension = std::min(X.size(), Y.size());

		for (int i = 0; i < dimension; ++i)
			res += X[i]*Y[i];

		return res;
	}



	/// custom analytical solver
	template<typename Float, class Vector>
	Vector solve(matrix<Float> M, Vector RHS) {
		const int n = RHS.size();
		Vector X(n);

		// reduce into upper triangular
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				// include the jth line
				Float weight = M(i, j);
				M(i, j) = 0;

				RHS[i] -= weight*RHS[j];
				for (int k = j + 1; k < n; ++k)
					M(i, k) -= weight*M(j, k);
			}

			// normalize ith line
			Float diagonal = M(i, i);
			M(i, i) = 1;

			// normalize
			RHS[i] /= diagonal;
			for (int j = i + 1; j < n; ++j)
				M(i, j) /= diagonal;
		}

		// "back propagate" to solve
		for (int i = n - 1; i >= 0; --i) {
			Float res = RHS[i];

			for (int j = i + 1; j < n; ++j)
				res -= M(i, j)*X[j];

			X[i] = res;
		}

		return X;
	}
}