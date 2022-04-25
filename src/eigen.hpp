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



	/// dot product function
	template<class vector>
	double dot(vector const &X, vector const &Y) {
		double res = 0;
		const int dimension = std::min(X.size(), Y.size());

		for (int i = 0; i < dimension; ++i)
			res += X[i]*Y[i];

		return res;
	}



	/// custom analytical solver
	template<class vector, class matrix>
	vector solve(matrix M, vector RHS) {
		const int n = RHS.size();
		vector X(n);

		// reduce into upper triangular
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				// include the jth line
				double weight = M(i, j);
				M(i, j) = 0;

				RHS[i] -= weight*RHS[j];
				for (int k = j + 1; k < n; ++k)
					M(i, k) -= weight*M(j, k);
			}

			// normalize ith line
			double diagonal = M(i, i);
			M(i, i) = 1;

			// normalize
			RHS[i] /= diagonal;
			for (int j = i + 1; j < n; ++j)
				M(i, j) /= diagonal;
		}

		// "back propagate" to solve
		for (int i = n - 1; i >= 0; --i) {
			double res = RHS[i];

			for (int j = i + 1; j < n; ++j)
				res -= M(i, j)*X[j];

			X[i] = res;
		}

		return X;
	}
}