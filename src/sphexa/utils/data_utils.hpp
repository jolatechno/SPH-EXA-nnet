namespace sphexa {
	/// simple "iota" vector:
	/**
	 * "virtual" vector that respects the equation vect[i] = i + i0
	 */
	template<typename T>
	class iota_vector {
	private:
		size_t i0;

	public:
		// constructor
		template<typename Int=size_t>
		iota_vector(Int const i0_=0) : i0(i0_) {}

		template<typename Int=size_t>
		T operator[](const Int i) const {
			return i + i0;
		}
	};

	/// simple "constant" vector:
	/**
	 * "virtual" vector that respects the equation vect[i] = v0
	 */
	template<typename T>
	class const_vector {
	private:
		size_t v0;

	public:
		// constructor
		template<typename Int=size_t>
		const_vector(T const v0_=0) : v0(v0_) {}

		template<typename Int=size_t>
		T operator[](const Int i) const {
			return v0;
		}
	};

	/// vector for nuclear IO
	/**
	 * vector for nuclear IO,
	 * allows access by reference of nuclear data
	 * therby allowing "splitting" the vector of vector of nuclear species (Y) multiple vector, without memory overhead.
	 */
	template<int n_species, typename Float=double>
	class nuclear_IO_vector {
	private:
		std::vector<std::array<Float, n_species>> *Y;
		int species;

	public:
		// constructors
		nuclear_IO_vector() {};
		nuclear_IO_vector(std::vector<std::array<Float, n_species>> &Y_, const int species_) : Y(&Y_), species(species_) {}

		template<typename Int=size_t>
		auto operator[](const Int i) const {
			return (*Y)[i][species];
		}

		template<typename Int=size_t>
		auto at(const Int i) const {
			return Y->at(i)[species];
		}
	};
}