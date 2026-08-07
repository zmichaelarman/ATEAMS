
import numpy as np
from math import comb


class BernoulliSite():
	_name = "BernoulliSite"

	def __init__(self, C, p=1/2, dimension=1, **kwargs):
		"""
		Initializes Bernoulli site percolation on the provided complex, detecting
		percolation in the `dimension`th homology group.

		Where `ateams.models.Bernoulli` includes cells directly, here each vertex is
		occupied with probability `p`, and a cell is included when all of its
		vertices are.

		Occupying vertices reorders cells of every dimension at once, which
		`ateams.arithmetic.ComputePersistencePairs` does not handle, so persistence
		is computed with GUDHI. GUDHI is imported inside this constructor rather
		than at the top of the file, so it is only required by this model.

		Args:
			C (Complex): The `Complex` object on which we'll be running experiments.
				Must expose the vertices of each cell, as
				`ateams.complexes.Permutohedral` does.
			p (float=1/2): The probability with which vertices are occupied.
			dimension (int=1): The dimension of cells whose homology we track.
		"""
		try: import gudhi
		except ImportError:
			raise ImportError(
				"BernoulliSite computes persistence with GUDHI; install it with "
				"`pip install gudhi`."
			)

		if not hasattr(C, "simplices"):
			raise ValueError(
				f"site percolation needs the vertices of each cell, which "
				f"{type(C).__name__} doesn't provide."
			)

		if dimension+1 not in C.simplices:
			raise ValueError(
				f"homology in dimension {dimension} needs cells of dimension "
				f"{dimension+1}, but this complex stops at {max(C.simplices)}."
			)

		# Object access. Set a field.
		self.complex = C
		self.dimension = dimension
		self._returns = 2
		self.field = 2
		self.p = p

		# Phantom spins attribute, required by the Chain.
		self.spins = None

		# Useful values to have later.
		self.siteCount = len(C.Boundary[0])
		self.cells = len(C.simplices[dimension])
		self.faces = len(C.simplices[dimension-1])
		self.rank = comb(len(C.corners), dimension)

		# Build the simplex tree once. Inserting the (dimension+1)-cells is enough,
		# since GUDHI fills in their faces.
		self.tree = gudhi.SimplexTree()

		for s in C.simplices[dimension+1]:
			self.tree.insert([int(v) for v in s])

		# Pad every simplex to a common width so the last-arriving vertex of all of
		# them is one numpy operation. Repeating the first vertex leaves the maximum
		# unchanged.
		self.simplices = [list(s) for s, _ in self.tree.get_simplices()]
		width = dimension+2
		self.padded = np.array(
			[s + [s[0]]*(width - len(s)) for s in self.simplices], dtype=np.int64
		)

		# Seed the random number generator.
		self.RNG = np.random.default_rng()


	def _filtrate(self, occupied=None):
		"""
		Constructs a filtration over the vertices: the occupied ones first, in a
		random order, then the rest.

		Args:
			occupied (np.array=None): Boolean array over vertices. Sampled if not
				provided.

		Returns:
			A triple: the filtration, the number of occupied vertices, and the
			occupancy itself.
		"""
		if occupied is None:
			uniform = self.RNG.uniform(size=self.siteCount)
			occupied = uniform < self.p

		include = np.nonzero(occupied)[0]
		exclude = np.nonzero(~occupied)[0]
		m = include.shape[0]

		self.RNG.shuffle(include)

		filtration = np.arange(self.siteCount)

		filtration[:m] = include
		filtration[m:] = exclude

		return filtration, m, occupied


	def _initial(self):
		"""
		Computes an initial state for the model's Complex.

		Returns:
			A numpy `np.array` of vertex occupancies.
		"""
		return (self.RNG.uniform(size=self.siteCount) < self.p).astype(np.uint8)


	def proposal(self, time):
		"""
		Proposal scheme for Bernoulli site percolation.

		Args:
			time (int): Step in the chain.

		Returns:
			A 2-tuple:

			1. a boolean array with a \(1\) in each entry corresponding to an
				occupied vertex, so its length is the number of vertices rather than
				the number of cells;
			2. times at which \(d\)-dimensional homological percolation events
				occurred.
		"""
		filtration, m, occupied = self._filtrate()

		# A cell arrives when its last vertex does.
		values = filtration[self.padded].max(axis=1).astype(float)

		for s, v in zip(self.simplices, values): self.tree.assign_filtration(s, v)

		self.tree.compute_persistence(
			homology_coeff_field=self.field, persistence_dim_max=True
		)

		# Giant cycles never die. Keep those born before the occupied vertices ran
		# out.
		pairs = self.tree.persistence_intervals_in_dimension(self.dimension)
		giants = filtration[np.where(np.isinf(pairs[:,1]))[0]]

		return occupied.astype(np.uint8), giants[giants < m]


	def _assign(self, cochain): pass
