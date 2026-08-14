
import numpy as np

from .Bernoulli import Bernoulli


class BernoulliSite(Bernoulli):
	_name = "BernoulliSite"

	def __init__(self, C, p=1/2, dimension=1, **kwargs):
		"""
		Initializes Bernoulli site percolation on the provided complex, detecting
		percolation in the `dimension`th homology group.

		Where `ateams.models.Bernoulli` gives each cell its own trial, here each
		vertex is occupied with probability `p`, and a cell is included when all of
		its vertices are. The rest of the model is inherited, since the giant cycles
		depend only on which cells are included, not on the order they arrived in.

		Args:
			C (Complex): The `Complex` object on which we'll be running experiments.
				Must expose the vertices of each cell, as
				`ateams.complexes.Permutohedral` does.
			p (float=1/2): The probability with which vertices are occupied.
			dimension (int=1): The dimension of cells whose homology we track.
		"""
		if not hasattr(C, "simplices"):
			raise ValueError(
				f"site percolation needs the vertices of each cell, which "
				f"{type(C).__name__} doesn't provide."
			)

		super().__init__(C, p=p, dimension=dimension, **kwargs)

		# Vertices of each cell, and the number of vertices.
		self.cellVertices = C.simplices[dimension]
		self.siteCount = len(C.Boundary[0])


	def _filtrate(self):
		"""
		Constructs a filtration. Occupies vertices, then includes the cells whose
		vertices are all occupied.

		Returns:
			A pair: the filtration, and the indices of the included cells.
		"""
		occupied = self.RNG.uniform(size=self.siteCount) < self.p
		present = occupied[self.cellVertices].all(axis=1)

		include = np.nonzero(present)[0]
		exclude = np.nonzero(~present)[0]
		m = include.shape[0]

		filtration = np.arange(self.cellCount)

		filtration[self.low:self.low+m] = self.target[include]
		filtration[self.low+m:self.high] = self.target[exclude]

		return filtration, include
