
import json
import numpy as np

from pathlib import Path
from fractions import Fraction
from itertools import combinations, permutations, product
from collections import deque

from ..common import MINT
from .Cubical import flatten
from .construction import boundaryMatrices, fullBoundaryMatrix


class Matrices:
	boundary = None
	coboundary = None
	full = None


_SQUAREGENERATORS = {
	2: [[1, -1], [1, 1]],
	3: [[-1, -1, 0], [-1, 0, -1], [0, -1, -1]],
	4: [[1, -1, 0, 0], [1, 1, -1, -1], [0, 0, 1, -1], [1, 1, 1, 1]],
}
"""
Lattice vectors spanning an axis-aligned box for \(A^*_d\), checked at runtime by
the Betti numbers of the resulting torus. Only \(d = 2, 3, 4\) are filled in.
"""


def directionDeltas(d):
	"""
	The \(d+1\) steps tracing out a simplex: the \(d\) unit-axis steps, plus one
	that decrements every coordinate. They sum to zero, which lets the simplices
	tile.

	Args:
		d (int): Dimension of the lattice.

	Returns:
		A list of \(d+1\) integer NumPy arrays.
	"""
	deltas = [np.eye(d, dtype=int)[k] for k in range(d)]
	deltas.append(-np.ones(d, dtype=int))

	return deltas


def squareRepetitions(d, N):
	"""
	How many times to tile each axis so the torus is about \(N\) wide in every
	direction. Shorter axes get tiled more. Clamped at two so no direction is too
	thin to hold the topology.

	Args:
		d (int): Dimension of the lattice.
		N (int): Requested scale.

	Returns:
		An integer NumPy array of \(d\) repetition counts.
	"""
	B = np.array(_SQUAREGENERATORS[d], dtype=float)
	Q = np.diag(B @ (np.eye(d) - 1.0 / (d + 1)) @ B.T)

	return np.maximum(2, np.rint(N * np.sqrt(d / ((d + 1) * Q))).astype(int))


def adjugate(M):
	"""
	Adjugate and determinant of a square integer matrix, computed over the
	rationals to avoid rounding error in the coset reduction below.

	Args:
		M (np.array): Square integer matrix.

	Returns:
		A pair: the adjugate, and the determinant.
	"""
	d = len(M)
	A = [
		[Fraction(int(M[i][j])) for j in range(d)] + [Fraction(int(i == k)) for k in range(d)]
		for i in range(d)
	]
	det = Fraction(1)

	for c in range(d):
		pivot = next(r for r in range(c, d) if A[r][c] != 0)

		if pivot != c:
			A[c], A[pivot] = A[pivot], A[c]
			det = -det

		det *= A[c][c]
		f = Fraction(1) / A[c][c]
		A[c] = [x * f for x in A[c]]

		for r in range(d):
			if r != c and A[r][c] != 0:
				g = A[r][c]
				A[r] = [A[r][j] - g * A[c][j] for j in range(2 * d)]

	inverse = [[A[i][d + j] for j in range(d)] for i in range(d)]
	determinant = int(det)
	adjugated = np.array(
		[[int(determinant * inverse[i][j]) for j in range(d)] for i in range(d)], dtype=np.int64
	)

	return adjugated, determinant


def squareReducer(M):
	"""
	Builds the map sending any lattice point to its representative inside the box
	spanned by `M`. All integer arithmetic, so the wrapping is exact.

	Args:
		M (np.array): Integer matrix whose columns span the box.

	Returns:
		A pair: the reduction function and the determinant (the number of points
		on the resulting torus).
	"""
	M = np.array(M, dtype=np.int64)
	adjugated, determinant = adjugate(M)

	if determinant < 0:
		adjugated, determinant = -adjugated, -determinant

	transposed, adjugatedT = M.T.copy(), adjugated.T.copy()

	def reduce(P):
		return P - ((P @ adjugatedT) // determinant) @ transposed

	return reduce, determinant


def delaunayRhombic(d, N):
	"""
	Traces every simplex on the lattice's own slanted torus, where wrapping is
	just mod \(N\).

	Args:
		d (int): Dimension of the lattice.
		N (int): Scale; the torus has \(N^d\) vertices.

	Returns:
		A set of frozensets, each holding the \(d+1\) corners of a simplex.
	"""
	deltas = directionDeltas(d)
	raw = set()

	for base in product(range(N), repeat=d):
		L = np.array(base, dtype=int)

		for permutation in permutations(range(d + 1)):
			vertices, c = [tuple(L % N)], L.copy()

			for step in permutation[:d]:
				c = c + deltas[step]
				vertices.append(tuple(c % N))

			s = frozenset(vertices)

			# Discard walks that wrapped onto themselves and collapsed.
			if len(s) == d + 1: raw.add(s)

	return raw


def delaunaySquare(d, N):
	"""
	Same as `delaunayRhombic`, but on an axis-aligned box so the torus lines up
	with the cubical one. Lattice points come from a flood fill, and wrapping uses
	the exact reducer instead of mod \(N\).

	Args:
		d (int): Dimension of the lattice.
		N (int): Requested scale.

	Returns:
		A set of frozensets, each holding the \(d+1\) corners of a simplex.
	"""
	if d not in _SQUAREGENERATORS:
		raise ValueError(
			f"square orientation is only specified for dimensions {sorted(_SQUAREGENERATORS)}; got {d}"
		)

	deltas = np.array([np.array(x, dtype=np.int64) for x in directionDeltas(d)])
	repetitions = squareRepetitions(d, N)
	M = np.array(_SQUAREGENERATORS[d], dtype=np.int64).T * repetitions[None, :]
	reduce, determinant = squareReducer(M)

	# Flood fill from the origin to collect every point of the torus.
	steps = np.vstack([deltas, -deltas])
	start = tuple(int(x) for x in reduce(np.zeros((1, d), dtype=np.int64))[0])
	seen, queue = {start}, deque([np.array(start, dtype=np.int64)])

	while queue:
		p = queue.popleft()

		for q in reduce(p[None, :] + steps):
			qt = tuple(int(x) for x in q)

			if qt not in seen:
				seen.add(qt)
				queue.append(q)

	cosets = np.array(sorted(seen), dtype=np.int64)
	raw = set()

	for permutation in permutations(range(d + 1)):
		current, columns = cosets.copy(), [cosets]

		for step in permutation[:d]:
			current = current + deltas[step]
			columns.append(reduce(current))

		for points in np.stack(columns, axis=1):
			s = frozenset(tuple(int(x) for x in v) for v in points)

			if len(s) == d + 1: raw.add(s)

	return raw


def delaunayTorus(d, N, orientation="rhombic"):
	"""
	Constructs the Delaunay triangulation of \(A^*_d\) on a \(d\)-torus, then
	relabels the coordinates as plain integers.

	Args:
		d (int): Dimension of the lattice.
		N (int): Scale.
		orientation (str="rhombic"): Either `"rhombic"` (the lattice's own
			slanted box) or `"square"` (an axis-aligned box).

	Returns:
		A triple: the sorted list of vertex coordinates, the sorted list of
		top-dimensional simplices as tuples of vertex indices, and the map
		sending coordinates to indices.
	"""
	raw = delaunaySquare(d, N) if orientation == "square" else delaunayRhombic(d, N)
	vertices = sorted({p for s in raw for p in s})
	vertexMap = {p: i for i, p in enumerate(vertices)}
	simplices = sorted(tuple(sorted(vertexMap[p] for p in s)) for s in raw)

	return vertices, simplices, vertexMap


def skeleton(simplices, top):
	"""
	Expands the top-dimensional simplices into all their faces, up to dimension
	`top`. Sorted faces keep the indexing reproducible, and make a \(0\)-cell's
	index equal to its vertex index.

	Args:
		simplices (list): Top-dimensional simplices as sorted tuples of vertex
			indices.
		top (int): Highest dimension to enumerate.

	Returns:
		A pair of lists indexed by dimension: the sorted faces, and dictionaries
		sending each face to its local index.
	"""
	faces = [set() for _ in range(top + 1)]

	for s in simplices:
		for k in range(top + 1):
			faces[k].update(combinations(s, k + 1))

	keys = [sorted(f) for f in faces]
	lookup = [{f: j for j, f in enumerate(k)} for k in keys]

	return keys, lookup


def boundaryFromSkeleton(keys, lookup, top):
	"""
	Builds boundary data from a skeleton. Each simplex lists its faces in the
	order the vertices get dropped, which gives the \(j\)th face the sign
	\((-1)^j\).

	Args:
		keys (list): Sorted faces per dimension, from `skeleton`.
		lookup (list): Face-to-index dictionaries, from `skeleton`.
		top (int): Highest dimension present.

	Returns:
		A dictionary sending each dimension to its boundary array, where entry
		\((j, m)\) is the index of the \(m\)th face of cell \(j\).
	"""
	Boundary = {0: np.arange(len(keys[0]), dtype=MINT)}

	for k in range(1, top + 1):
		previous = lookup[k - 1]
		cells = np.empty((len(keys[k]), k + 1), dtype=MINT)

		for j, face in enumerate(keys[k]):
			for m in range(k + 1):
				cells[j, m] = previous[face[:m] + face[m + 1:]]

		Boundary[k] = cells

	return Boundary


def vertexTable(Boundary, top):
	"""
	Vertices of every cell, from the boundary data. Face \(m\) is the simplex with
	vertex \(m\) dropped, so face \(0\) supplies all but the smallest vertex, and
	the smallest entry of face \(1\) supplies that one.

	Args:
		Boundary (dict): Boundary arrays, keyed by dimension.
		top (int): Highest dimension present.

	Returns:
		A dictionary sending each dimension \(k\) to an array of the vertex
		indices of each \(k\)-cell.
	"""
	simplices = {0: np.arange(len(Boundary[0]), dtype=MINT).reshape(-1, 1)}

	for k in range(1, top + 1):
		faces = Boundary[k]
		table = np.empty((faces.shape[0], k + 1), dtype=MINT)
		table[:, 1:] = simplices[k - 1][faces[:, 0]]
		table[:, 0] = simplices[k - 1][faces[:, 1]][:, 0]
		simplices[k] = table

	return simplices


def simplicialBoundaryMatrix(Boundary, D):
	"""
	Constructs the boundary and coboundary matrices in flat format, with the
	simplicial signs \((-1)^j\).

	Args:
		Boundary (dict): Boundary arrays, keyed by dimension.
		D (int): Dimension whose matrices we want.

	Returns:
		A pair of flat integer NumPy arrays.
	"""
	cells = np.ascontiguousarray(Boundary[D], dtype=MINT)
	coefficients = np.array([(-1) ** j for j in range(cells.shape[1])], dtype=MINT)
	matrices = boundaryMatrices(cells, coefficients)

	return np.array(matrices[0], dtype=MINT), np.array(matrices[1], dtype=MINT)


class Permutohedral:
	_name = "Permutohedral"

	def __init__(self): pass

	def fromScale(self, dimension, scale, orientation="rhombic", cutoff=None):
		"""
		Creates the Delaunay triangulation of \(A^*_d\) on a \(d\)-torus.

		Args:
			dimension (int): Dimension \(d\) of the lattice.
			scale (int): Scale of the torus; the rhombic torus has \(N^d\)
				vertices, while the square torus has as many as its box admits.
			orientation (str="rhombic"): Either `"rhombic"` or `"square"`.
			cutoff (int=None): Highest cell dimension to build. Homology in
				dimension \(i\) only needs cells up to \(i+1\), so passing
				`cutoff=i+1` saves a lot of memory. Defaults to the full skeleton.

		Returns:
			This `Permutohedral` object.
		"""
		return self._construct(dimension, scale, orientation, cutoff)


	def fromCorners(self, corners, periodic=True, orientation="rhombic", cutoff=None):
		"""
		Alias for `fromScale`, matching `ateams.complexes.Cubical`. These tori take a
		single scale rather than separate side lengths, so the corners must all be
		equal.

		Args:
			corners (list): Corners of the complex. All must be equal, and their
				count sets the dimension.
			periodic (bool=True): Only here to match `Cubical`; these are always
				tori.
			orientation (str="rhombic"): Either `"rhombic"` or `"square"`.
			cutoff (int=None): Highest cell dimension to construct.

		Returns:
			This `Permutohedral` object.
		"""
		corners = list(corners)

		if not periodic:
			raise ValueError("permutohedral complexes are always periodic")

		if len(set(corners)) != 1:
			raise ValueError(f"permutohedral tori take a single scale; got corners {corners}")

		return self._construct(len(corners), corners[0], orientation, cutoff)


	def _construct(self, dimension, scale, orientation, cutoff, data=None):
		self.dimension = dimension
		self.scale = scale
		self.orientation = orientation
		self.periodic = True
		self.corners = [scale] * dimension
		self.matrices = Matrices()

		cutoff = dimension if cutoff is None else min(cutoff, dimension)
		self.cutoff = cutoff

		if not data:
			_, simplices, vertexMap = delaunayTorus(dimension, scale, orientation)
			keys, lookup = skeleton(simplices, cutoff)

			self.vertexMap = vertexMap
			self.Boundary = boundaryFromSkeleton(keys, lookup, cutoff)
			self.simplices = {k: np.array(keys[k], dtype=MINT) for k in range(cutoff + 1)}
		else:
			self.vertexMap = data.get("vertexMap", None)
			self.Boundary = {
				int(t): data["Complex"][t].astype(MINT) for t in data["Complex"].files
			}
			data["Complex"].close()
			self.simplices = vertexTable(self.Boundary, cutoff)

		# Global indices and boundary matrices.
		self.reindexer, self.reindexed, self.flattened = flatten(self.Boundary, cutoff)

		if cutoff > 0:
			boundary, coboundary = simplicialBoundaryMatrix(self.Boundary, cutoff)
			self.matrices.boundary = boundary
			self.matrices.coboundary = coboundary

		# Simplices of every dimension use the same alternating signs, so one
		# pattern covers every key `fullBoundaryMatrix` looks up.
		alternating = np.array([(-1) ** j for j in range(cutoff + 2)], dtype=MINT)
		self.orientations = coefficients = {k: alternating for k in range((cutoff + 1) // 2 + 2)}
		self.matrices.full = np.array(
			fullBoundaryMatrix(self.flattened, coefficients), dtype=MINT
		)

		# Index ranges and percolation breaks.
		self.tranches = np.zeros((cutoff + 1, 2), dtype=int)
		self.tranches[0][1] = len(self.Boundary[0])

		for d in range(1, cutoff + 1):
			self.tranches[d] = [
				self.tranches[d - 1][1], self.tranches[d - 1][1] + len(self.Boundary[d])
			]

		self.breaks = np.array(self.tranches[:, 0])

		return self


	def recomputeBoundaryMatrices(self, dimension):
		"""
		Recomputes the boundary matrices in the provided dimension.

		Args:
			dimension (int): Dimension whose matrices we want.

		Returns:
			A pair of matrices --- the flat boundary and coboundary.
		"""
		return simplicialBoundaryMatrix(self.Boundary, dimension)


	def toFile(self, fp, vertexMap=False):
		"""
		JSON-serializes this object and writes it to file so we can reconstruct
		it later.

		Args:
			fp (str): Filepath.
			vertexMap (bool): Do we save the vertex map? Not doing so saves
				a lot of space.
		"""
		absolute = Path(fp).resolve()
		root = absolute.parent
		stem = absolute.stem
		ComplexFile = root / f".{stem}.complex.npz"
		np.savez(ComplexFile, **{str(t): v for t, v in self.Boundary.items()})

		with open(fp, "w") as write:
			json.dump(
				{
					"dimension": self.dimension,
					"scale": self.scale,
					"orientation": self.orientation,
					"cutoff": self.cutoff,
					"periodic": int(self.periodic),
					"corners": self.corners,
					"Complex": str(ComplexFile),
					"vertexMap": {
						str(k): int(v) for k, v in self.vertexMap.items()
					} if vertexMap and self.vertexMap else {}
				}, write
			)


	def fromFile(self, fp: str, vertexMap=False):
		"""
		Reconstructs a serialized `Permutohedral` complex.

		Args:
			fp (str): Filepath.
			vertexMap (bool): Is there a vertex map to load?
		"""
		from ast import literal_eval as le

		with open(fp, "r") as read:
			serialized = json.load(read)
			data = {"Complex": np.load(serialized["Complex"], allow_pickle=True)}

			if vertexMap:
				data["vertexMap"] = {le(k): int(v) for k, v in serialized["vertexMap"].items()}

			return self._construct(
				serialized["dimension"], serialized["scale"], serialized["orientation"],
				serialized["cutoff"], data
			)
