
"""
Tests for the Permutohedral complex and the BernoulliSite model.
"""

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
import numpy as np
from math import comb
from itertools import combinations

from ateams.complexes import Permutohedral
from ateams.models import BernoulliSite


def rank(columns):
	"""
	Rank over GF(2). Columns are bitmasks, so XOR does the row reduction.
	"""
	pivots = {}
	r = 0

	for column in columns:
		c = column

		while c:
			p = c.bit_length() - 1

			if p in pivots: c ^= pivots[p]
			else:
				pivots[p] = c
				r += 1
				break

	return r


def nullspace(columns):
	"""
	Null space over GF(2), as bitmasks over column indices.
	"""
	pivots = {}
	null = []

	for j, column in enumerate(columns):
		c, tracked = column, 1 << j

		while c:
			p = c.bit_length() - 1

			if p in pivots:
				pc, pt = pivots[p]
				c ^= pc
				tracked ^= pt
			else:
				pivots[p] = (c, tracked)
				break

		if c == 0: null.append(tracked)

	return null


def columns(C, k):
	"""Columns of the kth boundary map, one bitmask per k-cell."""
	return [
		sum(1 << int(f) for f in faces) for faces in C.Boundary[k]
	]


def betti(C, k):
	"""
	kth Betti number: n_k - rank(d_k) - rank(d_k+1).
	"""
	n = len(C.Boundary[k])
	lower = rank(columns(C, k)) if k >= 1 else 0
	upper = rank(columns(C, k+1)) if (k+1) in C.Boundary else 0

	return n - lower - upper


def giants(C, k, occupied, cache):
	"""
	Giant k-cycles of the subcomplex on `occupied`, computed as
	rank([Z_k(occupied) | B_k(full)]) - rank(B_k(full)). `cache` holds the slow
	parts that don't change between configurations.
	"""
	if k not in cache:
		boundaries = columns(C, k+1)
		cache[k] = (boundaries, rank(boundaries), columns(C, k))

	boundaries, base, faces = cache[k]

	# k-cells with all their vertices occupied.
	alive = [j for j, s in enumerate(C.simplices[k]) if all(occupied[v] for v in s)]

	# Cycles among those cells.
	restricted = [faces[j] for j in alive]
	cycles = []

	for vector in nullspace(restricted):
		mask, i = 0, 0

		while vector:
			if vector & 1: mask |= 1 << alive[i]
			vector >>= 1
			i += 1

		cycles.append(mask)

	return rank(cycles + boundaries) - base


#############TESTS#####################################

def testComplex(cases):
	"""Betti numbers of the complex against the closed form C(d,k)."""
	print("1. COMPLEX -- Betti numbers vs closed form")
	ok = True

	for d, N, orientation in cases:
		top = d//2 + 1
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=top)

		for k in range(top):
			got, want = betti(C, k), comb(d, k)
			good = (got == want)
			ok &= good
			print(f"   d={d} N={N} {orientation:<8} betti_{k} = {got:<4} expected {want:<4} "
			      f"{'ok' if good else 'FAIL'}")

	return ok


def testDuality(cases, trials, rng):
	"""
	K(occupied) + K(vacant) = C(d,k). Only holds in this form when d = 2i, so even
	dimensions only.
	"""
	print("\n2. DUALITY -- K(occupied) + K(vacant) = C(d,k), per configuration")
	ok = True

	for d, N, orientation in cases:
		if d % 2:
			print(f"   d={d} skipped: only self-dual when d=2i")
			continue

		k = d//2
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=k+1)
		sites = len(C.Boundary[0])
		want, cache, bad = comb(d, k), {}, 0

		for _ in range(trials):
			occupied = rng.random(sites) < 0.5

			if giants(C, k, occupied, cache) + giants(C, k, ~occupied, cache) != want: bad += 1

		ok &= (bad == 0)
		print(f"   d={d} N={N} {orientation:<8} {trials} configs, violations = {bad:<4} "
		      f"{'ok' if bad == 0 else 'FAIL'}")

	return ok


def testModel(cases, trials):
	"""Checks the sampling and the filtration order."""
	print("\n3. MODEL -- sampling and filtration order")
	ok = True

	for d, N, orientation in cases:
		k = d//2
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=k+1)
		M = BernoulliSite(C, dimension=k)
		M.RNG = np.random.default_rng(0)

		low, high = int(C.breaks[k]), int(C.breaks[k+1])
		densities, malformed = [], 0

		for _ in range(trials):
			filtration, include = M._filtrate()
			m = include.shape[0]
			densities.append(m / len(C.simplices[k]))

			# Included cells come first, and the block boundaries are respected.
			block = set(int(x) for x in filtration[low:low+m])
			wanted = set(int(low + j) for j in include)

			if block != wanted: malformed += 1

		# Each cell needs all k+1 of its vertices, so the expected share of included
		# cells is p^(k+1). Cells share vertices, so a binomial error bar is too
		# tight. Take the spread from the trials and allow five standard errors.
		q = M.p ** (k+1)
		share = float(np.mean(densities))
		tolerance = 5 * float(np.std(densities)) / np.sqrt(trials)
		good = (malformed == 0 and abs(share - q) < tolerance)
		ok &= good
		print(f"   d={d} N={N} {orientation:<8} included {share:.4f} vs {q:.4f} "
		      f"(+/-{tolerance:.4f})  malformed {malformed}  {'ok' if good else 'FAIL'}")

	return ok


def testGiants(cases, trials, rng, ps=(0.3, 0.5, 0.7)):
	"""
	The giant count from the model against the independent GF(2) computation, on
	the same configurations. Also checks away from p=1/2, since duality fixes the
	mean at C(d,k)/2 there and a wrong read-out can still hit it.
	"""
	print("\n4. GIANTS -- model count vs independent GF(2), at several p")
	ok = True

	for d, N, orientation in cases:
		k = d//2
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=k+1)
		cache, bad = {}, 0

		for p in ps:
			M = BernoulliSite(C, p=p, dimension=k)
			M.RNG = np.random.default_rng(rng.integers(1 << 30))

			for _ in range(trials):
				filtration, include = M._filtrate()
				essential, __, _ = M.persist(filtration)
				stop = M.low + include.shape[0]
				got = int(len(essential[(M.low <= essential) & (essential < stop)]))

				occupied = np.zeros(len(C.Boundary[0]), dtype=bool)
				occupied[np.unique(C.simplices[k][include])] = True

				# Occupancy from the included cells, so both sides see the same
				# subcomplex.
				if giants(C, k, occupied, cache) != got: bad += 1

		ok &= (bad == 0)
		print(f"   d={d} N={N} {orientation:<8} {trials*len(ps)} configs, "
		      f"mismatches = {bad:<4} {'ok' if bad == 0 else 'FAIL'}")

	return ok


def testOrder(cases, rng, orderings=25):
	"""
	The count depends only on which cells are included, not on their order, so
	shuffling the included block must never change it.
	"""
	print("\n5. ORDER -- same cells, many orderings, same count")
	ok = True

	for d, N, orientation in cases:
		k = d//2
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=k+1)
		M = BernoulliSite(C, dimension=k)
		M.RNG = np.random.default_rng(0)
		worst = 1

		for _ in range(5):
			filtration, include = M._filtrate()
			m = include.shape[0]
			counts = set()

			for _ in range(orderings):
				f = filtration.copy()
				f[M.low:M.low+m] = rng.permutation(f[M.low:M.low+m])
				essential, __, _ = M.persist(f)
				counts.add(int(len(essential[(M.low <= essential) & (essential < M.low+m)])))

			worst = max(worst, len(counts))

		ok &= (worst == 1)
		print(f"   d={d} N={N} {orientation:<8} {orderings} orderings, distinct counts = "
		      f"{worst:<4} {'ok' if worst == 1 else 'FAIL'}")

	return ok


if __name__ == "__main__":
	try: trials = int(sys.argv[-1])
	except: trials = 25

	rng = np.random.default_rng(0)

	# Small tori, since the linear algebra is cubic.
	cases = [
		(2, 3, "rhombic"), (2, 4, "rhombic"), (2, 4, "square"),
		(3, 3, "rhombic"), (3, 3, "square"),
		(4, 3, "rhombic"),
	]

	results = [
		testComplex(cases),
		testDuality([(2, 4, "rhombic"), (2, 4, "square"), (4, 3, "rhombic")], trials, rng),
		testModel(cases, trials),
		testGiants([(2, 4, "rhombic"), (2, 4, "square"), (4, 3, "rhombic")], trials, rng),
		testOrder([(4, 3, "rhombic"), (4, 3, "square")], rng),
	]

	print()
	print("PASS" if all(results) else "FAIL")
	sys.exit(0 if all(results) else 1)
