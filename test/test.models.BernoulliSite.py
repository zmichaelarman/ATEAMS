
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


def testModel(cases, trials, rng):
	"""Checks the sampling, the filtration order, and the essential count."""
	print("\n3. MODEL -- sampling, filtration, essential count")
	ok = True

	for d, N, orientation in cases:
		k = d//2
		C = Permutohedral().fromScale(d, N, orientation=orientation, cutoff=k+1)
		M = BernoulliSite(C, dimension=k)
		M.RNG = np.random.default_rng(0)

		want = comb(d, k)
		densities = []
		malformed = 0
		wrongcount = 0

		for _ in range(trials):
			filtration, m, occupied = M._filtrate()
			densities.append(occupied.mean())

			# Occupied sites should all come before unoccupied ones.
			if not (occupied[filtration[:m]].all() and not occupied[filtration[m:]].any()):
				malformed += 1

			# The torus always has C(d,k) essential classes, so all should be found.
			values = filtration[M.padded].max(axis=1).astype(float)
			for s, v in zip(M.simplices, values): M.tree.assign_filtration(s, v)
			M.tree.compute_persistence(homology_coeff_field=M.field, persistence_dim_max=True)
			pairs = M.tree.persistence_intervals_in_dimension(k)

			if int(np.isinf(pairs[:,1]).sum()) != want: wrongcount += 1

		# Standard error of the density is 0.5/sqrt(sites*trials); allow five of
		# those. A fixed tolerance would fail the small tori.
		sites = len(C.Boundary[0])
		tolerance = 5 * 0.5 / np.sqrt(sites * trials)

		density = float(np.mean(densities))
		good = (malformed == 0 and wrongcount == 0 and abs(density - 0.5) < tolerance)
		ok &= good
		print(f"   d={d} N={N} {orientation:<8} density {density:.4f} (+/-{tolerance:.4f})  "
		      f"malformed {malformed}  wrong essential count {wrongcount}  "
		      f"{'ok' if good else 'FAIL'}")

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
		testModel(cases, trials, rng),
	]

	print()
	print("PASS" if all(results) else "FAIL")
	sys.exit(0 if all(results) else 1)
