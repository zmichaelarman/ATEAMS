
from ateams.complexes import Permutohedral
from ateams.models import BernoulliSite
from ateams import Chain
from pathlib import Path


def construct(L, DIM, ORIENTATION="rhombic"):
	# Construct complex object. Homology in dimension DIM//2 needs cells no higher
	# than DIM//2 + 1, so the skeleton stops there.
	homology = DIM//2
	fname = Path(f"./data/permutohedral.{ORIENTATION}.{L}.{DIM}.json")

	if not fname.exists():
		fname.parent.mkdir(exist_ok=True, parents=True)
		C = Permutohedral().fromScale(DIM, L, orientation=ORIENTATION, cutoff=homology+1)
		C.toFile(fname)
	else:
		C = Permutohedral().fromFile(fname)

	# Set up Model and Chain.
	BS = BernoulliSite(C, dimension=homology)
	N = 1000
	M = Chain(BS, steps=N)

	return M

def chain(M, DESC=""):
	for occupied, giants in M.progress(dynamic_ncols=True, desc=DESC):
		pass

	return M._exitcode

if __name__ == "__main__":
	M = construct(3,4)
	chain(M)
