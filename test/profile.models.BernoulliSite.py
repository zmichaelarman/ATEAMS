
import warnings
warnings.simplefilter("ignore", UserWarning)

from Profile import profile
import BernoulliSite
import sys


try:
    DIM = int(sys.argv[-2])
    L = int(sys.argv[-1])
    sparse = False
except:
    L = 3
    DIM = 4

M = BernoulliSite.construct(L, DIM)

TESTS = [L, DIM, M.model.faces]
WIDTHS = [8, 8, 8]
DESC = [str(thing).ljust(width) for thing, width in zip(TESTS, WIDTHS)]
DESC = " ".join(DESC)
DESC = ("      "+DESC).ljust(30)

profile(BernoulliSite.chain, [M,DESC], f"./profiles/BernoulliSite/{L}.{DIM}.txt")
