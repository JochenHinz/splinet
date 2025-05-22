import numpy as np
from template import railroad_switch_template
from genus import MultiPatchBSplineGridObject, default_ids
from mapping.ko import KnotObject
from nutils import function, util
import itertools


def L():

  A = railroad_switch_template().stretch([1, 0], .5)
  edges = tuple(itertools.chain(*default_ids(A.patches)))

  kv = KnotObject(np.linspace(0, 1, 7))

  knotvectors = dict(zip(edges, [kv]*len(edges)))

  mg = MultiPatchBSplineGridObject(A.patches, A.patchverts, knotvectors=knotvectors)

  x, y = mg.geom

  s, p = function.stack, function.piecewise
  bottom = s([ p(x, [0.5], 0, 4 * (x - 0.5)), p(x, [0.5], 2*(1 - 2*x), 0) ])
  top = s([ p(x, [0.5], 1, 1 + 2 * (x - 0.5)), p(x, [0.5], 2 - 2*x, 1) ])
  left = s([ y, 2 ])
  right = s([ 2, y ])

  cons = util.NanVec((len(mg.basis)*2,))

  cons |= mg.project_edge((0, 1), left)
  cons |= mg.project_edge((1, 2), left)

  cons |= mg.project_edge((7, 8), right)
  cons |= mg.project_edge((8, 9), right)

  cons |= mg.project_edge((2, 5), top)
  cons |= mg.project_edge((5, 9), top)

  cons |= mg.project_edge((0, 3), bottom)
  cons |= mg.project_edge((3, 7), bottom)

  from mapping import sol

  mg.cons = cons
  mg.x = cons | 0

  import ipdb
  ipdb.set_trace()


if __name__ == '__main__':
  L()
