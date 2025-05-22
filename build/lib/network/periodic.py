from template import MultiPatchTemplate
import numpy as np


def railroad_switch():
  patchverts = [ [0, 0], [0, .5],
                 [0, 1], [1, 0],
                 [.5, .5], [1, 1],
                 [1.5, .5], [2, 0],
                 [2, .5], [2, 1] ]

  patches = [ [0, 1, 3, 4],
              [1, 2, 4, 5],
              [3, 4, 6, 5],
              [3, 6, 7, 8],
              [6, 5, 8, 9] ]

  knotvector_edges = (0, 3), (0, 1), (1, 2), (3, 7)

  template = MultiPatchTemplate(patches, patchverts, knotvector_edges)
  periodic_edges = ((0, 1), (7, 8)), ((1, 2), (8, 9))

  return template, periodic_edges


def double_railroad_switch():

  template = railroad_switch()[0]

  template = template.fuse(template.translate((2, 0)))

  return template, [((0, 1), (14, 15)), ((1, 2), (15, 16))]


if __name__ == '__main__':

  from pmul import PeriodicMultiPatchBSplineGridObject, Blechschmidt
  from mapping import ko, sol

  A, pedges = double_railroad_switch()
  A = A.stretch((1, 0), .25)

  kv = ko.KnotObject(np.linspace(0, 1, 7))
  knotvectors = dict(zip(A.knotvector_edges, [kv]*len(A.knotvector_edges)))

  from nutils import function

  mg = PeriodicMultiPatchBSplineGridObject(A.patches, A.patchverts, knotvectors, pedges)
  print(len(mg.basis))
  x, y = mg.geom

  def circle(R):
    return lambda gr: R * (1 + 0.1 * function.sin(10*np.pi*(gr[0] + .125))) * \
                          function.stack([ function.cos(2*np.pi*(gr[0] + .125)),
                                           function.sin(2*np.pi*(gr[0] + .125))] )

  R1, R2 = 2, 1
  func = (1 - mg.geom[1])*circle(R1)(mg.geom) + mg.geom[1]*circle(R2)(mg.geom)
  mg.x = mg.project(func)
  mg.set_cons_from_x()

  mg.qplot()

  # f = mg.g_geom()
  # basis_patch = f.domain.basis_patch()
  # prefunc = basis_patch.dot([1, 1, 5, 1, 1, 1, 1, 5, 1, 1])
  from control import multipatch_trace_penalty

  # f = multipatch_trace_penalty(f, prefunc=prefunc)
  # f.qplot()
  # mg.controlmap = f.mapping
  sol.Blechschmidt(mg)

  import ipdb
  ipdb.set_trace()
