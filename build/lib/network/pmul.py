from index import IndexSequence, OrientableTuple
from template import MultiPatchTemplate
from mapping import mul, ko
import numpy as np
import itertools
from numba import njit
from numba.typed import List
from nutils import types, function, solver, topology, mesh
from nutils.topology import Patch, PatchBoundary, MultipatchTopology
from mapping.aux import unpack_GridObject
from mapping.go import GridObject


@unpack_GridObject
def Blechschmidt(g, *, domain, controlmap, basis, ischeme, gamma='gamma',
                       J=None, eta=1e2, btol=1e-5, tau='delta', solveargs=None, **ignorekwargs):

  solveargs = dict(solveargs or {})

  if J is None:
    J = function.J(controlmap)

  def _make_A(U):
    s = function.stack
    u, v = U[:, 0], U[:, 1]
    g11 = (u ** 2).sum()
    g12 = (u * v).sum()
    g22 = (v ** 2).sum()
    A = s([ s([g22, -g12]), s([-g12, g11]) ])  # + 1e-5 * function.eye(2)
    return A

  x = g.xArgument()

  A = _make_A(x.grad(controlmap))

  if gamma == 'gamma':
    gamma = function.trace(A) / (A**2).sum([0, 1])
  elif gamma == 'trace':
    gamma = 1 / (function.trace(A) + 1e-4)
  elif gamma is None:
    gamma = 1
  else:
    raise AssertionError

  Ax = function.stack([ (A * x[i].grad(controlmap).grad(controlmap)).sum([0, 1]) for i in range(2) ])

  n = controlmap.normal().normalized()
  vn = function.jump(basis.vector(2).grad(controlmap))[..., None] * n[None, None, None]
  xn = function.jump(x.grad(controlmap))[..., None] * n[None, None]

  if tau == 'delta':
    test = function.laplace(basis.vector(2), controlmap)
  elif tau == 'Id':
    test = basis.vector(2)
  else:
    raise AssertionError

  res = domain.integral( gamma * (test * Ax[None]).sum(1) * J, degree=10)

  try:
    res += eta * domain.interfaces['interpatch'].integral( (vn * xn[None]).sum([1, 2, 3]) * J, degree=10 )
  except Exception:  # for some reason this fails when I give him a singlepatch multipatch gridobject
    pass

  g.x = solver.newton('target', res, constrain=g.cons, lhs0=g.x, solveargs=solveargs).solve(btol)


def periodic_multipatch(domain, periodic_edges):

  directions = [1 for i in range(len(periodic_edges))]

  periodic_edges = list(map(sorted, periodic_edges))
  all_ids = set(bndry.id for patch in domain.patches for bndry in patch.boundaries)

  for i, pair in enumerate(periodic_edges):
    for j, edge in enumerate(map(tuple, pair)):
      if edge in all_ids: continue
      elif edge[::-1] in all_ids:
        directions[i] = -directions[i]
        periodic_edges[i][j] = edge[::-1]
      else:
        raise AssertionError

  map_periodic_edges = {}

  for direction, (edge0, edge1) in zip(directions, periodic_edges):
    map_periodic_edges[edge0] = (edge0, 1)
    map_periodic_edges[edge1] = (edge0, direction)

  patches = []

  for patch in domain.patches:
    patchboundaries = []
    for bndry in patch.boundaries:
      id, dim, side, reverse, transpose = bndry.id, bndry.dim, bndry.side, bndry.reverse, bndry.transpose
      if id in map_periodic_edges:
        id, direction = map_periodic_edges[id]
        if direction == -1:
          # XXX: this will probably only work in 2D
          reverse = tuple( not i for i in reverse[:-1] ) + reverse[-1:]
      patchboundaries.append(PatchBoundary(id=id, dim=dim, side=side, reverse=reverse, transpose=transpose))

    patches.append(Patch(patch.topo, patch.verts, patchboundaries))

  return MultipatchTopology(patches)


class PeriodicMultiPatchBSplineGridObject(mul.MultiPatchBSplineGridObject):

  def _lib(self):
    return {**super()._lib(), **{'periodic_edges': self._periodic_edges}}

  def __init__(self, patches, patchverts, knotvectors, periodic_edges, **kwargs):

    # XXX: add fail switches

    self._periodic_edges = tuple(sorted(map(lambda x: tuple(sorted(x)), periodic_edges)))
    super().__init__(patches, patchverts, knotvectors, **kwargs)

    # just overwrite self.domain by the one created from the uncoupled one
    self._domain = periodic_multipatch(self.domain, self._periodic_edges)
    GridObject.__init__(self, self.domain, self.geom)

  def make_basis(self, **kwargs):
    knotvectors = self._knotvectors
    knots = {key: value.knots for key, value in knotvectors.items()}
    knotmultiplicities = {key: value.knotmultiplicities for key, value in knotvectors.items()}
    kwargs.setdefault('degree', self._degree)
    kwargs.setdefault('knotvalues', knots)
    kwargs.setdefault('knotmultiplicities', knotmultiplicities)
    kwargs.setdefault('patchcontinuous', self._patchcontinuous)
    return self.domain.basis_spline(**kwargs)

  @property
  def periodic_edges(self):
    return self._periodic_edges

  def copy(self):
    return self.__class__(self.patches, self.patchverts, self._knotvectors, self._periodic_edges)

  def g_geom(self):
    ret = mul.MultiPatchBSplineGridObject(self.patches, self.patchverts, self._knotvectors,
                                          patchcontinuous=self.patchcontinuous)
    ret.x = ret.project(ret.geom)
    return ret

  def g_controlmap(self):
    ret = mul.MultiPatchBSplineGridObject(self.patches, self.patchverts, self._knotvectors,
                                          patchcontinuous=self.patchcontinuous)
    ret.x = ret.project(self.controlmap)
    return ret