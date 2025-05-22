from index import IndexSequence, OrientableTuple
from template import MultiPatchTemplate
from mapping import mul, ko
import numpy as np
import itertools
from numba import njit
from numba.typed import List
from nutils import types, function, solver, topology
from mapping.aux import unpack_GridObject


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
    res += eta * domain.interfaces.integral( (vn * xn[None]).sum([1, 2, 3]) * J, degree=10 )
  except Exception:  # for some reason this fails when I give him a singlepatch multipatch gridobject
    pass

  g.x = solver.newton('target', res, constrain=g.cons, lhs0=g.x, solveargs=solveargs).solve(btol)


@njit(cache=True)
def apply_pairs(ndofs, list_of_pairs):
  ret = np.arange(ndofs)
  for pair in list_of_pairs:
    mask = np.empty( (len(pair),), dtype=np.int64 )
    for i in range(len(pair)):
      mask[i] = pair[i]
    _min = np.min( ret[mask] )
    ret[mask] = _min
  return ret


class PeriodicMultipatchTopology(topology.MultipatchTopology):

  @property
  def _patchinterfaces(self):
    patchinterfaces = super()._patchinterfaces
    for (edge0, ipatch0, side0), (edge1, ipatch1, side1) in self.periodic_pairs:
      edge = min(edge0, edge1)
      patchinterfaces.setdefault(edge, []).append((self.patches[ipatch0].topo, self.patches[ipatch0].topo.boundary[side0]))
      patchinterfaces[edge].append((self.patches[ipatch1].topo, self.patches[ipatch1].topo.boundary[side1]))
    return patchinterfaces


def basis_spline(self, degree, patchcontinuous=True,
                               knotvalues=None,
                               periodic_pairs=None,
                               knotmultiplicities=None, *, continuity=-1):
  '''spline from vertices
  Create a spline basis with degree ``degree`` per patch.  If
  ``patchcontinuous``` is true the basis is $C^0$-continuous at patch
  interfaces.
  '''

  if periodic_pairs is None:
    periodic_pairs = []

  # assert all(len(pair) == 3 for pair in periodic_pairs)

  # all patch indices that need to be mapped
  map_patch_id_and_side_direction = {}

  for i, (pair0, pair1) in enumerate(periodic_pairs):
    map_patch_id_and_side_direction.setdefault(pair0[0], []).append((i, *pair0[1:]))
    map_patch_id_and_side_direction.setdefault(pair1[0], []).append((i, *pair1[1:]))

  if knotvalues is None:
    knotvalues = {None: None}
  else:
    knotvalues, _knotvalues = {}, knotvalues
    for edge, k in _knotvalues.items():
      if k is None:
        rk = None
      else:
        k = tuple(k)
        rk = k[::-1]
      if edge is None:
        knotvalues[edge] = k
      else:
        l, r = edge
        assert (l, r) not in knotvalues
        assert (r, l) not in knotvalues
        knotvalues[(l, r)] = k
        knotvalues[(r, l)] = rk

  if knotmultiplicities is None:
    knotmultiplicities = {None: None}
  else:
    knotmultiplicities, _knotmultiplicities = {}, knotmultiplicities
    for edge, k in _knotmultiplicities.items():
      if k is None:
        rk = None
      else:
        k = tuple(k)
        rk = k[::-1]
      if edge is None:
        knotmultiplicities[edge] = k
      else:
        l, r = edge
        assert (l, r) not in knotmultiplicities
        assert (r, l) not in knotmultiplicities
        knotmultiplicities[(l, r)] = k
        knotmultiplicities[(r, l)] = rk

  missing = object()

  coeffs = []
  dofmap = []
  dofcount = 0
  commonboundarydofs = {}
  commonperiodicdofs = {}

  for ipatch, patch in enumerate(self.patches):
    # build structured spline basis on patch `patch.topo`
    patchknotvalues = []
    patchknotmultiplicities = []
    for idim in range(self.ndims):
      left = tuple(0 if j == idim else slice(None) for j in range(self.ndims))
      right = tuple(1 if j == idim else slice(None) for j in range(self.ndims))
      dimknotvalues = set()
      dimknotmultiplicities = set()
      for edge in zip(patch.verts[left].flat, patch.verts[right].flat):
        v = knotvalues.get(edge, knotvalues.get(None, missing))
        m = knotmultiplicities.get(edge, knotmultiplicities.get(None, missing))
        if v is missing:
          raise 'missing edge'
        dimknotvalues.add(v)
        if m is missing:
          raise 'missing edge'
        dimknotmultiplicities.add(m)
      if len(dimknotvalues) != 1:
        raise 'ambiguous knot values for patch {}, dimension {}'.format(ipatch, idim)
      if len(dimknotmultiplicities) != 1:
        raise 'ambiguous knot multiplicities for patch {}, dimension {}'.format(ipatch, idim)
      patchknotvalues.extend(dimknotvalues)
      patchknotmultiplicities.extend(dimknotmultiplicities)

    patchcoeffs, patchdofmap, patchdofcount = \
      patch.topo._basis_spline(degree, knotvalues=patchknotvalues,
                                       knotmultiplicities=patchknotmultiplicities,
                                       continuity=continuity)
    coeffs.extend(patchcoeffs)
    dofmap.extend(types.frozenarray(dofs+dofcount, copy=False) for dofs in patchdofmap)
    if patchcontinuous:
      # reconstruct multidimensional dof structure
      dofs = dofcount + np.arange(np.prod(patchdofcount), dtype=int).reshape(patchdofcount)
      for boundary in patch.boundaries:
        # get patch boundary dofs and reorder to canonical form
        import ipdb
        ipdb.set_trace()
        boundarydofs = boundary.apply_transform(dofs)[..., 0].ravel()
        # append boundary dofs to list (in increasing order, automatic by outer loop and dof increment)
        commonboundarydofs.setdefault(boundary.id, []).append(boundarydofs)

      id_side = map_patch_id_and_side_direction.get(ipatch, [])
      for (Id, side, direction) in id_side:
        boundary = patch.topo.boundary[side]
        commonperiodicdofs.setdefault(Id, []).append(
                dofs[{'bottom': (slice(None), 0),
                      'top'   : (slice(None), -1),
                      'left'  : (0,),
                      'right' : (-1,)             }[side]][ {-1: slice(None, None, -1), 1: slice(None)}[direction] ]
        )

    dofcount += np.prod(patchdofcount)

  if patchcontinuous:
    # build merge mapping: merge common boundary dofs (from low to high)
    pairs = itertools.chain(*(zip(*dofs) for dofs in itertools.chain(commonboundarydofs.values(), commonperiodicdofs.values()) if len(dofs) > 1))
    pairs = List(list(map(List, sorted(pairs))))
    merge = apply_pairs(dofcount, pairs)
    assert all(np.all(merge[a] == merge[b]) for a, *B in commonboundarydofs.values() for b in B), 'something went wrong is merging interface dofs; this should not have happened'
    # build renumber mapping: renumber remaining dofs consecutively, starting at 0
    remainder, renumber = np.unique(merge, return_inverse=True)
    # apply mappings
    dofmap = tuple(types.frozenarray(renumber[v], copy=False) for v in dofmap)
    dofcount = len(remainder)
  else:
    merge = np.arange(dofcount)

  return function.PlainBasis(coeffs, dofmap, dofcount, self.transforms)


class PeriodicMultiPatchBSplineGridObject(mul.MultiPatchBSplineGridObject):

  def _lib(self):
    return {**super()._lib(), **{'periodic_edges': self._periodic_edges}}

  def __init__(self, patches, patchverts, knotvectors, periodic_edges, **kwargs):

    template = MultiPatchTemplate(patches, patchverts, sorted(knotvectors.keys()))
    obe, = template.ordered_boundary_edges

    boundary_edges = abs(IndexSequence(obe, dtype=OrientableTuple))

    all_periodic_edges = IndexSequence(list(itertools.chain(*periodic_edges)), dtype=OrientableTuple)

    for edge in all_periodic_edges:
      assert edge @ boundary_edges

    periodic_pairs = []

    for edges in periodic_edges:
      pair = []
      for edge in map(OrientableTuple, edges):
        # get the boundary patch that contains edge
        ipatch, = set(template.edge2pols[abs(edge)]) & set(template.boundary_pols)

        # get corresponding edges in clockwise direction starting at bottom
        pol_edges = IndexSequence(template.get_edges(template.pol_indices[ipatch]), dtype=OrientableTuple)

        # get the local index of the edge
        local_index = abs(pol_edges).tuple_index(abs(edge))

        # map to the side
        side = {0: 'bottom', 1: 'right', 2: 'top', 3: 'left'}[local_index]

        # get the direction in which the coupling takes place.
        # From the clockwise direction, we get bottom, right are positively
        # and top, left negatively oriented.
        # if -edge is contained in pol_edges, multiply reverse orientation
        direction = {0: 1, 1: 1, 2: -1, 3: -1}[local_index] * (edge @ pol_edges)

        # pols are ordered from 1 - n + 1
        # while patches start couting at 0
        pair.append( (ipatch - 1, side, direction) )

      periodic_pairs.append(pair)

    # find and save all (ipatch, side) pairs that lie on the boundary
    # this excludes the pairs that coresponds to the edges in all_periodic_edges
    obe, = map(lambda x: IndexSequence(x, dtype=OrientableTuple), template.ordered_boundary_edges)
    bndry_ids = []
    for edge in set(abs(obe)) - set(abs(all_periodic_edges)):
      ipatch, = set(template.edge2pols[abs(edge)]) & set(template.boundary_pols)
      pol_edges = IndexSequence(template.get_edges(template.pol_indices[ipatch]), dtype=OrientableTuple)
      local_index = abs(pol_edges).tuple_index(abs(edge))
      side = {0: 'bottom', 1: 'right', 2: 'top', 3: 'left'}[local_index]
      bndry_ids.append( (ipatch - 1, side) )

    self._periodic_edges = tuple(sorted(map(sorted, periodic_edges)))
    self._periodic_pairs = tuple(map(tuple, periodic_pairs))
    self.template = template

    # for set_cons_from_x
    self.bndry_ids = tuple(bndry_ids)

    super().__init__(patches, patchverts, knotvectors, **kwargs)

  @property
  def ordered_boundary_indices(self):
    return self.template.ordered_boundary_indices

  @property
  def ordered_boundary_edges(self):
    return self.template.ordered_boundary_edges

  def make_basis(self, **kwargs):
    knotvectors = self._knotvectors
    knots = {key: value.knots for key, value in knotvectors.items()}
    knotmultiplicities = {key: value.knotmultiplicities for key, value in knotvectors.items()}
    kwargs.setdefault('degree', self._degree)
    kwargs.setdefault('knotvalues', knots)
    kwargs.setdefault('knotmultiplicities', knotmultiplicities)
    kwargs.setdefault('patchcontinuous', self._patchcontinuous)
    kwargs.setdefault('periodic_pairs', self.periodic_pairs)
    return basis_spline(self.domain, **kwargs)

  @property
  def boundary_indices(self):
    return self.template.boundary_indices

  @property
  def boundary_edges(self):
    return self.template.boundary_edges

  @property
  def periodic_edges(self):
    return self._periodic_edges

  @property
  def periodic_pairs(self):
    return self._periodic_pairs

  def set_cons_from_x(self):
    cons = None
    for ipatch, side in self.bndry_ids:
      cons = self.project(self.mapping, constrain=cons,
                                        domain=self.domain.patches[ipatch].topo.boundary[side])
    self.cons = cons


def test():
  from template import singlepatch_template
  # from sol import elliptic_partial, Blechschmidt
  A = singlepatch_template().refine((0, 1))

  kv = ko.KnotObject(np.linspace(0, 1, 14))
  knotvectors = dict(zip(A.knotvector_edges, [kv]*len(A.knotvector_edges)))

  from nutils import function

  mg = PeriodicMultiPatchBSplineGridObject(A.patches, A.patchverts, knotvectors, (((0, 2), (1, 3)),))
  x, y = mg.geom

  def circle(R):
    return lambda gr: R * (1 + 0.1 * function.sin(10*np.pi*gr[1])) * \
                          function.stack([ function.cos(2*np.pi*gr[1]),
                                           function.sin(2*np.pi*gr[1])] )

  R1, R2 = 1, 2
  func = (1 - mg.geom[0])*circle(R1)(mg.geom) + mg.geom[0]*circle(R2)(mg.geom)
  mg.x = mg.project(func)
  mg.set_cons_from_x()

  Blechschmidt(mg)

  import ipdb
  ipdb.set_trace()


if __name__ == '__main__':
  test()
