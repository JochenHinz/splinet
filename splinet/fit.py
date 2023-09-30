from .index import Edge, as_KnotObject, SplineEdge
from .aux import get_edges, edge_neighbours, infer_knotvectors

from nutils import mesh, function
try:
  from nutils import log
except Exception:
  import treelog as log
import numpy as np
from mapping import std, ko, pc, go, gridop
from collections import defaultdict
from functools import reduce


def univariate_fit(pointcloud, knotvector=None, btol=1e-3, mu=None, maxiter=8, constrain_tangent=False):
  assert isinstance(pointcloud, pc.PointCloud)
  assert pointcloud.periodic is False, NotImplementedError

  points, verts = pointcloud.points, pointcloud.verts
  t0, t1 = (points[1] - points[0]) / (verts[1] - verts[0]), \
           (points[-1] - points[-2]) / (verts[-1] - verts[-2])

  if knotvector is None:
    knotvector = std.KnotObject

  assert isinstance(knotvector, ko.KnotObject)
  assert maxiter > 0

  if mu is None:
    mu = btol * 1e-2

  import scipy
  from scipy import sparse
  from nutils import matrix

  domain, geom = mesh.rectilinear([knotvector.knots])
  basis = domain.basis('spline',
                       degree=knotvector.degree,
                       knotvalues=[knotvector.knots],
                       knotmultiplicities=[knotvector.knotmultiplicities])

  cons = None
  for side, t in zip(('left', 'right'), (t0, t1)):
    _domain = domain.boundary[side]
    cons = _domain.project(pointcloud.points[{'left': 0,
                                              'right': -1}[side]],
                           onto=basis.vector(2),
                           geometry=geom,
                           ischeme='gauss6',
                           constrain=cons)
    if constrain_tangent:
      cons |= _domain.project(t,
                              onto=basis.vector(2).grad(geom)[..., 0],
                              geometry=geom,
                              ischeme='gauss6',
                              constrain=cons)

  sample = domain.locate(geom, pointcloud.verts[:, None], eps=1e-7)
  X = sparse.csr_matrix(sample.eval(basis))
  M = X.T @ X

  if mu > 0:
    log.info( 'project > Stabilizing via least-distance penalty method' )
    A = sparse.csr_matrix( domain.integrate(
                function.outer( basis.grad(geom) ).sum(-1) * function.J(geom),
                degree=2*knotvector.degree).export('dense') )
    log.info( 'project > Building stabilization matrix' )
    M = M + mu * A

  M = matrix.ScipyMatrix( sparse.block_diag([M] * 2).tocsr(), scipy=scipy )
  rhs = np.concatenate( [X.T.dot(pc) for pc in pointcloud.points.T] )
  cons = M.solve(rhs, constrain=cons.reshape([-1, 2]).T.ravel()).reshape([2, -1]).T.ravel()
  myfunc = basis.vector(2).dot(cons)

  func = sample.asfunction(pointcloud.points)
  dist = (sample.eval(func - myfunc) ** 2).sum(1)
  indices = np.where(dist > btol)[0]

  if maxiter == 1 and len(indices) != 0:
    log.warning('Failed to converge to the boundary tolerance')
    return knotvector, cons
  elif len(indices) == 0:
    log.warning('Convergence criterion reached.')
    return knotvector, cons
  knotvector = knotvector.ref_by_vertices(pointcloud.verts[indices])
  return univariate_fit(pointcloud, knotvector=knotvector, btol=btol,
                                    mu=mu, maxiter=maxiter-1,
                                    constrain_tangent=constrain_tangent)


def round_univariate_go(g, n=10):

  _g = g.__class__(g.knotvector.round([n]), targetspace=g.targetspace)
  _g.x = g.x

  return _g


def unify_univariate_boundary_gos(mg, dictionary_of_gos):
  """
  Prolong all univariate TensorGridObjects in dict: ``dictionary_of_gos``
  that correspond to equivalent sides in MultiPatchBSplineGridObject: ``mg``
  to a unified knotvector and return the result.
  """

  boundary_edges = set(mg.boundary_edges)
  assert all( isinstance(g, go.TensorGridObject) and g.ndims == 1 for g in dictionary_of_gos.values() )
  assert all( key in boundary_edges or key[::-1] in boundary_edges for key in dictionary_of_gos.keys() )

  dictionary_of_gos = {key: round_univariate_go(value) for key, value in dictionary_of_gos.items()}

  knotvectors = defaultdict(list)
  for edge, g in dictionary_of_gos.items():
    neighbours = [edge] + list(edge_neighbours(mg.patches, edge))
    for otheredge in neighbours:
      knotvectors[otheredge].append(g.knotvector)

  knotvectors = {key: reduce(lambda x, y: x + y, value) for key, value in knotvectors.items()}
  return {key: go.refine_GridObject(g, knotvectors[key]) for key, g in dictionary_of_gos.items()}


def multipatch_boundary_fit(mg, edges, **kwargs):
  """
  Fit a selection of input pointclouds to the (outer) edges of a multipatch
  gridobject by adaptively re-selecting the associated knotvectors until a
  convergence criterion has been reached.
  Parameters
  ----------
  mg: initial MultipatchBSplineGridObject
  edges: dictionary of edges of the form {edge_index: Edge} or {edge_index: PointCloud},
  Edge-inputs are transformed to chord-length parameterized PointCloud's.
  **kwargs: keyword-args forwarded to ``func: univariate_fit``.
  """
  boundary_edges = mg.boundary_edges
  edge_keys = set(list(edges.keys()))

  assert all( edge in edge_keys or edge[::-1] in edge_keys for edge in boundary_edges )
  assert all( edge[::-1] not in edge_keys for edge in edge_keys )
  assert all(isinstance(edge, (pc.PointCloud, Edge)) for edge in edges.values())

  _edges = {key: edge if isinstance(edge, pc.PointCloud) else edge.toPointCloud() for key, edge in edges.items()}
  edges = {}
  for key, _pc in _edges.items():
    if key in mg.knotvectors: edges[key] = _pc
    else: edges[key[::-1]] = _pc.flip()

  univariate_gos = {}
  for key, pointcloud in edges.items():
    knotvector = mg.knotvectors[key]
    newknotvector, x = univariate_fit(pointcloud, knotvector=knotvector, **kwargs)
    univariate_gos[key] = go.TensorGridObject(newknotvector, targetspace=2)
    univariate_gos[key].x = x

  return multipatch_boundary_fit_from_univariate_gos(mg, univariate_gos)


def multipatch_boundary_fit_from_univariate_gos(mg, dictionary_of_gos):
  """
  Given a dictionary of univariate gridobjects ``dictionary_of_gos`` of the form
  dictionary_of_gos = {edge: gridobject}, prolong all griobjects that correspond
  to equivalent sides under the patch layout of ``mg`` to one unified knotvector
  and build a new MultiPatchBSplineGridObject with mg's layout and the new knotvectors.
  Assign the corresponding control points to each side of the new mg.
  In case the inferred knotvectors associated with ``dictionary_of_gos`` is not exhaustive,
  use the knotvectors associated with ``mg``.
  """
  assert all( isinstance(g, go.TensorGridObject) and g.ndims == 1 for g in dictionary_of_gos.values() )
  dictionary_of_gos = unify_univariate_boundary_gos(mg, dictionary_of_gos)

  knotvectors = infer_knotvectors(mg.patches,
                                  {edge: g.knotvector[0] for edge, g in dictionary_of_gos.items()})

  if len(knotvectors) < len(mg.knotvectors):  # inference not exhaustive
    newknotvectors = dict(mg.knotvectors)
    for key in knotvectors.keys():
      newknotvectors.pop(key, None)
      newknotvectors.pop(key[::-1], None)
    knotvectors = {**knotvectors, **newknotvectors}

  ret = mg.__class__(patches=mg.patches, patchverts=mg.patchverts, knotvectors=knotvectors,
                     patchcontinuous=mg._patchcontinuous,
                     patch_identifiers=mg._patch_identifiers)

  cons = None
  for edge, g in dictionary_of_gos.items():
    patchindex, = [i for i, patchverts in enumerate(mg.patches) if set(edge).issubset(set(patchverts))]
    patchedges = get_edges(mg.patches[patchindex])
    if edge[::-1] in patchedges:
      edge = edge[::-1]
      _g = g
      g = _g.__class__(_g.knotvector.flip(), targetspace=_g.targetspace)
      g.x = _g.x.reshape([-1, 2])[::-1].ravel()
    index = patchedges.index(edge)
    side = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}[index]
    kv = ret.get_knotvector(edge)
    topo = ret.domain.patches[patchindex].topo.boundary[side]
    basis = topo.basis('spline', degree=kv.degree,
                                 knotvalues=[kv.knots],
                                 knotmultiplicities=[kv.knotmultiplicities])
    if g.knotvector[0] == kv: func = basis.vector(2).dot(g.x)
    elif g.knotvector[0] == kv.flip():
      func = basis.vector(2).dot(g.x.reshape([-1, 2])[::-1].ravel())
    else: raise
    cons = ret.project(func, domain=topo, constrain=cons)

  ret.cons = cons
  ret.x = cons | ret.x

  return ret


def all_edge_pairings(network):
  assert len(network.templates) == len(network.face_indices)
  pairings = defaultdict(list)
  for face_index, template in network.templates.items():
    mypairings = template.edge_pairings()
    for pair in mypairings:
      for p0, p1 in zip(pair, [pair[1], pair[0]]):
        pairings[tuple(p0)].append(tuple(p1))
        pairings[tuple(-p0)].append(tuple(-p1))
  return {key: tuple(value) for key, value in pairings.items()}


def compute_breakpoints(network):

  pairings = all_edge_pairings(network)
  sorted_keys = sorted(pairings.keys(), key=lambda x: len(x), reverse=True)

  breakpoints = defaultdict(set)
  newbreakpoints = breakpoints.copy()

  while True:
    for key in sorted_keys:
      mypairings = pairings[key]
      all_keys = (key,) + mypairings
      lengths = [ tuple(network.get_edges(edge).length for edge in myindices) for myindices in all_keys ]
      mybreakpoints = [ np.round(np.cumsum([0, *mylengths]) / total_length, 10) for mylengths, total_length in zip(lengths, map(sum, lengths)) ]
      existing_breakpoints = []
      for indices, breaks in zip(all_keys, mybreakpoints):
        for index, a, b in zip(indices, breaks, breaks[1:]):
          existing_breakpoints.append( tuple(a + bp * (b - a) for bp in breakpoints[index]) )
      all_breakpoints = set.union(*map(set, mybreakpoints)) | \
                        set.union(*map(lambda x: set(np.round(x, 10)), existing_breakpoints))
      for mysides, mybreakpoints in zip(all_keys, mybreakpoints):
        for i, side in enumerate(mysides):
          a, b = mybreakpoints[i: i+2]
          added_breakpoints = np.round([ (i - a) / (b - a) for i in filter(lambda x: a < x < b, all_breakpoints) ], 10)
          newbreakpoints[side].update(set(added_breakpoints))
          newbreakpoints[-side].update(set(np.round([1-i for i in added_breakpoints], 10)))
    if newbreakpoints == breakpoints: break
    breakpoints = newbreakpoints.copy()

  return {key: tuple(sorted(value)) for key, value in newbreakpoints.items() if key > 0}


def make_knotvectors(network, baseknotvector=None):
  baseknotvector = as_KnotObject(baseknotvector)
  assert isinstance(baseknotvector, ko.KnotObject)
  breakpoints = compute_breakpoints(network)
  ret = {}
  for index, mybreakpoints in breakpoints.items():
    if len(mybreakpoints) == 0:
      ret[index] = baseknotvector
    else:
      ret[index] = gridop.join_KnotObjects([baseknotvector]*(len(mybreakpoints)+1), mybreakpoints).round(5).to_c(1)
  return ret


def fit_hierarchically(network, baseknotvector=None):
  base_knotvectors = {}
  baseknotvector = as_KnotObject(baseknotvector)
  assert isinstance(baseknotvector, ko.KnotObject)

  pairings = all_edge_pairings(network)
  sorted_keys = sorted(pairings.keys(), key=lambda x: len(x), reverse=True)

  breakpoints = defaultdict(set)
  newbreakpoints = breakpoints.copy()

  while True:
    for key in sorted_keys:
      mypairings = pairings[key]
      all_keys = (key,) + mypairings

      for index in map(abs, key):
        if index not in base_knotvectors:
          base_knotvectors[index] = baseknotvector
          base_knotvectors[-index] = baseknotvector.flip()
          network.make_fit(index, kv=base_knotvectors[index])

      lengths = [ tuple(SplineEdge.from_gridobject(network.fits(edge)).length for edge in myindices) for myindices in all_keys ]
      mybreakpoints = [ np.round(np.cumsum([0, *mylengths]) / total_length, 10) for mylengths, total_length in zip(lengths, map(sum, lengths)) ]

      existing_breakpoints = []
      for indices, breaks in zip(all_keys, mybreakpoints):
        for index, a, b in zip(indices, breaks, breaks[1:]):
          existing_breakpoints.append( tuple(a + bp * (b - a) for bp in breakpoints[index]) )
      all_breakpoints = set.union(*map(set, mybreakpoints)) | \
                        set.union(*map(lambda x: set(np.round(x, 10)), existing_breakpoints))
      for mysides, mybreakpoints in zip(all_keys, mybreakpoints):
        for i, side in enumerate(mysides):
          a, b = mybreakpoints[i: i+2]
          added_breakpoints = np.round([ (i - a) / (b - a) for i in filter(lambda x: a < x < b, all_breakpoints) ], 10)
          newbreakpoints[side].update(set(added_breakpoints))
          newbreakpoints[-side].update(set(np.round([1-i for i in added_breakpoints], 10)))
    if newbreakpoints == breakpoints: break
    breakpoints = newbreakpoints.copy()

  return {key: tuple(sorted(value)) for key, value in newbreakpoints.items() if key > 0}
