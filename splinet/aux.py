import numpy as np
import itertools
from functools import wraps
from shapely import geometry, validation
from matplotlib import pyplot as plt
from nutils import function

from mapping import go

try:
  from nutils import log
except Exception:
  import treelog as log


def unit_vector(vector):
  """ Returns the unit vector of the vector.  """
  vector = np.asarray(vector)
  if vector.shape == (2,):
    return vector / np.linalg.norm(vector)
  assert vector.shape[1:] == (2,)
  return vector / np.linalg.norm(vector, axis=1)[:, None]


def angle_between(v1, v2):
  """ Returns the angle in radians between vectors 'v1' and 'v2'::

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
  """
  v1_u = unit_vector(v1)
  v2_u = unit_vector(v2)
  assert v1_u.shape == v2_u.shape
  if v1_u.shape == (2,):
    return np.arccos(np.clip((v1_u * v2_u).sum(), -1, 1))
  return np.arccos(np.clip((v1_u * v2_u).sum(1), -1, 1))


def abs_tuple(tpl):
  a, b = tpl
  if a > b: return b, a
  return tuple(tpl)


def frozen(arr: np.ndarray) -> np.ndarray:
  arr = np.asarray(arr)
  arr.flags.writeable = False
  return arr


def freeze(fn):
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return frozen(fn(*args, **kwargs))
  return wrapper


def _normalized(vec):
  if len(vec.shape) == 1:
    return vec / np.linalg.norm(vec, 2)
  return vec / (np.linalg.norm(vec, axis=1)[:, None])


def angle_between_vectors_(v0, v1):
  v0, v1 = np.asarray(v0), np.asarray(v1)
  scalar = len(v0.shape) == 1
  assert v0.shape == v1.shape
  if len(v0.shape) == 1:
    v0, v1 = v0[None], v1[None]
  assert len(v0.shape) == 2 and v0.shape[1] == 2
  v0 = _normalized(v0)
  v1 = _normalized(v1)
  dot = np.clip((v1 * v0).sum(1), -1, 1)
  v1_min_proj = v1 - dot[:, None] * v0
  cross = np.round(v0[:, 0] * v1_min_proj[:, 1] - v0[:, 1] * v1_min_proj[:, 0], 10)
  ret = np.sign(cross) * np.arccos(dot)
  if scalar:
    return ret[0]
  return ret


def angle_between_vectors(v0, v1, positive=False):
  v0 = _normalized(np.asarray(v0))
  v1 = _normalized(np.asarray(v1))
  scalar = len(v0.shape) == 1
  if scalar:
    v0, v1 = v0[None], v1[None]
  assert v1.shape == v0.shape
  smallest_angle = np.arccos(np.clip((v1 * v0).sum(-1), -1, 1))
  cross = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
  ret = smallest_angle * ( (-1) ** (cross < 0) )
  if scalar:
    ret = ret[0]
  if positive:
    return np.mod(ret, 2*np.pi)
  return ret


def compute_angles(list_of_curves):
  list_of_curves = [list_of_curves[-1]] + list(list_of_curves)
  angles = []
  for curve1, curve0 in zip(list_of_curves[1:], list_of_curves[:-1]):
    t1 = curve1.eval(0.001, der=1)
    t0 = curve0.eval(0.999, der=1)
    angles.append( angle_between_vectors(t0, t1) )

  return np.asarray(angles)


def parametric_length(knots):
  return np.abs(knots[1:] - knots[:-1]).sum()


def flip_KnotObject(k):

  knots = np.cumsum(np.concatenate([ [0], k.dx()[::-1] ]))
  knotmultiplicities = k.knotmultiplicities[::-1]

  from load import KnotVector

  return KnotVector(knotvalues=knots,
                    degree=k.degree,
                    knotmultiplicities=knotmultiplicities)


def flip_curve(_curve):
  curve = _curve.copy()
  knotvector = flip_KnotObject(curve.pop('knotvector'))
  x = curve.pop('x')[::-1]
  return curve.__class__(knotvector, x, **curve)


def flip_pointclouds_in_positive_direction(list_of_pointclouds, indices=None):

  if indices is not None:
    assert len(list_of_pointclouds) == len(indices) == len(set(indices))
    return_indices = True
  else:
    indices = np.arange(len(list_of_pointclouds))
    return_indices = False

  indices = np.asarray(indices)

  def _mindist(pc0, pc1):
    dist = []
    for n, m in itertools.product(*[[0, -1]] * 2):
      dist.append( np.linalg.norm(pc0[n] - pc1[m]) )
    return dist

  vals = {0: (0, 0), 1: (0, -1), 2: (-1, 0), 3: (-1, -1)}

  distances = _mindist(*list_of_pointclouds[:2])
  n, m = vals[np.argmin(distances)]

  if n == 0:
    list_of_pointclouds = [ pc[::-1] for pc in list_of_pointclouds ]
    indices = -indices

  ret = [list_of_pointclouds[0]]
  for i in range(1, len(list_of_pointclouds)):
    pc0 = ret[-1]
    pc1 = list_of_pointclouds[i]
    distances = _mindist(pc0, pc1)
    n, m = vals[np.argmin(distances)]
    if n == m:
      pc1 = pc1[::-1]
      indices[i] *= -1
    ret.append(pc1)

  if return_indices:
    return ret, indices
  return ret


def attach_pointclouds(list_of_pointclouds):

  def _mindist(pc0, pc1):
    dist = []
    for n, m in itertools.product(*[[0, -1]] * 2):
      dist.append( np.linalg.norm(pc0[n] - pc1[m]) )
    return dist

  indices = {0: (0, 0), 1: (0, -1), 2: (-1, 0), 3: (-1, -1)}

  ret = [list_of_pointclouds[0]]
  list_of_pointclouds = list_of_pointclouds[1:]
  while True:
    curve_now = ret[-1]
    alldistances = [ _mindist(curve_now, curve) for curve in list_of_pointclouds ]
    matching_index = np.argmin( list(map(min, alldistances)) )
    matching_curve = list_of_pointclouds[matching_index]
    n, m = indices[ np.argmin(_mindist(curve_now, matching_curve)) ]
    if n == m:
      ret.append( matching_curve[::-1] )
    else:
      ret.append( matching_curve )
    list_of_pointclouds = list_of_pointclouds[:matching_index] + list_of_pointclouds[matching_index + 1:]
    if len(list_of_pointclouds) == 0:
      break
  return ret


def curves2polygon(list_of_curves, npoints=1000):
  points = np.concatenate([ curve.eval(np.linspace(*curve.range(), npoints))[1:] for curve in list_of_curves ])
  points = np.concatenate([ points, points[0][None] ])
  pol = geometry.Polygon(points)
  if not pol.is_valid:
    string = validation.explain_validity(pol)
    coord = np.array(string[string.find('[')+1: string.find(']')].split(' '), dtype=float)
    plt.scatter(*coord)
    print(validation.explain_validity(pol))
    mulpol = validation.make_valid(pol)
    areas = [geom.area for geom in mulpol.geoms]
    pol = mulpol.geoms[np.argmax(areas)]
  plt.plot( *pol.exterior.coords.xy )
  return pol


def split_pointcloud_max_angle( list_of_pointclouds, return_indices=False ):
  from collections import deque
  list_of_pointclouds_ext = [list_of_pointclouds[-1]] + list(list_of_pointclouds)
  angles = []
  for curve1, curve0 in zip(list_of_pointclouds_ext[1:], list_of_pointclouds_ext):
    vec0 = curve0[-1] - curve0[-2]
    vec1 = curve1[1] - curve1[0]
    angles.append( angle_between_vectors(vec0, vec1) )

  # the angles that are closest to π
  # angles = np.abs(np.pi - np.asarray(angles))
  angles = np.abs(angles)
  angle_indices = np.sort(np.argsort(angles)[-4:])

  curves = deque(list_of_pointclouds)
  curves.rotate(-angle_indices[0])

  indices = deque(range(len(list_of_pointclouds)))
  indices.rotate(-angle_indices[0])
  indices = np.array_split(indices, angle_indices[1:] - angle_indices[0])
  pointclouds = [np.concatenate(pcs) for pcs in np.array_split(curves, angle_indices[1:] - angle_indices[0])]
  pointclouds = []
  for pcs in np.array_split(curves, angle_indices[1:] - angle_indices[0]):
    pointcloud = np.concatenate([_pc[:-1] for _pc in pcs] + [pcs[-1][-1:]])
    pointclouds.append(pointcloud)

  if return_indices:
    return pointclouds, indices

  return pointclouds


def split_pointcloud_max_angle_( points ):
  points_ext = np.concatenate([ points[-1:], points, points[:1] ], axis=0)
  angles = []
  # for curve1, curve0 in zip(list_of_pointclouds_ext[1:], list_of_pointclouds_ext):
  n = points_ext.shape[0]
  for i in range(1, n-1):
    vec0 = points_ext[i-1] - points_ext[i]
    vec1 = points_ext[i+1] - points_ext[i]
    angles.append( angle_between_vectors( vec0, vec1 ) )

  # the angles that are closest to π
  # angles = np.abs(np.pi / 2 - np.asarray(angles))
  angles = np.abs(angles)
  # indices = np.sort(np.argsort(angles)[-4:])
  indices = np.sort(np.argsort(angles)[:4])

  # curves = deque(list_of_pointclouds)
  # curves.rotate(-indices[0])

  curves = np.roll(points, -indices[0], axis=0)
  curves = np.concatenate([curves, curves[0][None]])
  indices = [0, *(indices[1:] - indices[:-1]).cumsum()]
  pcs = []
  for j, i in zip(indices[1:], indices):
    pcs.append( curves[i: j+1] )

  pcs += [curves[indices[-1]:]]

  return pcs


def create_corner(pointcloud, index, dist=1, width=100):
  assert index > width
  n = pointcloud.shape[0]
  scale = np.array([1, -1])
  _range = np.arange(index - width, index + width)
  vec0 = scale[None] * (pointcloud[_range % n] - pointcloud[(_range - 1) % n])[:, ::-1]
  vec0 = vec0 / (np.sqrt( (vec0**2).sum(1) ))[:, None]
  vec1 = scale[None] * (pointcloud[(_range + 1) % n] - pointcloud[_range % n])[:, ::-1]
  vec1 = vec1 / (np.sqrt( (vec1**2).sum(1) ))[:, None]

  n = (vec0 + vec1) / 2 * ((1 - np.abs( index - _range ) / width )[:, None])

  pointcloud[_range] += dist * n


def reverse_univariate_gridobject(g):
  assert g.ndims == 1
  assert isinstance(g, go.TensorGridObject)
  ret = g.__class__(g.knotvector.flip(), targetspace=g.targetspace)
  ret.x = g.x.reshape([-1, g.targetspace])[::-1].ravel()
  return ret


def straight_line(v0, v1, npoints=101):
  v0, v1 = np.asarray(v0), np.asarray(v1)
  assert v0.shape == v1.shape == (2,)

  xi = np.linspace(0, 1, npoints)[:, None]

  return v0[None] * (1 - xi) + v1[None] * xi


# operations on patches, patchverts, knotvector_edges in nutils format


def get_edges(list_of_vertices):
  assert len(list_of_vertices) == 4
  list_of_vertices = np.asarray(list_of_vertices).reshape([2, 2])
  return tuple(map(tuple, np.concatenate([list_of_vertices, list_of_vertices.T])))


opposite_side = {0: 1, 1: 0, 2: 3, 3: 2}


def get_edges_clockwise(patch):
  a, b, c, d = patch
  return ((a, c), (c, d), (d, b), (b, a))


def edge_neighbours(patches, edge):
  edge = tuple(edge)
  assert len(edge) == 2

  edges_per_patch = tuple(map(get_edges, patches))
  assert any( edge in patch_edges or edge[::-1] in patch_edges for patch_edges in edges_per_patch )

  neighbours = {edge}
  newneighbours = neighbours.copy()
  while True:
    for neigh in neighbours:
      for patchverts in edges_per_patch:
        reverse = False
        if neigh in patchverts:
          index = patchverts.index(neigh)
        elif neigh[::-1] in patchverts:
          index = patchverts.index(neigh[::-1])
          reverse = True
        else: continue
        newneighbours.update({tuple(patchverts[opposite_side[index]])[{False: slice(None),
                                                                       True: slice(None, None, -1)}[reverse]]})
    if len(newneighbours) == len(neighbours): break
    neighbours = newneighbours.copy()

  return tuple(newneighbours - {edge})


def infer_knotvector_edges(patches, knotvector_edges):
  assert all( len(patch) == 4 for patch in patches )
  assert set(itertools.chain(*knotvector_edges)).issubset(set(itertools.chain(*patches)))
  edges = list(knotvector_edges)

  while True:
    newedges = edges.copy()
    for edge in edges:
      myneighbours = edge_neighbours(patches, edge)
      for neighbour in myneighbours:
        if neighbour not in newedges:
          newedges.append(neighbour)
    if len(newedges) == len(edges): break
    edges = newedges

  assert all(edge[::-1] not in newedges for edge in newedges)

  return tuple(sorted(newedges))


def infer_knotvectors(patches, knotvectors):
  """ from a dictionary ``knotvectors`` of the form dict = {edge: knotvector}
      and a list of lists ``patches`` in standard nutils format, infer
      the knotvector corresponding to missing edges from the topology provided
      by ``patches`` and at it to knotvectors. If an edge is not missing but
      possesses a incompatible knotvector, raise and Error. """

  # make sure entries are in standard format
  newknotvectors = knotvectors.copy()

  # add missing entries
  while True:
    for edge, knotvector in knotvectors.items():
      myneighbours = edge_neighbours(patches, edge)  # all entries are in standard format
      for neighbour in myneighbours:
        otherknotvector = knotvectors.get(neighbour, None)
        if otherknotvector is None:
          newknotvectors[neighbour] = knotvector
          if neighbour[::-1] in newknotvectors:
            raise AssertionError
        else: assert otherknotvector == knotvector
    if len(newknotvectors) == len(knotvectors): break
    knotvectors = newknotvectors.copy()

  # check that entries are compatible
  for edge, knotvector in newknotvectors.items():
    myneighbours = edge_neighbours(patches, edge)
    for neighbour in myneighbours:
      if newknotvectors[neighbour] != knotvector:
        raise AssertionError('The knotvectors are incompatible.')

  return newknotvectors


def map_polygon_unit_disc(mg):
  center = mg.integrate(mg.geom * function.J(mg.geom), domain=mg.domain.boundary)
  center /= mg.integrate(function.J(mg.geom), domain=mg.domain.boundary)
  bedges = list(set(mg.to_template().ordered_boundary_edges[0]))

  bindices = list(set(itertools.chain(*bedges)))
  patchverts = np.asarray(mg.patchverts) - center[None]

  angles = {}
  for index in bindices:
    vec = patchverts[index]
    angles[index] = np.arctan2(*vec[::-1])

  intervals = [(angles[i], angles[j]) for (i, j) in bedges]
  breaks = np.array([interval[0] for interval in intervals])
  shuffle = np.argsort(breaks)
  breaks = breaks[shuffle]
  bedges = [bedges[i] for i in shuffle]
  ts = [patchverts[j] - patchverts[i] for (i, j) in bedges]
  ss = [ patchverts[i] for (i, j) in bedges ]
  snorms = [ np.linalg.norm(s) for s in ss ]
  TSs = [ np.stack([t, s], axis=1) for t, s in zip(ts, ss) ]
  Ts_inv = [ np.linalg.inv(A)[1] for A in TSs ]

  x = mg.controlmap - center
  distance_funcs = [ (c2 * x).sum() / snorm for c2, snorm in zip(Ts_inv, snorms) ]

  angle_func = function.arctan2(x[1], x[0])

  interval_func = function.piecewise(angle_func, breaks[1:], *[dfunc * (x + 1e-10).normalized() for dfunc in distance_funcs])

  return interval_func


def hermite_interpolation(v0, v1, t0, t1, npoints=101):
  v0, v1, t0, t1 = map(np.asarray, [v0, v1, t0, t1])
  assert v0.shape == v1.shape == t0.shape == t1.shape == (2,)
  xi = np.linspace(0, 1, npoints)[:, None]
  return v0[None] * (2 * xi**3 - 3*xi**2 + 1) + \
         t0[None] * (xi**3 - 2 * xi**2 + xi) + \
         v1[None] * (-2 * xi**3 + 3 * xi**2) + \
         t1[None] * (xi**3 - xi**2)


# vim:expandtab:foldmethod=indent:foldnestmax=2:sta:et:sw=2:ts=2:sts=2:foldignore=#
