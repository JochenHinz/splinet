import numpy as np
from numbers import Number
from typing import Callable
from functools import cached_property, wraps, lru_cache
from itertools import chain, count
from scipy import integrate, optimize
from matplotlib import pyplot as plt
import treelog as log
from shapely.geometry import Polygon
from mapping.sol import transfinite_interpolation, forward_laplace
from nutils import function

from abc import abstractmethod
from copy import deepcopy

from .template import MultiPatchTemplate, get_edges, refine_patch
from .mesh import Triangulation
from .aux import frozen, convex_corner_pattern, unit_vector, abs_tuple, angle_between, EdgeNotFoundError, ex_ey
from .opt import default_strategy
from .index import as_edge
from .network import EdgeNetwork


def angle_2pi(v0, v1):
  v0 = unit_vector(v0)
  v1 = unit_vector(v1)
  angle = np.arctan2(np.cross(v1, v0), (v0 * v1).sum())
  if angle < 0:
    angle += 2 * np.pi
  return angle


def interior_angles(network):
  indices, = network.face_indices.values()
  edges = network.get_edges(indices)
  edges = edges[-1:] + edges
  ret = []
  for e0, e1 in zip(edges, edges[1:]):
    t0 = unit_vector(e0.points[-2] - e0.points[-1])
    t1 = unit_vector(e1.points[1] - e1.points[0])
    angle = np.arctan2(np.cross(t1, t0), (t0 * t1).sum())
    if angle < 0:
      angle += 2 * np.pi
    ret.append(angle)
  return np.asarray(ret)


def convex_corner_batches(network, concave_thresh=1e-2):
  face_index, = network.face_indices.keys()
  pattern = list(convex_corner_pattern(network, face_index, concave_thresh))
  if len(pattern) == 0:
    return (list(np.arange(len(network.face_indices[face_index]), dtype=int)),)
  elif len(pattern) == 1:
    batch = list(np.roll(np.arange(len(network.face_indices[face_index]), dtype=int), -pattern[0]))
    return (batch + batch[:1],)
  indices = np.arange(len(network.face_indices[face_index]), dtype=int)
  pattern = pattern + pattern[:1]
  return tuple(list(np.roll(indices, -i0)[:(i1 - i0) % len(indices) + 1]) for i0, i1 in zip(pattern, pattern[1:]))


_ = np.newaxis


METHODS = 'laplace', 'coons', 'local_laplace'
OPTIMIZE_CONTROLMAP = True


class BoundaryTransformation:
  """ Base class for boundary transformation. """

  def __init__(self, args):
    assert all(hasattr(arg, '__hash__') for arg in args)
    self.args = tuple(args)

  def evalf(self, geom):
    if isinstance(geom, Number):
      geom = np.array([geom])
    if len(geom.shape) == 0:
      geom = geom[_]
    return self._func(geom)

  def __call__(self, other):
    """ Compose with another function. Can either be `Callable` or self.__class__. """
    if isinstance(other, BoundaryTransformation):
      return BoundaryTransformation(lambda s: self.func(other.func(s)))
    assert isinstance(other, Callable)
    return BoundaryTransformation(lambda s: self.func(other(s)))

  def __hash__(self):
    return hash(self.args)

  def __eq__(self, other):
    if self.__class__ != other.__class__: return False
    return self.args == other.args

  @abstractmethod
  def _func(self):
    pass

  @cached_property
  def length(self):
    return integrate.quad(lambda x: self(x).ravel(), 0, 1)

  def _restrict(self, a, b):
    assert 0 <= a < b <= 1
    if a == 0 and b == 1:
      return self
    return Restriction(self, a, b)

  def restrict(self, a, b):
    """ Vectorised version of `_restrict`. """
    if isinstance(a, Number) and isinstance(b, Number):
      return self._restrict(a, b)
    a, b = map(np.asarray, (a, b))
    assert a.shape == b.shape and len(b.shape) == 1
    return tuple(map(self._restrict, a, b))

  def split(self, sbreak):
    return self.restrict(0, sbreak), self.restrict(sbreak, 1)

  def find_breakpoints(self, breaks, npoints=1001):
    breaks = np.asarray(breaks)
    assert len(breaks.shape) == 1
    assert np.logical_and((0 <= breaks).all(), (breaks <= 1).all())
    xi = np.linspace(0, 1, npoints)
    points = self.evalf(xi)
    verts = np.array([0, *np.linalg.norm(np.diff(points, axis=0), axis=1).cumsum()])
    verts /= verts[-1]
    return xi[np.argmin(np.abs(verts[:, _] - breaks[_]), axis=0)]

  def split_arclength(self, lbreak, npoints=1001):
    """ For now it's found numerically. """
    assert 0 < lbreak < 1
    sbreak, = self.find_breakpoints([lbreak], npoints=npoints)
    return self.split(sbreak)

  def split_by_length(self, lengths):
    verts = np.array([0, *lengths]).cumsum()
    verts /= verts[-1]
    breaks = self.find_breakpoints(verts)
    return self.restrict(breaks[:-1], breaks[1:])


class Restriction(BoundaryTransformation):

  def __init__(self, func, a, b):
    assert isinstance(func, BoundaryTransformation)
    assert 0 <= a < b <= 1
    self.function = func
    self.a, self.b = a, b
    super().__init__((func, a, b))

  def _func(self, geom):
    a, b = self.a, self.b
    return self.function._func(a + (b - a) * geom)


class Polynomial(BoundaryTransformation):

  def __init__(self, weights):
    self.weights = frozen(weights)
    assert weights.shape[1:] == (2,)
    self.powers = frozen(np.arange(len(self.weights)))
    super().__init__((self.weights.tobytes(),))

  def _func(self, geom):
    return ((geom[:, _] ** (self.powers[_]))[..., _] * self.weights[_]).sum(1)

  @cached_property
  def length(self):
    weights = (self.weights[1:] * np.arange(1, len(self.weights))[:, None])
    auxpol = Polynomial(weights)
    return integrate.quad(lambda x: np.sqrt((auxpol.evalf(x)**2).sum()), 0, 1)[0]


class Arc(BoundaryTransformation):

  def __init__(self, th0, th1, a=1, b=None):
    if b is None: b = a
    self.th0 = float(th0)
    self.th1 = float(th1)
    assert a > 0 and b > 0
    self.a, self.b = map(float, (a, b))
    super().__init__((self.a, self.b, self.th0, self.th1))

  def _func(self, geom):
    a, b, th0, th1 = self.a, self.b, self.th0, self.th1
    if isinstance(geom, function.Array):
      return function.stack([a * function.cos(th0 + (th1 - th0) * geom),
                             b * function.sin(th0 + (th1 - th0) * geom)], axis=1)
    return np.stack([a * np.cos(th0 + (th1 - th0) * geom),
                     b * np.sin(th0 + (th1 - th0) * geom)], axis=1)


class Translate(BoundaryTransformation):

  def __init__(self, func, v):
    assert isinstance(func, BoundaryTransformation)
    self.v = frozen(v)
    assert self.v.shape == (2,)
    self.function = func
    super().__init__((self.function, tuple(self.v)))

  def _func(self, geom):
    return self.function._func(geom) + self.v[_]


class Scale(BoundaryTransformation):

  def __init__(self, func, v):
    assert isinstance(func, BoundaryTransformation)
    self.v = frozen(v)
    assert self.v.shape == (2,)
    self.function = func
    super().__init__((self.function, tuple(self.v)))

  def _func(self, geom):
    return self.function._func(geom) * self.v[_]


def polynomial(weights):
  weights = frozen(weights.copy())
  return Polynomial(weights)


def linear(v0, v1):
  v0, v1 = map(np.asarray, (v0, v1))
  assert v0.shape == v1.shape == (2,)
  return polynomial(np.stack([v0, v1 - v0]))


def arc(th0, th1, **kwargs):
  return Arc(th0, th1, **kwargs)


def cubic_hermite(p0, t0, p1, t1):
  p0, t0, p1, t1 = map(np.asarray, (p0, t0, p1, t1))
  assert p0.shape == t0.shape == p1.shape == t1.shape == (2,)
  weights = [p0, t0, 3 * (p1 - p0) - 2 * t0 - t1, 2 * (p0 - p1) + t0 + t1]
  return polynomial(np.asarray(weights))


def _compute_template_edge_lengths(template):
  obe, = template.ordered_boundary_edges
  obi = [edge[0] for edge in obe]
  obi = obi + obi[:1]
  X = np.asarray(template.patchverts)[obi]
  return np.linalg.norm(np.diff(X, axis=0), axis=1)


def unit_disc(template: MultiPatchTemplate, lengths=None):
  if lengths is None:
    lengths = _compute_template_edge_lengths(template)
  obe, = template.ordered_boundary_edges
  assert len(lengths) == len(obe)
  funcs = arc(0, 2 * np.pi).split_by_length(lengths)
  return ControlMap.from_template(template, funcs)


def teardrop(template: MultiPatchTemplate, theta=None, origin=(0, 0), lengths=None, root=0):
  """
    theta: opening angle of the teardrop.
    root: the index of the template that is mapped onto the opening crease.
  """
  if theta is None:
    theta = np.pi/2
  assert 0 < theta < np.pi
  t0, t1 = [np.cos(theta / 2), np.sin(theta / 2)], [np.cos(theta / 2), -np.sin(theta / 2)]
  if lengths is None:
    lengths = _compute_template_edge_lengths(template)
  t0, t1, origin = map(np.asarray, (t0, t1, origin))
  assert t0.shape == t1.shape == origin.shape == (2,)
  obe, = template.ordered_boundary_edges
  assert len(lengths) == len(obe)
  
  # split by the length = [l0, l1, .., rootlength, ..., ln] -> [rootlength, ..., ln, l0, l1, ...]
  funcs = cubic_hermite(origin, t0, origin, t1).split_by_length(np.roll(lengths, -root))
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root))


@lru_cache
def _find_positions(list_of_lengths):
  """ Find positions on the unit circle such that the lengths of the edges drawn between
      the positions match the lengths in list_of_lengths divided by the total length.
  """
  assert len(list_of_lengths) >= 3
  assert all(length > 0 for length in list_of_lengths)

  f = lambda s: np.stack([np.cos(2 * np.pi * s), np.sin(2 * np.pi * s)], axis=1)

  L = np.asarray(list_of_lengths) / sum(list_of_lengths)

  forward_mask = (np.arange(1, len(L) + 1, dtype=int)) % len(L)

  def cost(s):
    fs = f(s)
    Ls = ((fs - fs[forward_mask])**2).sum(1)
    return .5 * ((Ls / Ls.sum() - L)**2).sum()

  cons = ({'type': 'ineq', 'fun': lambda s: np.concatenate([s[1:] - s[:-1] - 0.00001, [1 - s[-1] - 0.00001], [s[0]], [-s[0]], -s[1:] + 1, s[1:-1]])},)

  x0 = np.asarray([0, *L.cumsum()[:-1]])
  x = optimize.minimize(cost, x0, constraints=cons, tol=1e-10)
  log.warning("Succeeded in placing the polygon vertices on the unit circle: {}.".format(x.success))
  log.warning("Optimisation terminated with cost {}.".format(cost(x.x)))
  s = x.x

  points = f(s)

  return points


def find_positions(list_of_lengths):
  return _find_positions(tuple(list_of_lengths))


def polygon(template: MultiPatchTemplate, list_of_list_of_lengths=None, root=0):
  """
    root: start of the first batch.
    list_of_list_of_lengths = [[l0, l1, ...], ...] => root => l0, root + 1: l1, ...
  """
  if list_of_list_of_lengths is None:
    list_of_list_of_lengths = [[length] for length in _compute_template_edge_lengths(template)]
  obe, = template.ordered_boundary_edges
  assert sum(len(lengths) for lengths in list_of_list_of_lengths) == len(obe)
  total_lengths = tuple(sum(lengths) for lengths in list_of_list_of_lengths)
  points = find_positions(total_lengths)
  points = np.concatenate([points, points[:1]])
  funcs_packed = [linear(v0, v1) for v0, v1 in zip(points, points[1:])]
  funcs = list(chain(*[ func.split_by_length(lengths) for func, lengths in zip(funcs_packed, list_of_list_of_lengths) ]))
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root))


def lense(template: MultiPatchTemplate, width=1, langle=3 * np.pi/4, rangle=3 * np.pi/4, lengths=None, root0=0, root1=1):
  left = np.array([-width / 2, 0])
  right = np.array([width / 2, 0])
  if lengths is None:
    lengths = _compute_template_edge_lengths(template)
  obe, = template.ordered_boundary_edges
  assert len(lengths) == len(obe)
  lengths_top = np.roll(lengths, -root0)[:(root1 - root0) % len(obe)]
  lengths_bot = np.roll(lengths, -root0)[((root1 - root0) % len(obe)):]
  top = cubic_hermite(right, np.array([-np.cos(rangle/2), np.sin(rangle/2)]), left, -np.array([np.cos(langle/2), np.sin(langle/2)]))
  bottom = cubic_hermite(left, np.array([np.cos(langle/2), -np.sin(langle/2)]), right, np.array([np.cos(rangle/2), np.sin(rangle/2)]))
  funcs = top.split_by_length(lengths_top) + bottom.split_by_length(lengths_bot)
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root0))


def half_disc(template, lengths=None, root0=0, root1=1):
  """ root0: start of the lower edge,
      root1: end of the lower edge.
  """
  if lengths is None:
    lengths = _compute_template_edge_lengths(template)
  obe, = template.ordered_boundary_edges
  assert len(lengths) == len(obe)
  lengths_bot = np.roll(lengths, -root0)[:(root1 - root0) % len(obe)]
  lengths_top = np.roll(lengths, -root0)[((root1 - root0) % len(obe)):]
  L_bot, L_top = map(sum, (lengths_bot, lengths_top))
  bottom = linear([-1, 0], [1, 0])
  if L_top <= .5 * np.pi * L_bot:
    b = 1
  else:
    # the circumference is 2 * width(length) * ratio_top_bottom
    C = 4 * L_top / L_bot

    # find the root of Ramanujan's second approximation
    def f(b):
      h = ((b - 1) / (b + 1))**2
      return np.pi * (b + 1) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h))) - C

    b = optimize.root_scalar(f, bracket=[0.000001, 100], method='bisect')
    b = max(1, b.root)

  top = arc(0, np.pi, a=1, b=b)

  funcs = bottom.split_by_length(lengths_bot) + top.split_by_length(lengths_top)
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root0))


@lru_cache
def _find_acorn_parameters(l0, l1, l2):
  """
                P0 = (0, 1)
            t0  /\  t5
      edge0    /  \  edge2
          t1  /    \ t4
          p1 |      | p2
          t2  \____/  t3
              edge1

  batch1: batch that is placed at the edge2.
  root: first vertex index corresponding to the first vertex in the first batch. """
  assert all(li > 0 for li in (l0, l1, l2))
  total_length = l0 + l1 + l2
  relative_lengths = np.array([l0, l1, l2]) / total_length

  p0 = np.array([0, 1])

  def costfunc(c):
    t0, t1, p1, t2, t3, p2, t4, t5 = c.reshape(-1, 2)
    f0 = cubic_hermite(p0, t0, p1, t1)
    f1 = cubic_hermite(p1, t2, p2, t3)
    f2 = cubic_hermite(p2, t4, p0, t5)
    lengths = np.array([f1.length, f2.length, f0.length])
    return ((lengths / lengths.sum() - relative_lengths)**2).sum()

  def constraint(c):
    t0, t1, p1, t2, t3, p2, t4, t5 = c.reshape(-1, 2)
    angle0 = angle_2pi(p2 - p0, p1 - p0)
    angle1 = angle_2pi(p0 - p1, p2 - p1)
    angle2 = angle_2pi(p1 - p2, p0 - p2)
    return np.pi - np.array([angle0, angle1, angle2, angle_2pi(-t1, t2), angle_2pi(-t3, t4), angle_2pi(-t5, t0)])

  constraints = {'type': 'ineq', 'fun': constraint},

  x0 = np.array([ [-1, -1],  # t0
                  [-1, -1],  # t1
                  [-1, 0],  # p1
                  [1, -1],  # t2
                  [1, 1],  # t3
                  [1, 0],  # p2
                  [-1, 1],  # t4
                  [-1, 1] ]).ravel()  # t5

  opt = optimize.minimize(costfunc, x0, constraints=constraints)
  log.info("Acorn optimisation success: {}.".format(opt.success))
  return opt.x.reshape(-1, 2)


def acorn_(template, list_of_list_of_lengths=None, batch1=None, root=0):
  """
                /\
      edge0    /  \
              /    \
             |      |
              \____/
              edge1

  batch1: batch that is placed at the edge2.
  root: first vertex index corresponding to the first vertex in the first batch. """
  # XXX: implement a layout that chooses the acorn's dofs such that
  # angles / lengths are respected
  if list_of_list_of_lengths is None:
    list_of_list_of_lengths = [[length] for length in _compute_template_edge_lengths(template)]
  assert len(list_of_list_of_lengths) == 3
  list_of_list_of_lengths = tuple(list_of_list_of_lengths)
  if batch1 is None:  # take the longest batch with the largest number of elements by default
    total_elems = list(map(len, list_of_list_of_lengths))
    batch1 = np.argmax(total_elems)
  roll = (batch1 - 1) % 3
  list_of_list_of_lengths = list_of_list_of_lengths[roll:] + list_of_list_of_lengths[:roll]
  total_lengths = list(map(sum, list_of_list_of_lengths))
  p0 = [0, 1]
  t0, t1, p1, t2, t3, p2, t4, t5 = _find_acorn_parameters(*total_lengths)
  f0 = cubic_hermite(p0, t0, p1, t1)
  f1 = cubic_hermite(p1, t2, p2, t3)
  f2 = cubic_hermite(p2, t4, p0, t5)
  funcs = list(chain(*np.roll(np.array([ func.split_by_length(lengths) for func, lengths in zip((f0, f1, f2), list_of_list_of_lengths) ], dtype=object), roll)))  # undo the rolling
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root))


def acorn(template, list_of_list_of_lengths=None, batch1=None, root=0):
  """
                /\
      edge0    /  \
              /    \
             |      |
              \____/
              edge1

  batch1: batch that is placed at the edge2.
  root: first vertex index corresponding to the first vertex in the first batch. """
  # XXX: implement a layout that chooses the acorn's dofs such that
  # angles / lengths are respected
  if list_of_list_of_lengths is None:
    list_of_list_of_lengths = [[length] for length in _compute_template_edge_lengths(template)]
  assert len(list_of_list_of_lengths) == 3
  list_of_list_of_lengths = tuple(list_of_list_of_lengths)
  if batch1 is None:  # take the longest batch by default
    total_lengths = list(map(sum, list_of_list_of_lengths))
    batch1 = np.argmax(total_lengths)
  list_of_list_of_lengths = list_of_list_of_lengths[batch1:] + list_of_list_of_lengths[:batch1]
  total_lengths = list(map(sum, list_of_list_of_lengths))
  relative_length = total_lengths[0] / sum(total_lengths[1:])

  def costfunc(a):
    f = cubic_hermite([-1, 0], [1, *(-a)], [1, 0], [1, *a])
    return (f.length / (2 * np.sqrt(2)) - relative_length) ** 2

  constraints = {'type': 'ineq', 'fun': lambda x: x - 1e-1},
  a = optimize.minimize(costfunc, np.array([1]), constraints=constraints).x[0]

  f1 = cubic_hermite([-1, 0], [1, -a], [1, 0], [1, a])
  f2 = linear([1, 0], [0, 1])
  f0 = linear([0, 1], [-1, 0])
  funcs = list(chain(*np.roll(np.array([ func.split_by_length(lengths) for func, lengths in zip((f1, f2, f0), list_of_list_of_lengths) ], dtype=object), batch1)))  # undo the rolling
  return ControlMap.from_template(template, np.roll(np.array(funcs, dtype=object), root))


def pick_fitting_controlmap(template: MultiPatchTemplate, network, concave_thresh=1e-2, normalize=True):
  assert len(network.face_indices) == 1
  (face_index, indices), = network.face_indices.items()
  assert len(template.ordered_boundary_edges[0]) == len(indices)
  corner_pattern = np.sort(convex_corner_pattern(network, face_index, concave_thresh))
  lengths = np.array([edge.length for edge in network.get_edges(indices)])
  if len(corner_pattern) == 0:
    ret = unit_disc(template, lengths=lengths)
  elif len(corner_pattern) == 1:
    root, = corner_pattern
    # between 0 and pi
    theta = interior_angles(network)[root]
    assert theta > 0
    ret = teardrop(template, theta=theta, lengths=lengths, root=root)
  elif len(corner_pattern) == 2:
    root0, root1 = corner_pattern
    y = (root0 == (root1 + 1) % len(indices))
    if root1 == (root0 + 1) % len(indices) or y:
      if y: root0, root1 = root1, root0
      ret = half_disc(template, lengths=lengths, root0=root0, root1=root1)
    else:
      angles = interior_angles(network)
      rangle, langle = angles[root0], angles[root1]
      ret = lense(template, langle=langle, rangle=rangle, lengths=lengths, root0=root0, root1=root1)
  else:
    slices = [0, *(corner_pattern[1:] - corner_pattern[:-1]).cumsum(), len(indices)]  # [2, 3, 4] => [0, 1, 2, 5] => [0, 1], [1, 2], [2, 5]
    lengths = np.roll(lengths, -corner_pattern[0])  # the slices are defined w.r.t. starting vertex corner_patter[0]
    list_of_list_of_lengths = [ lengths[a: b] for a, b in zip(slices, slices[1:]) ]  # length batches. First batch starts at corner_pattern[0]
    root = corner_pattern[0]
    if len(corner_pattern) == 3:
      return acorn(template, list_of_list_of_lengths, root=root)  # don't pass batch1 because it's taking the longest by default. Undo the cyclic permutation though.
    ret = polygon(template, list_of_list_of_lengths=list_of_list_of_lengths, root=root)  # undo the cyclic permutation by shifting by + corner_pattern[0]
  if normalize:
    return ret.normalize()
  return ret


def as_BoundaryTransformation(arg):
  if isinstance(arg, BoundaryTransformation):
    return arg
  raise AssertionError


class ControlMap(MultiPatchTemplate):

  # XXX: Make this a subclass of MultiPatchTemplate

  @classmethod
  def from_template(cls, template, list_of_transforms=None):
    obi = [edge[0] for edge in template.ordered_boundary_edges[0]]
    obi = obi + obi[:1]
    X = np.asarray(template.patchverts)
    if list_of_transforms is None:
      list_of_transforms = [linear(X[i], X[j]) for i, j in zip(obi, obi[1:])]
      inner_verts = [X[i] for i in range(len(X)) if i not in obi]
    else:
      inner_verts = None
    return cls(template.patches, template.patchverts, list_of_transforms, knotvector_edges=template._knotvector_edges, inner_verts=inner_verts)

  def _lib(self):
    return {'patches': self.patches,
            'patchverts': self.patchverts,
            'knotvector_edges': self._knotvector_edges,
            'list_of_transforms': self._transforms,
            'inner_verts': self.inner_verts.copy()}

  def __init__(self, patches, patchverts, list_of_transforms, knotvector_edges=None, inner_verts=None):
    super().__init__(patches, patchverts, knotvector_edges)
    obe, = self.ordered_boundary_edges
    assert len(list_of_transforms) == len(obe)
    self._transforms = tuple(map(as_BoundaryTransformation, list_of_transforms))
    self.nedges = len(obe)
    self.ordered_boundary_indices = frozen([edge[0] for edge in self.ordered_boundary_edges[0]])
    self.dofindices = frozen(np.sort(np.setdiff1d(np.arange(len(self.patchverts)), self.ordered_boundary_indices)))
    if inner_verts is None:
      self.compute_vertices()
    else:
      self.inner_verts = frozen(inner_verts)
    assert len(self.inner_verts) == len(self.patchverts) - len(self.ordered_boundary_indices)
    assert self.inner_verts.shape[1:] == (2,)

  def sample_boundary(self, npoints, repeat_last=False):
    if isinstance(npoints, int):
      npoints = [npoints] * self.nedges
    assert len(npoints) == self.nedges
    return [func.evalf(np.linspace(0, 1, n+1)[:-1] if not repeat_last else np.linspace(0, 1, n)) for func, n in zip(self._transforms, npoints)]

  def refine_(self, edge, positions=.5):
    assert np.isscalar(positions), NotImplementedError
    edge = tuple(edge)
    if edge not in self.ordered_boundary_edges[0]:
      edge = edge[::-1]
      positions = 1 - positions

    if edge not in self.ordered_boundary_edges[0]:
      raise EdgeNotFoundError()

    boundary_edges = set(self.ordered_boundary_edges[0]) | set(edge[::-1] for edge in self.ordered_boundary_edges[0])
    i0 = self.ordered_boundary_edges[0].index(edge)

    template = self.to_MultiPatchTemplate()

    opposite_edge, = list(set(template.edge_neighbours(edge)) & boundary_edges)
    if opposite_edge not in self.ordered_boundary_edges[0]:
      opposite_edge = opposite_edge[::-1]

    i1 = self.ordered_boundary_edges[0].index(opposite_edge)
    if i1 < i0:
      i0, i1 = i1, i0
      positions = 1 - positions
      edge = self.ordered_boundary_edges[0][i0]

    f0, f1 = self._transforms[i0], self._transforms[i1]
    list_of_transforms = self._transforms[:i0] + f0.split_arclength(positions) + \
                         self._transforms[i0+1: i1] + f1.split_arclength(1 - positions) + \
                         self._transforms[i1+1:]

    opposites = (edge,) + template.edge_neighbours(edge)

    template = template.refine(edge, positions)
    newverts = set(range(len(self.patchverts), len(template.patchverts)))
    map_vert_neighbor = {}
    for patch in template.patches:
      for (i, j) in get_edges(patch):
        map_vert_neighbor.setdefault(i, set()).update({j})
        map_vert_neighbor.setdefault(j, set()).update({i})

    obe, = template.ordered_boundary_edges
    obi = np.asarray([edge[0] for edge in obe])
    X = self.X
    Xnew = list(X)
    for vert in sorted(newverts):
      myneighbors = tuple(map_vert_neighbor[vert] - newverts)
      if myneighbors not in opposites:
        myneighbors = myneighbors[::-1]
      assert myneighbors in opposites
      i, j = myneighbors
      Xnew.append((1 - positions) * X[i] + positions * X[j])

    inner_verts = np.stack(Xnew, axis=0)[np.sort(np.setdiff1d(np.arange(len(template.patchverts)), obi))]
    
    # XXX self.inner_vertices
    return ControlMap(template.patches, template.patchverts, list_of_transforms, knotvector_edges=template._knotvector_edges, inner_verts=inner_verts)

  def refine(self, edge, positions=.5):
    assert np.isscalar(positions), NotImplementedError
    if np.isscalar(positions):
      positions = positions,

    if edge[0] > edge[1]:
      edge = edge[::-1]
      positions = tuple(1 - pos for pos in positions)

    edge = tuple(edge)
    positions = sorted(set(map(lambda x: round(x, 5), positions)))

    assert all(0 < pos < 1 for pos in positions)

    if edge not in self.all_edges:
      raise EdgeNotFoundError()

    # get all sorted neighbours
    neighbours = tuple(map(tuple, [edge] + list(self.edge_neighbours(edge))))

    X = self.X
    patchverts = np.asarray(self.patchverts)

    def round_to_5(arr):
      return tuple(round(x, 5) for x in arr)

    map_vert_index = dict(zip(map(round_to_5, patchverts), range(len(patchverts))))
    map_x_index = dict(zip(map(round_to_5, X), range(len(X))))
    counter = count(len(X))

    edge2position = {edge: positions for edge in neighbours}
    edge2position.update({edge[::-1]: [1 - position for position in positions] for edge in neighbours})

    newpatchverts = []
    newX = []
    for patch in self.patches:
      mypatchverts = patchverts[list(patch)]
      myverts = X[list(patch)]
      edge_y, edge_x = patch[:2], patch[::2]
      newpatchverts.append( refine_patch(mypatchverts, edge2position.get(edge_x, None), edge2position.get(edge_y, None)) )
      newX.append( refine_patch(myverts, edge2position.get(edge_x, None), edge2position.get(edge_y, None)) )

    newpatchverts = np.concatenate(newpatchverts)
    newX = np.concatenate(newX)

    for vert, x in zip(map(round_to_5, newpatchverts.reshape(-1, 2)), map(round_to_5, newX.reshape((-1, 2)))):
      if vert not in map_vert_index:
        map_vert_index[vert] = counter.__next__()
        map_x_index[x] = map_vert_index[vert]

    patches = []
    for patchverts in newpatchverts:
      patch = []
      for vert in patchverts:
        patch.append(map_vert_index[round_to_5(vert)])
      patches.append(patch)

    newpatchverts = tuple(sorted(map(round_to_5, np.unique(newpatchverts.reshape(-1, 2), axis=0)), key=lambda x: map_vert_index[x]))
    newX = np.array(sorted(map(round_to_5, np.unique(newX.reshape(-1, 2), axis=0)), key=lambda x: map_x_index[x]))

    template = MultiPatchTemplate(patches, newpatchverts)
    new_obe, = template.ordered_boundary_edges
    inner_nodes = np.sort(np.setdiff1d(np.arange(len(newpatchverts)), np.asarray([edge[0] for edge in new_obe])))

    boundary_edges = set(self.ordered_boundary_edges[0]) | set(edge[::-1] for edge in self.ordered_boundary_edges[0])
    obe, = self.ordered_boundary_edges

    if edge not in obe:
      edge = edge[::-1]
    i0 = self.ordered_boundary_edges[0].index(edge)

    opposite_edge, = list(set(self.edge_neighbours(edge)) & boundary_edges)
    if opposite_edge not in self.ordered_boundary_edges[0]:
      opposite_edge = opposite_edge[::-1]
    i1 = obe.index(opposite_edge)

    if i1 < i0:
      i0, i1 = i1, i0
      positions = 1 - positions
      edge = obe[i0]

    f0, f1 = self._transforms[i0], self._transforms[i1]
    list_of_transforms = self._transforms[:i0] + f0.split_arclength(positions[0]) + \
                         self._transforms[i0+1: i1] + f1.split_arclength(1 - positions[0]) + \
                         self._transforms[i1+1:]
    ret = ControlMap(patches, newpatchverts, list_of_transforms, knotvector_edges=template._knotvector_edges, inner_verts=newX[inner_nodes])
    return ret

  @property
  def X(self):
    X = np.empty((len(self.patchverts), 2), dtype=float)
    X[self.ordered_boundary_indices] = np.concatenate(self.sample_boundary(1))
    X[self.dofindices] = self.inner_verts
    return np.round(X, 7)

  def to_MultiPatchTemplate(self):
    return MultiPatchTemplate(self.patches, self.patchverts, self._knotvector_edges)

  def normalize(self, npoints=101):
    """ Normalize to unit volume centered in the origin. """
    X = np.concatenate(self.sample_boundary(npoints))
    vol = Polygon(X).area
    scaling = 1 / np.sqrt(vol) * np.ones(2)
    distances = np.linalg.norm(np.diff(np.concatenate([X, X[:1]]), axis=0), axis=1)
    center = (distances[:, _] * X).sum(0) / distances.sum()
    funcs = [Scale(Translate(func, -center), scaling) for func in self._transforms]
    return ControlMap(self.patches, self.patchverts, funcs, knotvector_edges=self._knotvector_edges, inner_verts=(self.inner_verts - center[_])*scaling[_])

  def triangulate(self, npoints=5, **kwargs):
    vertices = np.concatenate(self.sample_boundary(npoints))
    return Triangulation.from_polygon(vertices, **kwargs)

  def compute_vertices(self, npoints=5, **kwargs):
    if len(self.dofindices) == 0:
      self.inner_verts = frozen(np.zeros((0, 2)))
    else:
      self_template = self.to_MultiPatchTemplate()
      template_tri = ControlMap.from_template(self_template).triangulate(npoints=npoints, **kwargs)
      X = np.concatenate(self.sample_boundary(npoints))
      patchverts = np.asarray(self.patchverts)
      Xi = np.array([x for i, x in enumerate(patchverts) if i not in self.ordered_boundary_indices])
      c = template_tri.floater(X, as_Triangulation=False)
      self.inner_verts = frozen(template_tri.evaluate(c, Xi))
      if OPTIMIZE_CONTROLMAP:
        try:
          optimize_template = self_template.edit(patchverts=self.X)
          optimize_template = default_strategy(optimize_template)
          self.inner_verts = frozen(np.asarray(optimize_template.patchverts)[self.dofindices])
        except Exception:
          pass

  @property
  def is_valid(self):
    ex, ey = ex_ey(self.patches)
    X = self.X
    us = X[ex[:, 1]] - X[ex[:, 0]]
    vs = X[ey[:, 1]] - X[ey[:, 0]]

    return (np.cross(us, vs) > 0).all()

  def areas(self, npoints=101):
    dX = self.sample_boundary(npoints, repeat_last=True)
    X = self.X
    obe, = self.ordered_boundary_edges
    ret = []
    for patch in np.asarray(self.patches)[:, [0, 2, 3, 1, 0]]:  # ccw ordering + repeat last vertex
      myverts = []
      for edge in zip(patch, patch[1:]):
        if tuple(edge) in obe or tuple(edge[::-1]) in obe:
          if tuple(edge[::-1]) in obe: myX = dX[obe.index(tuple(edge[::-1]))][::-1]
          else: myX = dX[obe.index(tuple(edge))]
        else:
          myX = X[list(edge)]
        myverts.append(myX[:-1])
      ret.append( Polygon(np.concatenate(myverts)).area )
    return np.asarray(ret, dtype=float)

  def plot(self, npoints=1001, ax=None, show=True):
    fig, ax = self.plot_boundary(npoints=npoints, ax=ax, show=False)
    patches = np.asarray(self.patches, dtype=int)
    X = np.empty((len(self.patchverts), 2), dtype=float)
    X[list(self.ordered_boundary_indices)] = np.concatenate(self.sample_boundary(1))
    X[self.dofindices] = self.inner_verts
    obe = set(self.ordered_boundary_edges[0])
    for patch in patches:
      for edge in get_edges(patch):
        if len({edge, edge[::-1]} & obe) != 0:
          continue
        ax.plot(*X[list(edge)].T, c='k')
        ax.scatter(*X[list(edge)].T, c='r', s=20, zorder=10)
    if show:
      plt.show()
    return fig, ax

  def plot_boundary(self, npoints=1001, ax=None, show=True, s=None):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.figure
    ax.set_aspect('equal')
    X = np.concatenate(self.sample_boundary(npoints))
    verts = np.concatenate(self.sample_boundary(1))
    X = np.concatenate([X, X[:1]])
    ax.plot(*X.T, c='k')
    ax.scatter(*verts.T, c='r', zorder=5, s=s)
    if show:
      plt.show()
    return fig, ax

  def create_controlmap(self, mg, method='laplace'):
    from nutils.util import NanVec
    assert self.patches == mg.patches
    obe, = self.ordered_boundary_edges

    patchverts = np.asarray(mg.patchverts)

    geom = mg.geom
    cons = NanVec([2*len(mg.basis)])
    for (i, j), func in zip(obe, self._transforms):
      p0, p1 = patchverts[i], patchverts[j]
      mylength = np.linalg.norm(p1 - p0)

      s = ((geom - p0)**2).sum()**.5 / mylength

      cons |= mg.project_edge((i, j), func.evalf(s)[0])

    mapped_edges = set(map(abs_tuple, obe))

    X = self.X
    for i, patch in enumerate(mg.patches):
      for edge in (patch[:2], patch[2:], patch[::2], patch[1::2]):
        if edge in mapped_edges or edge[::-1] in mapped_edges: continue
        i, j = edge
        p0, p1 = patchverts[i], patchverts[j]
        v0, v1 = X[i], X[j]
        mylength = np.linalg.norm(p1 - p0)

        s = ((mg.geom - p0)**2).sum()**.5 / mylength

        cons |= mg.project_edge((i, j), v0 + (v1 - v0) * s)
        mapped_edges.update({edge})

    mg_ = deepcopy(mg)
    mg_.cons = cons

    mg_.x = mg_.cons | 0

    _parameterise_interior(mg_, method=method)

    mg.controlmap = mg_.mapping

  def to_MultiPatchBSplineGridObject(self, *args, method='laplace', **kwargs):
    mg = self.to_MultiPatchTemplate().to_MultiPatchBSplineGridObject(*args, **kwargs)
    self.create_controlmap(mg, method=method)
    return mg

  def to_EdgeNetwork(self, npoints=101, inner_points=None):
    if inner_points is None:
      inner_points = npoints

    counter = count(1)
    xi = np.linspace(0, 1, inner_points)

    X = self.sample_boundary(npoints)
    X = [np.concatenate([X[i], X[(i+1) % len(X)][:1]]) for i in range(len(X))]
    edges = tuple(map(as_edge, X))

    # edges = tuple(map(lambda func: as_edge(func.evalf(xi)), self._transforms))
    mapped_edges = set(map(abs_tuple, self.ordered_boundary_edges[0]))

    patches = self.patches
    inner_edges = sorted(set(chain(*[map(abs_tuple, get_edges(patch)) for patch in patches])) - mapped_edges)

    edge2index = dict(zip(self.ordered_boundary_edges[0] + tuple(inner_edges), counter))

    X, xi = self.X, xi[:, _]
    edges = edges + tuple(as_edge(X[i][_] * (1 - xi) + X[j][_] * xi) for (i, j) in inner_edges)

    face_indices = {}
    for i, patch in enumerate(patches):
      myedges = (patch[::2], patch[2:], patch[1::2][::-1], patch[:2][::-1])
      face_indices[(i,)] = [edge2index[edge] if edge in edge2index else -edge2index[edge[::-1]] for edge in myedges]

    indices = np.arange(1, len(edges) + 1)
    return EdgeNetwork(indices, edges, face_indices)


def apply_transfinite_to_all_patches(mg):

  gs = mg.break_apart()
  basis_disc = mg.make_basis(patchcontinuous=False)

  for g in gs:
    g.set_cons_from_x()
    transfinite_interpolation(g)

  x = np.concatenate([g.x for g in gs])
  func = basis_disc.vector(2).dot(x)

  mg.x = mg.project(func)


def _get_vertex_positions_from_mapping(mg):
  verts = {}

  for i, patch in enumerate(mg.patches):
    topo = mg.domain.patches[i].topo
    for ii, j in enumerate(patch):
      side0, side1 = { 0: ('left', 'bottom'),
                       1: ('left', 'top'),
                       2: ('right', 'bottom'),
                       3: ('right', 'top') }[ii]
      verts[j] = topo.boundary[side0].boundary[side1].sample('vertex', 0).eval(mg.mapping).ravel()
  return np.stack([verts[j] for j in range(len(verts))])


def check_if_method_is_valid(fn):

  @wraps(fn)
  def wrapper(*args, method='coons', **kwargs):
    assert method in METHODS
    return fn(*args, method=method, **kwargs)

  return wrapper


@check_if_method_is_valid
def _parameterise_interior(mg, method='coons'):
  if method == 'laplace':
    forward_laplace(mg)
  elif method == 'coons':
    apply_transfinite_to_all_patches(mg)
  elif method == 'local_laplace':
    controlmap = mg.controlmap
    mg.controlmap = mg.localgeom
    forward_laplace(mg)
    mg.controlmap = controlmap


def test_controlmap():
  from .template import even_n_leaf_template
  A = even_n_leaf_template(8)
  obe, = A.ordered_boundary_edges
  obi = [edge[0] for edge in obe]
  X = np.asarray(A.patchverts)[obi]
  center = X.sum(0) / len(obi)
  fac = 2

  funcs = []
  verts = np.concatenate([X, X[:1]])
  for _v0, _v1 in zip(verts, verts[1:]):
    funcs.append(linear(center + (_v0 - center) * fac, center + (_v1 - center) * fac))

  controlmap = ControlMap(**A._lib(), list_of_transforms=funcs)

  controlmap.plot()

  edge = obe[-1]
  controlmap.refine(edge, .1).plot()

  edge = obe[0]
  controlmap.refine(edge, .1).plot()

  controlmap = unit_disc(A)
  controlmap.plot()

  controlmap.refine(edge, .1).plot()

  controlmap = teardrop(A)
  controlmap.plot()

  controlmap.to_EdgeNetwork().qplot()

  controlmap.refine(edge, .1).plot()

  controlmap = lense(A, root1=3)
  controlmap.plot()

  controlmap.to_MultiPatchBSplineGridObject(knotvectors=5).g_controlmap().qplot()

  controlmap.refine(edge, .1).plot()

  controlmap.refine(edge, .1).to_MultiPatchBSplineGridObject(knotvectors=5).g_controlmap().qplot()

  controlmap = polygon(A)
  controlmap.plot()

  controlmap.refine(edge, .1).plot()


def test_pick_fitting_controlmap():
  from .network import EdgeNetwork
  from .template import even_n_leaf_template
  connie = EdgeNetwork.fromfolder('networks/connie').take([(5,)])
  indices, = connie.face_indices.values()
  template = even_n_leaf_template(len(indices))
  controlmap = pick_fitting_controlmap(template, connie)
  controlmap.plot()