from index import as_OrientableTuple, as_OrientableFloat, PointCloudEdge, \
                  as_IndexSequence, SaveLoadMixin, as_KnotObject, SplineEdge, \
                  OrientableTuple, ReflectedDictionary, LinearEdge, OrientableFloat, \
                  as_sorted_tuple_FrozenDict, IndexSequence, FrozenDict, reflect_dict, \
                  geometry, ops
from aux import get_edges, edge_neighbours, infer_knotvectors, opposite_side, \
                get_edges_clockwise, infer_knotvector_edges
from network import Network, EdgeNetwork, log
from collections import defaultdict
from functools import wraps, lru_cache, cached_property
from mapping import ko
import itertools
import numpy as np
from matplotlib import pyplot as plt
from canvas import MatplotlibEventHandler
from collections import deque


# VERTEX-EDGE NETWORK


def normalize_split_index(c, i):
  return as_OrientableTuple(c), as_OrientableFloat(i)


def apply_pairing(func):

  @wraps(func)
  def wrapper(self, edge, position, *args, **kwargs):
    pairing = as_OrientableTuple(edge) @ self.indices
    assert pairing
    edge, position = pairing * as_OrientableTuple(edge), pairing * as_OrientableFloat(position)
    return func(self, edge, position, *args, **kwargs)

  return wrapper


# auxiliary function for multipatch domains


def orient_knotvectors(knotvectors):
  """ Make sure that a dict of the form {edge: knotvector} is such that edge[1] > edge[0].
      In case edge[1] < edge[0], replace entry by (edge[1], edge[0]): knotvector.flip(). """
  ret = {}
  for (a, b), knotvector in knotvectors.items():
    if b > a: ret[(a, b)] = knotvector
    else: ret[(b, a)] = knotvector.flip()
  return ret


def check_knotvector_compatibility(patches, knotvectors):
  if not all( b > a for (a, b) in knotvectors.keys() ): return False
  for edge, knotvector in knotvectors.items():
    neighbours = edge_neighbours(patches, edge)
    for neigh in neighbours:
      if neigh not in knotvectors: return False
      if knotvectors[neigh] != knotvector: return False
  return True


@lru_cache
def _patches_verts_to_network(patches, patchverts):
  """ Docstring forthcoming. """
  assert all( len(patch) == 4 for patch in patches )
  assert all( len(verts) == 2 for verts in patchverts )
  face_indices = defaultdict(lambda: [0, 0, 0, 0])
  edgedict = {}  # reversed
  map_index_original_edge = {}
  current_edge = 1
  for i, patch in enumerate(patches):
    # patchverts = {j: patchverts[j] for j in patch}
    edges = get_edges(patch)
    edges = [edges[2], edges[1], edges[3], edges[0]]  # bottom right top left
    for j, edge in enumerate(edges):
      # myedge is given by the concatenation of self.patchverts[k] for k in edge
      myedge = PointCloudEdge(np.stack([ patchverts[k] for k in edge ], axis=0))
      if myedge in edgedict: myindex = edgedict[myedge]
      elif -myedge in edgedict: myindex = -edgedict[-myedge]
      else:
        myindex = current_edge
        current_edge += 1
      edgedict[myedge] = myindex
      map_index_original_edge[myindex], map_index_original_edge[-myindex] = tuple(edge), tuple(edge)[::-1]
      face_indices[(i, 0)][j] = myindex if j in (0, 1) else -myindex  # bottom right are in positive direction, else negative
  edgedict = {index: edge for edge, index in edgedict.items()}
  from network import EdgeNetwork
  return EdgeNetwork.from_edgedict(edgedict, face_indices), map_index_original_edge


def patches_verts_to_network(patches, patchverts):
  ret = _patches_verts_to_network(tuple(map(tuple, patches)),
                                  tuple(map(tuple, patchverts)))
  return (ret[0].copy(), dict(ret[1]))


def silhouette(patches, patchverts, index0=None, show=True, ax=None):
  if index0 is None:
    index0 = 0
  network, _map = patches_verts_to_network(patches, patchverts)
  network._colors = defaultdict(lambda: 'b')
  fig, ax = network.plot_polygons(indices=False, show=False, ax=ax)
  # index0 = network.boundary_network().face_indices[(0, 0)][0]
  index0 = network.ordered_boundary_indices[0][index0]
  edge = network.get_edges(index0)
  ax.plot(*edge.points.T, linewidth=3, c='r')
  if not show:
    return fig, ax
  plt.show()


def select_index(patches, patchverts, network, index):
  assert len(network.face_indices) == 1
  indices, = list(network.face_indices.values())

  fig, axes = plt.subplots(nrows=1, ncols=2)
  from canvas import select_edges
  fig, ax0 = silhouette(patches, patchverts, show=False, ax=axes[0], index0=index)
  key, = list(network.face_indices.keys())
  index0, = select_edges(network, key,
                         nclicks=1,
                         ax=axes[1],
                         title='Select the edge that corresponds to the'
                                'highlighted edge in the domain')
  return index0


select_index0 = lambda *args, **kwargs: select_index(*args, index=0, **kwargs)


class DragTemplateVertices(MatplotlibEventHandler):

  def __init__(self, template, vfac=None, **kwargs):
    if vfac is None: vfac = 0.0125
    assert 0 < vfac < 1

    self.patchverts = np.stack(template.patchverts).astype(np.float64)
    self.patches = np.stack(template.patches).astype(np.int64)

    map_vertex_patch = defaultdict(set)
    for i, patch in enumerate(self.patches):
      for value in patch:
        map_vertex_patch[value].update({i})
    self.map_vertex_patch = {key: sorted(value) for key, value in map_vertex_patch.items()}

    self.vertices = dict()
    self.lines = dict()

    self.vfac = vfac
    self.point_highlighted = False
    self.vminind = None
    self.highlightedpoint = None
    self.clicked = False
    self.patchverts_changed = True  # make sure axes limits etc are set right away

    self.x_pressed = False
    self.y_pressed = False

    self.undo = deque(maxlen=5)

    fig, ax = plt.subplots()
    ax.grid(visible=True)
    super().__init__(ax)

    for index in range(len(self.patches)):
      self.redraw_patch(index)

  def draw(self):
    if self.patchverts_changed:  # reset limits
      xmin, xmax = self.patchverts[:, 0].min(), self.patchverts[:, 0].max()
      ymin, ymax = self.patchverts[:, 1].min(), self.patchverts[:, 1].max()
      width = xmax - xmin
      height = ymax - ymin
      self.rvertex = self.vfac * np.linalg.norm([width, height])
      self.ax.set_xlim( xmin - 0.75 * width, xmax + 0.75 * width )
      self.ax.set_ylim( ymin - 0.75 * height, ymax + 0.75 * height )
      self.ax.set_aspect('equal')
      self.patchvers_changed = False
    super().draw()

  def redraw_patch(self, index):
    indices = self.patches[index]
    vertices = self.patchverts[indices[[0, 2, 3, 1, 0]]]

    # remove previous scatter plot if present
    try:
      self.vertices[index].remove()
    except KeyError:
      pass

    # remove previous line plot if present
    try:
      self.lines[index].remove()
    except KeyError:
      pass

    # reset vertices and lines to new plots
    self.vertices[index] = self.ax.scatter(*vertices.T, c='k', zorder=2)
    self.lines[index] = self.ax.plot(*vertices.T, c='r', zorder=1)[0]

  def on_press(self, event):
    'button_press_event'
    if not self.point_highlighted:  # ignore unintended click
      return
    else:
      if self.clicked is False:  # vertex is highlighted and clicked => it is selected
        self.clicked = True
      else:  # vertex is highlighted and has already been clicked
        x, y = event.xdata, event.ydata
        if x is None or y is None:  # click outside of bounds => ignore
          return

        vminind = self.vminind
        self.undo.append((vminind, self.patchverts[vminind].copy()))

        # set patchvert to new value
        if self.x_pressed:
          self.patchverts[vminind][0] = x
        elif self.y_pressed:
          self.patchverts[vminind][1] = y
        else:
          self.patchverts[vminind] = np.array([x, y])

        # patchverts have now changed
        self.patchverts_changed = True

        # redraw all relevant patches
        for patchindex in self.map_vertex_patch[self.vminind]:
          self.redraw_patch(patchindex)

        # no longer clicked
        self.clicked = False

        # no longer highlighted
        self.point_highlighted = False
        self.highlightedpoint.remove()

        # redraw
        self.draw()

  def on_motion(self, event):
    'motion_notify_event'
    if not self.clicked:  # nothing selected => highlight closest point
      if event.xdata is not None:
        x, y = event.xdata, event.ydata
        dist = np.linalg.norm(self.patchverts - np.array([x, y])[None], axis=1)
        minind = np.argmin(dist)
        if dist[minind] < self.rvertex:
          point = self.patchverts[minind]
          if self.point_highlighted:
            self.highlightedpoint.remove()
          self.vminind = minind
          self.highlightedpoint = self.ax.scatter(*point, c='k', s=80, zorder=3)
          self.point_highlighted = True
          self.draw()
        else:
          if self.point_highlighted:
            self.highlightedpoint.remove()
            self.point_highlighted = False
            self.draw()
        return
    return

  def on_key(self, event):
    'key_press_event'
    if event.key == ' ': self.close()
    elif event.key == 'x':
      self.x_pressed = True
    elif event.key == 'y':
      self.y_pressed = True
    elif event.key == 'u' and self.undo:
      index, vertex = self.undo.pop()
      if self.point_highlighted:
        self.highlightedpoint.remove()
        self.point_highlighted = False
      self.patchverts_changed = True
      self.clicked = False
      self.patchverts[index] = vertex
      for patchindex in self.map_vertex_patch[index]:
        self.redraw_patch(patchindex)
      self.draw()
    super().on_key(event)

  def on_release(self, event):
    'key_release_event'
    if event.key == 'x':
      self.x_pressed = False
    if event.key == 'y':
      self.y_pressed = False


@lru_cache(maxsize=32)
def MultiPatchTemplate2EdgeNetwork(mpt):
  edges = [np.array([ mpt.patchverts[e] for e in edge]) for edge in mpt.edges]
  return EdgeNetwork(mpt.indices, edges, mpt.pol_indices)


@lru_cache(maxsize=32)
def patches_to_network(patches):
  edgedict = ReflectedDictionary()
  current_index = 1
  pol_indices = {}
  for i, patch in enumerate(patches, 1):
    myindices = []
    for edge in map(as_OrientableTuple, get_edges_clockwise(patch)):
      myindex = edgedict.get(edge, None) or edge.sign * current_index
      myindices.append( myindex )
      if abs(myindex) == current_index:
        edgedict[edge] = myindex
        current_index += 1
    pol_indices[i] = myindices

  # invert
  edges, indices = zip(*edgedict.positive_items())
  return indices, edges, pol_indices


def _normalize_periodic_edges(self, periodic_edges):
  if not periodic_edges: return tuple()
  _periodic_edges = tuple( tuple(map(OrientableTuple, pair)) for pair in periodic_edges )
  boundary_edges = as_IndexSequence(self.get_edges(self.boundary_indices))
  assert all( edge @ boundary_edges for edge in itertools.chain(*_periodic_edges) )
  assert all( e0 != e1 for e0, e1 in _periodic_edges )
  periodic_edges = set.union(*({tuple(min((e0, e1), (-e0, -e1),
                                          (e1, e0), (-e1, -e0)) )} for e0, e1 in _periodic_edges))

  # Removed the requirement that coupled edges are each other's boundary opposites.
  # As a downside, refinement of coupled edges that are not coupled to their opposites
  # will throw an error (for now).

  # for e0, e1 in _periodic_edges:
  #   if not self.boundary_opposite_edges[tuple(e0 * e0.sign)] == tuple(e1 * e0.sign):
  #     raise NotImplementedError("Cannot couple edges that are each other's boundary opposites.")

  return sorted(periodic_edges)


class MultiPatchTemplate(Network):

  edge_dtype = OrientableTuple

  @staticmethod
  def _fuse(template0, template1, tol=1e-7):
    # XXX: replace by more readable function

    if not len(template0.periodic_edges) == len(template1.periodic_edges) == 0:
      raise NotImplementedError

    vertex_array0 = np.asarray(template0.patchverts)
    vertex_array1 = np.asarray(template1.patchverts)

    outer = ((vertex_array0[:, None] - vertex_array1[None])**2).sum(-1)**.5
    pairs = {}

    for i, j in zip(*np.where(outer < tol)):
      assert j not in pairs, 'Found a vertex that matches with more than one other vertex. Please reduce the tolerance.'
      pairs[j] = i

    assert len(pairs) >= 2

    patchverts = list(template0.patchverts)

    N = len(vertex_array0) - 1
    map_old_index_new_index = {}

    for j, edge in enumerate(vertex_array1):
      if (match := pairs.get(j, None)) is not None:
        map_old_index_new_index[j] = match
      else:
        patchverts.append(edge)
        map_old_index_new_index[j] = N = N + 1

    patches = template0.patches + tuple( tuple(map_old_index_new_index[j] for j in patch) for patch in template1.patches )

    knotvector_edges = list(template0._knotvector_edges)
    for edge in template1._knotvector_edges:
      y = tuple(map_old_index_new_index[i] for i in edge)
      if y in knotvector_edges or y[::-1] in knotvector_edges:
        continue
      test = knotvector_edges + [y]
      try:
        infer_knotvector_edges(patches, test)
        knotvector_edges.append(y)
      except Exception:
        continue

    knotvector_edges = tuple(sorted(set(knotvector_edges)))

    return MultiPatchTemplate(patches, patchverts, knotvector_edges)  # .repair_orientation()

  @staticmethod
  def stack(template0, template1, edge0, edge1, direction=1, return_vertex_map=False):
    bindices0, = template0.ordered_boundary_edges
    bindices1, = template1.ordered_boundary_edges

    assert direction in (-1, 1)

    assert edge0 in bindices0 or edge0[::-1] in bindices0
    assert edge1 in bindices1 or edge1[::-1] in bindices1

    offset0 = np.asarray(template0.patchverts[edge0[0]])
    offset1 = np.asarray(template1.patchverts[edge1[0]])

    template0 = template0.translate(-offset0)
    template1 = template1.translate(-offset1)

    patchverts0 = np.array(template0.patchverts)
    patchverts1 = np.array(template1.patchverts)
    from aux import angle_between_vectors_

    v0 = patchverts0[edge0[1]] - patchverts0[edge0[0]]
    v1 = patchverts1[edge1[1]] - patchverts1[edge1[0]]

    v0_norm = np.linalg.norm(v0)
    v1_norm = np.linalg.norm(v1)

    theta0 = angle_between_vectors_((0, 1), v0)
    theta1 = angle_between_vectors_((0, 1), v1)

    template0, template1 = template0.rotate(-theta0), template1.rotate(-theta1)
    template1 = template1.stretch((1, 1), v0_norm / v1_norm)

    map_verts = {}
    map_verts[edge1[0]] = edge0[0]
    map_verts[edge1[1]] = edge0[1]

    n = len(template0.patchverts)

    for i in range(len(template1.patchverts)):
      if i not in edge1:
        map_verts[i] = n
        n += 1

    patchverts = template0.patchverts + tuple( vert for i, vert in enumerate(template1.patchverts) if i not in edge1 )
    patches = template0.patches + tuple( tuple(map_verts[i] for i in patch) for patch in template1.patches )
    knotvector_edges = list(template0._knotvector_edges)
    for edge in template1._knotvector_edges:
      y = tuple(map_verts[i] for i in edge)
      if y in knotvector_edges or y[::-1] in knotvector_edges:
        continue
      test = knotvector_edges + [y]
      try:
        infer_knotvector_edges(patches, test)
        knotvector_edges.append(y)
      except Exception:
        continue

    knotvector_edges = tuple(sorted(set(knotvector_edges)))

    temp = MultiPatchTemplate(patches, patchverts, knotvector_edges).rotate(theta0).translate(offset0).repair_orientation()

    if return_vertex_map:
      return temp, map_verts

    return temp

  def _lib(self):
    return {'patches': self.patches,
            'patchverts': self.patchverts,
            'knotvector_edges': self._knotvector_edges,
            'periodic_edges': self.periodic_edges}

  def __init__(self, patches, patchverts, knotvector_edges=None, periodic_edges=None, undo=None):
    assert all( len(patch) == 4 for patch in patches )
    assert all( len(vert) == 2 for vert in patchverts )
    assert set(itertools.chain(*patches)) == set(range(len(patchverts)))
    self.patches = tuple(map(tuple, patches))
    self.patchverts = tuple(map(lambda x: tuple(np.round(x, 5)), patchverts))
    indices, edges, pol_indices = patches_to_network(self.patches)
    super().__init__(indices, edges, pol_indices, undo=undo)

    if knotvector_edges is None:
      knotvector_edges = _find_knotvector_edges(self)

    # assert set(itertools.chain(*knotvector_edges)).issubset(set(range(len(patchverts))))
    self._knotvector_edges = tuple(map(tuple, knotvector_edges))
    self.knotvector_edges = infer_knotvector_edges(self.patches, self._knotvector_edges)
    self.pol_edges = FrozenDict({key: self.get_edges(value) for key, value in self.pol_indices.items()})

    self.periodic_edges = _normalize_periodic_edges(self, periodic_edges)

    if len(self.ordered_boundary_indices) > 1:
      raise NotImplementedError

  def get_edges(self, indices):
    if np.isscalar(indices):
      return super().get_edges((indices,))[0]
    return super().get_edges(indices)

  @cached_property
  def immediate_opposites(self):
    immediate_opposites = defaultdict(set)
    for pol_index, indices in self.pol_indices.items():
      for a, b in zip(indices, indices.roll(-2)):
        immediate_opposites[a].update({-b})
        immediate_opposites[-a].update({b})
    return as_sorted_tuple_FrozenDict(immediate_opposites)

  @cached_property
  def opposites(self):
    opposites_old = {key: set(value) for key, value in self.immediate_opposites.items()}
    while True:
      opposites = {key: value.copy() for key, value in opposites_old.items()}
      for index, myopposites in opposites_old.items():
        opposites[index].update(set.union(*[opposites_old[i] for i in myopposites]) - {index})
      if opposites == opposites_old:
        break
      opposites_old = opposites
    return as_sorted_tuple_FrozenDict(opposites)

  @cached_property
  def boundary_opposites(self):
    bindices = IndexSequence(self.boundary_indices)
    opposites = {key: IndexSequence(value) for key, value in self.opposites.items()}
    ret = { index: [val for val in opposites[index] if val @ bindices][0] for index in bindices }
    return FrozenDict(reflect_dict(ret))

  @cached_property
  def immediate_opposite_edges(self):
    return as_sorted_tuple_FrozenDict({self.get_edges(index): self.get_edges(indices)
                                       for index, indices in self.immediate_opposites.items()})

  @cached_property
  def opposite_edges(self):
    return as_sorted_tuple_FrozenDict({self.get_edges(index): self.get_edges(indices)
                                       for index, indices in self.opposites.items()})

  @cached_property
  def boundary_opposite_edges(self):
    return FrozenDict({self.get_edges(key): self.get_edges(value)
                       for key, value in self.boundary_opposites.items()})

  @cached_property
  def ordered_boundary_indices(self):
    obi = list(map(as_IndexSequence, self.ordered_boundary_indices_orientation_free))
    pols = (geometry.Polygon(np.array([ self.patchverts[e0] for e0, e1 in self.get_edges(ob) ])) for ob in obi)
    oriented_obi = [ob * {False: -1, True: 1}[ops.orient(pol) == pol] for ob, pol in zip(obi, pols)]
    return tuple(map(tuple, oriented_obi))

  @cached_property
  def ordered_boundary_indices_periodic(self):
    if not self.periodic_edges: return self.ordered_boundary_indices
    map_edge_index = dict(zip(self.edges, self.indices))
    index0, *periodic_indices = ( abs(map_edge_index[abs(edge)])
                                  for edge in itertools.chain(*self.periodic_edges) )
    obi, = map(lambda x: as_IndexSequence(x).roll_to(index0), self.ordered_boundary_indices)
    local_indices = [0] + sorted(abs(obi).tuple_index(x) for x in periodic_indices) + [len(obi)]
    chunks = [obi[i0+1: i1] for i0, i1 in zip(local_indices, local_indices[1:]) if i1 - (i0+1) != 0]
    return tuple(map(tuple, chunks))

  @cached_property
  def ordered_boundary_edges_periodic(self):
    return tuple(map(self.get_edges, self.ordered_boundary_indices_periodic))

  def add_periodic_edges(self, periodic_edges):
    assert bool(periodic_edges)
    periodic_edges = set(_normalize_periodic_edges(self, periodic_edges))
    if periodic_edges.issubset(set(self.periodic_edges)):
      return self
    return self.edit(periodic_edges=periodic_edges | set(self.periodic_edges))

  def __hash__(self):
    return hash((self.patches, self.patchverts, self.knotvector_edges, self.periodic_edges))

  def __eq__(self, other):
    if self.__class__ != other.__class__: return False
    if self.periodic_edges != other.periodic_edges: return False
    if self.knotvector_edges != other.knotvector_edges: return False
    if self.patches != other.patches: return False
    return self.patchverts == other.patchverts

  def translate(self, x):
    return self.edit(patchverts=np.asarray(self.patchverts) + np.array(x)[None])

  def rotate(self, theta):
    A = np.array([ [np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)] ])
    return self.edit(patchverts=(A @ np.array(self.patchverts).T).T)

  def stretch(self, vec, magnitude, x0=None):
    if x0 is None:
      x0 = np.array([0, 0])
    vec, x0 = np.asarray(vec), np.asarray(x0)
    patchverts = (np.array(self.patchverts) - x0[None])
    patchverts = (patchverts + (magnitude - 1) * (patchverts * vec[None])) + \
                 (magnitude - 1) * (vec * x0)[None]
    return self.edit(patchverts=patchverts)

  def translate_vertex(self, index, vec):
    patchverts = np.array(self.patchverts, dtype=float)
    patchverts[index] += np.asarray(vec)
    return self.edit(patchverts=patchverts)

  @cached_property
  def patch_matrix(self):
    """ Matrix M where M[i] = [j, k, l, m] means that the neighbours of the i+1-th
        patch (counting from 1) are:
          left => j, right => k, bottom => l, top => m.
        If the patch has no neighbour to a side, the index is 0.
    """
    edge_to_patch, patch_to_edge = defaultdict(list), defaultdict(list)

    for i, patch in enumerate(self.patches, 1):
      for edge in map(OrientableTuple, get_edges(patch)):
        edge_to_patch[edge].append(i), edge_to_patch[-edge].append(-i)
        patch_to_edge[i].append(edge), patch_to_edge[-i].append(-edge)

    patch_neighbours = {}
    for i in range(1, len(self.patches) + 1):
      myneighbours = {}
      for side, edge in zip(('left', 'right', 'bottom', 'top'), patch_to_edge[i]):
        for patch in set(edge_to_patch[edge]) - {i}:
          myneighbours[side] = patch
      patch_neighbours[i] = myneighbours

    patch_matrix = np.zeros([len(self.patches), 4], dtype=int)
    for index, neighbours in patch_neighbours.items():
      for side, neighbour in neighbours.items():
        patch_matrix[index-1, {'left': 0, 'right': 1, 'bottom': 2, 'top': 3}[side]] = neighbour

    return patch_matrix

  def fuse(self, other):
    return self.__class__._fuse(self, other)

  def refine(self, edge=None, positions=.5, index=None):

    # XXX: this routine is too long. Clean up, make more readable.

    if np.isscalar(positions):
      positions = positions,

    # round to 5 figures for the hashmap we will be using later on
    positions = IndexSequence(list(map(lambda x: round(x, 5), positions)), dtype=OrientableFloat)
    assert len(positions) == len(set(positions))

    # either refine via edge or via index (not both)
    assert bool(index) != bool(edge)

    # if via index: get corresponding edge
    if index:
      edge = self.get_edges(index)

    # turn to standard orientation edge = (a, b) with a < b
    edge = OrientableTuple(edge)
    positions = edge.sign * positions
    edge = abs(edge)

    position, *positions = sorted(positions)

    assert edge in self.edges

    # get neighbours
    neighbours = IndexSequence([edge] + list(self.opposite_edges[edge]))

    patches = {}
    m = len(self.patches)
    fac = np.array([1 - position, position])[:, None]

    # create patches made of actual vertex positions
    for i, patch in enumerate(self.patches):
      myvertices = tuple(self.patchverts[i] for i in patch)
      myedges = tuple(map(OrientableTuple, get_edges(patch)))

      # loop over neighbours (a, b) and check if (a, b) or (b, a) is in the patch
      for neigh in neighbours:
        if neigh in myedges: index = myedges.index(neigh)
        elif -neigh in myedges: index = myedges.index(-neigh)
        else: continue
        myfac = fac if neigh in myedges else 1 - fac
        if index in (0, 1):  # left or right
          v0 = tuple(np.round((np.stack([myvertices[0], myvertices[1]]) * myfac).sum(0), 5))
          v1 = tuple(np.round((np.stack([myvertices[2], myvertices[3]]) * myfac).sum(0), 5))
          patches[i] = [myvertices[0], v0, myvertices[2], v1]
          patches[m] = [v0, myvertices[1], v1, myvertices[3]]
          m += 1
        else:  # bottom or top
          v0 = tuple(np.round((np.stack([myvertices[0], myvertices[2]]) * myfac).sum(0), 5))
          v1 = tuple(np.round((np.stack([myvertices[1], myvertices[3]]) * myfac).sum(0), 5))
          patches[i] = [myvertices[0], myvertices[1], v0, v1]
          patches[m] = [v0, v1, myvertices[2], myvertices[3]]
          m += 1
        break
      # None of the neighbours in the patch => patch remains unchanged
      else:
        patches[i] = myvertices

    # get all vertices created (old + new ones)
    all_vertices = list(set(itertools.chain(*list(patches.values()))))

    # map each vertex to a new number
    # start with the old vertices
    map_vertex_index = dict(zip(self.patchverts, range(len(self.patchverts))))

    # add the new ones
    n = len(map_vertex_index)
    for vert in all_vertices:
      if vert not in map_vertex_index:
        map_vertex_index[vert] = n
        n += 1

    # create new patches by mapping the new verts to the new number for all
    # vertices in all patches
    patches = [ [map_vertex_index[vert] for vert in patches[i]] for i in range(len(patches)) ]

    # take inverse
    map_index_vertex = {value: key for key, value in map_vertex_index.items()}

    # make ordered list of new patchverts
    patchverts = [ map_index_vertex[ind] for ind in range(len(map_index_vertex)) ]

    original_edge = edge

    # map old knotvector_edges (a, b) to new ones by replacing
    # (a, b) by (a, c), (c, b) (c = a new vertex index) if edge (a, b)
    # has been refined, else just keep (a, b)
    knotvector_edges = []
    for edge in map(OrientableTuple, self._knotvector_edges):
      pairing = edge @ neighbours
      if pairing:
        myfac = fac if pairing == 1 else 1 - fac
        c = map_vertex_index[tuple(np.round((np.stack([self.patchverts[e] for e in edge]) * myfac).sum(0), 5))]
        knotvector_edges.extend( [(edge[0], c), (c, edge[1])] )
      else:
        knotvector_edges.append(edge)

    periodic_edges = []
    for pair in self.periodic_edges:
      pair = tuple(map(OrientableTuple, pair))

      if not (pair[0] @ neighbours == pair[1] @ neighbours):
        log.warning("Warning, found a pair of periodic edges that cannot be preserved"
                    " under refinement. Aborting.")
        raise RuntimeError("Refinement does not preserve periodicity structure.")

      if not pair[0] @ neighbours:
        new_pairs = [pair]
      else:
        new_pairs = [[], []]
        for edge in pair:
          myfac = fac if (edge @ neighbours == 1) else 1 - fac
          c = map_vertex_index[tuple(np.round((np.stack([self.patchverts[e] for e in edge]) * myfac).sum(0), 5))]
          new_pairs[0].append((edge[0], c))
          new_pairs[1].append((c, edge[1]))

      periodic_edges.extend(new_pairs)

    # create new class instantiation with new data
    ret = self.__class__(patches, patchverts, knotvector_edges, periodic_edges=periodic_edges)

    # if no more positions need to be refined, return
    if len(positions) == 0:
      return ret

    # else get the new vertex v0 in original_edge = (a, b) which has become
    # (a, new_index(v0)), (new_index(v0), b)
    v0 = tuple(np.round((np.stack([self.patchverts[e] for e in original_edge]) * np.array([1 - position, position])[:, None]).sum(0), 5))

    # since positions are sorted, we know for sure that the other positions
    # will now refine newedge = (new_index(v0), b)
    newedge = (map_vertex_index[v0], original_edge[1])

    # change the positions on the new edge, i.e., position
    # .6 has a new position on edge (new_index(v0), b) after refining
    # (a, b) at position pos < .6
    positions = tuple( (pos - position) / (1 - position) for pos in positions )
    return ret.refine(newedge, positions=positions)

  def qplot(self, show=True, show_axis=True, ax=None, **kwargs):
    if ax is None:
      fig, ax = plt.subplots()
    fig = ax.figure
    if not show_axis:
      ax.set_axis_off()
    ax.set_aspect('equal')
    mapped_verts = set()
    for i, patch in enumerate(self.patches, 1):
      myvertices = [self.patchverts[j] for j in patch]
      center = np.stack(myvertices).sum(0) / len(myvertices)
      XY = np.stack([myvertices[0], myvertices[2],
                     myvertices[3], myvertices[1], myvertices[0]])
      ax.plot(*XY.T, color='k')
      ax.fill(*XY.T, color='b', alpha=.1)
      vertices = set(patch) - mapped_verts
      for index in vertices:
        vertex = np.asarray(self.patchverts[index])
        position = center + 1.1 * (vertex - center)
        ax.text(*position, str(index), fontsize=8, color='r', zorder=5)
        ax.scatter(*vertex, s=20, color='r', zorder=4)
        mapped_verts.update({index})
      ax.text(*center, str([i]), fontsize=10, color='k')
    for pair, color in zip(self.periodic_edges, itertools.cycle(['b', 'g'])):
      for edge in pair:
        plt.plot( *np.stack([self.patchverts[e] for e in edge]).T, color=color, linewidth=3, zorder=3 )
    if not show:
      return fig, ax
    plt.show()

  def to_EdgeNetwork(self):
    return MultiPatchTemplate2EdgeNetwork(self)

  def to_MultiPatchBSplineGridObject(self, knotvectors, **kwargs):
    if isinstance(knotvectors, int):
      knotvectors = ko.KnotObject(np.linspace(0, 1, knotvectors))
    if isinstance(knotvectors, ko.KnotObject):
      knotvectors = dict(zip(self.knotvector_edges, [knotvectors]*len(self.knotvector_edges)))
    from mapping import mul
    return mul.MultiPatchBSplineGridObject(self.patches, self.patchverts, knotvectors, **kwargs)

  # canvas operations (derived from network.EdgeNetwork)

  def select_edges(self, *args, **kwargs):
    return self.to_EdgeNetwork().select_edges(*args, **kwargs)

  def select_vertices(self, *args, **kwargs):
    return self.to_EdgeNetwork().select_vertices(*args, **kwargs)

  def select_faces(self, *args, **kwargs):
    return self.to_EdgeNetwork().select_faces(*args, **kwargs)

  def take_faces(self, *args, **kwargs):
    return self.to_EdgeNetwork().take_faces(*args, **kwargs)

  def drag_vertices(self):
    with DragTemplateVertices(self) as handler:
      patchverts = handler.patchverts
    return self.edit(patchverts=patchverts)

  def create_hole(self, edge):
    raise NotImplementedError
    edge = tuple(edge)

    obe = set(self.ordered_boundary_indices)
    assert edge not in obe and edge[::-1] not in obe

    max_index = len(self.patchverts)
    new_edge = (max_index, max_index + 1)
    map_index_new_index = dict(zip(edge, new_edge))

    patchverts = self.patchverts + tuple( self.patchverts[e] for e in edge )

    # find the two patches that contain the edge
    ipatch0, ipatch1 = [i for i, patch in enumerate(self.patches) if set(edge).issubset(set(patch))]
    patches = [ patch if i != ipatch0 else tuple(map_index_new_index.get(j, j) for j in patch)
                for i, patch in enumerate(self.patches) ]

    knotvector_edges = self._knotvector_edges + (new_edge,)

    return self.edit(patchverts=patchverts,
                     patches=patches,
                     knotvector_edges=knotvector_edges,
                     periodic_edges=self.periodic_edges,
                     undo=self)


def _find_knotvector_edges(template: MultiPatchTemplate) -> tuple:
  all_edges = set(map(lambda x: tuple(sorted(x)), template.edges))
  boundary_edges = set(map(lambda x: tuple(sorted(x)), template.boundary_edges))

  root = boundary_edges.pop()

  opposites = template.opposite_edges

  knotvector_edges = []

  while True:

    if len(boundary_edges) == len(all_edges) == 0:
      break

    # pop boundary edge
    root = boundary_edges.pop() if len(boundary_edges) else all_edges.pop()

    knotvector_edges.append(root)

    myopposites = set(map(lambda x: tuple(sorted(x)), opposites[root]))
    mymapped = myopposites | {root}

    boundary_edges = boundary_edges - mymapped
    all_edges = all_edges - mymapped

  return tuple(knotvector_edges)


def select_indices(template, network):
  assert len(network.faces) == 1
  pols, = network.faces.values()
  if len(pols) == 1:
    return select_indices_genus0(template, network)
  elif len(pols) == 2:
    return select_indices_genus1(template, network)
  else:
    raise NotImplementedError


def select_indices_genus0(template, network):
  assert len(template.periodic_edges) == 0
  (face_index,), ((pol_index,),) = zip(*network.faces.items())
  indices, = network.pol_indices.values()
  ordered_bindices, = template.ordered_boundary_indices
  edge0 = template.get_edges(ordered_bindices[0])

  from canvas import select_edges

  if not len(indices) == len(ordered_bindices):
    raise NotImplementedError('The number of face edges must match the number of template boundary edges.')

  fig, axes = plt.subplots(nrows=1, ncols=2)
  fig, ax0 = template.qplot(ax=axes[0], show=False)
  ax0.plot( *np.array([template.patchverts[e] for e in edge0 ]).T, zorder=5, color='r', linewidth=3 )
  clicked_index, = select_edges(network, face_index,
                                ax=axes[1],
                                nclicks=1,
                                title='Select the edges that correspond to the'
                                      'highlighted edge in the domain')
  assert clicked_index in as_IndexSequence(indices)
  return (tuple(ordered_bindices),), (tuple(indices.roll_to(clicked_index)),), template


def select_indices_genus1(template, network):
  assert not template.periodic_edges
  (face_index,), ((pol_index0, pol_index1),) = zip(*network.faces.items())
  pol_indices0, pol_indices1 = (network.pol_indices[index] for index in (pol_index0, pol_index1))
  obi, = map(as_IndexSequence, template.ordered_boundary_indices)
  template_network = template.to_EdgeNetwork().boundary_network()

  assert len(pol_indices0) + len(pol_indices1) < len(obi)

  from canvas import select_edges

  sides = []
  clickable_indices = set(obi)
  for indices in [pol_indices0, -pol_indices1]:
    index0, index1 = indices[0], indices[-1]
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig, ax0 = network.qplot(ax=axes[0], show=False)
    ax0.plot( *network.get_edges(index0).points.T, zorder=10, color='r', linewidth=3 )
    ax0.plot( *network.get_edges(index1).points.T, zorder=10, color='b', linewidth=3 )
    clicked_index, = select_edges(template_network, face_index,
                                  ax=axes[1],
                                  nclicks=1,
                                  indices=list(clickable_indices),
                                  title='Select the edge that correspond to first'
                                        ' edge in the sequence from red to blue.')
    obi = obi.roll_to(clicked_index)
    split_index = len(indices)
    mysides, obi = obi[:split_index], obi[split_index:]
    sides.append(tuple(mysides))
    assert len(mysides) == len(indices)
    clickable_indices = clickable_indices - set(mysides)

  obi, = map(as_IndexSequence, template.ordered_boundary_indices)
  obi = obi.roll_to(sides[0][0])
  i0 = obi.tuple_index(sides[0][-1])
  i1 = obi.tuple_index(sides[1][0])
  i2 = obi.tuple_index(sides[1][-1])

  pindices0, pindices1 = obi[i0+1: i1], obi[i2+1:]
  periodic_edges = list(zip(template.get_edges(pindices0), template.get_edges(-pindices1)))

  return tuple(sides), (pol_indices0, pol_indices1), template.add_periodic_edges(periodic_edges)


class MultiPatchTemplate_(SaveLoadMixin):

  @cached_property
  def all_edges(self):
    return tuple(itertools.chain(*[list(map(tuple, map(sorted, get_edges(patch)))) for patch in self.patches]))

  def to_MultiPatchBSplineGridObject(self, knotvectors, **kwargs):
    if isinstance(knotvectors, int):
      knotvectors = ko.KnotObject(np.linspace(0, 1, knotvectors))
    if isinstance(knotvectors, ko.KnotObject):
      knotvectors = dict(zip(self.knotvector_edges, [knotvectors]*len(self.knotvector_edges)))
    from multipatch import MultiPatchBSplineGridObject
    return MultiPatchBSplineGridObject(self.patches, self.patchverts, knotvectors, **kwargs)

  def to_network(self):
    return patches_verts_to_network(self.patches, self.patchverts)

  def silhouette(self, **kwargs):
    return silhouette(self.patches, self.patchverts, **kwargs)

  @cached_property
  def patch_matrix(self):
    edge_to_patch, patch_to_edge = defaultdict(list), defaultdict(list)

    for i, patch in enumerate(self.patches, 1):
      for edge in (patch[:2], patch[2:], (patch[0], patch[2]), (patch[1], patch[3])):
        edge_to_patch[edge].append(i)
        patch_to_edge[i].append(edge)
        edge_to_patch[edge[::-1]].append(-i)
        patch_to_edge[-i].append(edge[::-1])

    patch_neighbours = {}
    for i in range(1, len(self.patches) + 1):
      myneighbours = {}
      for side, edge in zip(('left', 'right', 'bottom', 'top'), patch_to_edge[i]):
        for patch in set(edge_to_patch[edge]) - {i}:
          myneighbours[side] = patch
      patch_neighbours[i] = myneighbours

    patch_matrix = np.zeros([len(self.patches), 4], dtype=int)
    for index, neighbours in patch_neighbours.items():
      for side, neighbour in neighbours.items():
        patch_matrix[index-1, {'left': 0, 'right': 1, 'bottom': 2, 'top': 3}[side]] = neighbour

    return patch_matrix

  @property
  def is_valid(self):
    return all( (neighbours >= 0).all() for neighbours in self.patch_matrix )

  def edge_neighbours(self, edge):
    return edge_neighbours(self.patches, edge)

  def trampoline_refine(self, npatch, w0=.5, w1=.5, h0=.5, h1=.5, w=None, h=None):
    if w is not None:
      w0 = w1 = w
    if h is not None:
      h0 = h1 = h
    assert 0 < h0 < 1 and 0 < h1 < 1 and 0 < w0 < 1 and 0 < w1 < 1
    assert 0 <= npatch < len(self.patches)
    patch = self.patches[npatch]
    verts = np.stack([ self.patchverts[i] for i in patch ])
    P0, P1, P2, P3 = verts
    x = lambda xi, eta: (1 - eta) * (P0 + xi * (P2 - P0)) + eta * (P1 + xi * (P3 - P1)) + \
                        (1 - xi) * (P0 + eta * (P1 - P0)) + xi * (P2 + eta * (P3 - P2)) + \
                        -(1 - xi) * (1 - eta) * P0 - \
                        xi * eta * P3 - xi * (1 - eta) * P2 - (1 - xi) * eta * P1
    inner_verts = np.stack([ x(.5 - w0 * .5, .5 - h0 * .5),
                             x(.5 - w0 * .5, .5 + h1 * .5),
                             x(.5 + w1 * .5, .5 - h0 * .5),
                             x(.5 + w1 * .5, .5 + h0 * .5) ], axis=0)
    # inner_verts = (verts - center[None]) * np.array([fac, fac])[None] + center[None]
    n = len(self.patchverts)
    patchverts = self.patchverts + tuple(inner_verts)
    patches = self.patches[:npatch] + self.patches[npatch + 1:]
    inner_patches = ( (n, patch[0], n+1, patch[1]),
                      (n, n+2, patch[0], patch[2]),
                      (n, n+1, n+2, n+3),
                      (n+1, patch[1], n+3, patch[3]),
                      (n+2, n+3, patch[2], patch[3]) )
    patches = self.patches[:npatch] + inner_patches[:1] + self.patches[npatch + 1:] + inner_patches[1:]
    knotvector_edges = self._knotvector_edges + ((patch[0], n),)

    return self.__class__(patches, patchverts, knotvector_edges)

  def flip_patch_orientation(self, patch):
    patches = list(self.patches)
    patches[patch] = patches[patch][::-1]
    return self.__class__(patches, self.patchverts, self._knotvector_edges)

  def repair_orientation(self, i=10):
    if self.is_valid: return self

    if i == 0:
      raise RuntimeError('Failed to repair the orientation.')

    patch_matrix = self.patch_matrix

    # count the times (i+1) appears negatively and subtract the number of times it appears positively
    degree = lambda i: len(np.where(patch_matrix == -i - 1)[0]) - len(np.where(patch_matrix == i + 1)[0])
    repair_patches = sorted([i for i, patch in enumerate(patch_matrix) if (patch < 0).any()],
                            key=degree, reverse=True)

    return self.flip_patch_orientation(repair_patches[0]).repair_orientation(i=i-1)

  def take(self, indices):
    patches = [self.patches[i] for i in indices]
    allverts = sorted(set(itertools.chain(*patches)))
    map_old_verts_new = dict(zip(allverts, range(len(allverts))))
    patches = tuple( [map_old_verts_new[i] for i in patch] for patch in patches )
    patchverts = [self.patchverts[i] for i in allverts ]
    knotvector_edges = [ [map_old_verts_new[e] for e in edge] for edge in self._knotvector_edges if all(e in allverts for e in edge) ]
    return self.__class__(patches, patchverts, knotvector_edges)

  def drag_vertices(self):
    with DragTemplateVertices(self) as handler:
      patchverts = handler.patchverts
    return self.__class__(self.patches, patchverts, self._knotvector_edges)


def singlepatch_template():
  patches = ((0, 1, 2, 3),)
  patchverts = ((0, 0), (0, 1), (1, 0), (1, 1))
  knotvector_edges = ( (0, 1), (0, 2) )
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def chessboard_template(nrows, ncols, width=1, height=1):

  assert nrows >= 1 and ncols >= 1

  pmatrix = np.arange((nrows+1) * (ncols+1), dtype=int).reshape((ncols+1, nrows+1))

  patches = tuple( (pmatrix[i, j], pmatrix[i, j+1], pmatrix[i+1, j], pmatrix[i+1, j+1])
                   for i in range(ncols) for j in range(nrows) )

  cols = np.ones((nrows+1,), dtype=float)
  rows = height * np.arange(nrows+1)

  patchverts = np.concatenate([ np.stack([j * width * cols, rows], axis=1) for j in range(ncols+1) ])
  knotvector_edges = tuple( item for item in zip(pmatrix[:, 0], pmatrix[1:, 0]) )
  knotvector_edges = knotvector_edges + tuple( item for item in zip(pmatrix[0, :], pmatrix[0, 1:]) )

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def trampoline_template(inner_height=.5, inner_width=.5):
  """
    1 - - - - - 7
    | \   D   / |
    |  3 - - 5  |
    |A |  C  | E|
    |  |     |  |
    |  2 - - 4  |
    | /   B   \ |
    0 - - - - - 6

  """
  assert 0 < inner_height < 1
  assert 0 < inner_width < 1

  h, w = (1 - inner_height) / 2, (1 - inner_width) / 2

  patches = ( (2, 0, 3, 1),
              (2, 4, 0, 6),
              (2, 3, 4, 5),
              (3, 1, 5, 7),
              (4, 5, 6, 7) )

  patchverts = ( (0, 0), (0, 1),
                 (w, h), (w, 1 - h),
                 (1 - w, h), (1 - w, 1 - h),
                 (1, 0), (1, 1) )
  knotvector_edges = ( (0, 1), (0, 6), (0, 2) )
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def incomplete_trampoline_template(orientation=0):
  """
    1 -3 - - 5 -7
    |A |  C  | E|
    |  |     |  |
    |  2 - - 4  |
    | /   B   \ |
    0 - - - - - 6

  """
  patches = ( (2, 0, 3, 1),
              (2, 4, 0, 6),
              (2, 3, 4, 5),
              (4, 5, 6, 7) )

  patchverts = ( (0, 0), (0, 3),
                 (1, 1), (1, 3),
                 (3, 1), (3, 3),
                 (4, 0), (4, 3) )
  knotvector_edges = ( (0, 1), (0, 6), (0, 2) )
  ret = MultiPatchTemplate(patches, patchverts, knotvector_edges).stretch([1, 0], .25).stretch([0, 1], 1/3)
  if orientation == 0:
    return ret
  return ret.rotate(np.pi).translate([1, 1])


def extended_trampoline_template():
  """ extend the trampoline template by two sides running from 1 and 6 in the direction of (5, 7)"""

  A = trampoline_template()
  patches, patchverts, knotvector_edges = A.patches, A.patchverts, A.knotvector_edges
  patches += ((1, 8, 7, 9), (6, 7, 10, 9))
  patchverts += ((0, 5), (5, 5), (5, 0))
  knotvector_edges += ((1, 8),)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def diamond_template(height=1):
  """
       4
     /   \
   1   C   6
   | \   / |
   |   3   |
   | A | B |  h
   |   |   |
   |   |   |
   0 - 2 - 5
     w   w

  """

  patches = ( (0, 1, 2, 3),
              (2, 3, 5, 6),
              (1, 4, 3, 6) )
  patchverts = ( (0, 0), (0, height),
                 (.5, 0), (.5, height - .5),
                 (.5, height + .5), (1, 0), (1, height) )
  knotvector_edges = ((0, 1), (0, 2), (2, 5))
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def double_diamond_template():
  """
       5
     /   \   h4
   1   C   7
   | \   / | h3
   |   4   |
   | A | D | h2
   |   3   |
   | /   \ | h1
   0   B   6
     \   / | h0
       2 _ .
     w   w

  Well-suited for tuble-like shaped domains


  dims = (h0, h1, h2, h3, h4, w)

  """
  patches = ( (0, 1, 3, 4),
              (0, 3, 2, 6),
              (1, 5, 4, 7),
              (3, 4, 6, 7) )
  patchverts = ( (0, 0), (0, 3), (1, -1), (1, 1),
                 (1, 2), (1, 4), (2, 0), (2, 3) )
  knotvector_edges = ((0, 1), (0, 2), (2, 6))
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def n_leaf_template(N=3):
  """
                  4
                X X X
              X   X  X
             X C  X   X
            5     6  B 3
            X   X  X   X
            X X      X X
            0     A    2
              X      X
                X  X
                 1
  """

  assert N >= 3
  xi = np.linspace(0, 2*np.pi, 2*N+1)[:-1]
  nouter = len(xi)
  patchverts = tuple(map(tuple, np.stack([np.cos(xi), np.sin(xi)], axis=1))) + ((0, 0),)
  patches = tuple( (nouter, (2*(i+1)) % nouter, (2*i) % nouter, (2*i + 1) % nouter) for i in range(N) )
  knotvector_edges = tuple( patch[:2] for patch in patches )
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def even_n_leaf_template(N=6):
  """
    ASCII art forthcoming.
  """
  assert N % 2 == 0
  assert N >= 6
  xi = np.linspace(0, 2*np.pi, N+1)[:-1]
  ring = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  nring = len(xi)
  patchverts = tuple(map(tuple, ring)) + tuple(map(tuple, ring/2)) + ((0, 0),)
  if N == 6:
    patches = ( (0, 6, 1, 7), (2, 1, 8, 7), (3, 2, 9, 8),
                (4, 3, 10, 9), (4, 10, 5, 11), (5, 11, 0, 6),
                (12, 8, 6, 7), (10, 9, 12, 8), (10, 12, 11, 6) )
  elif N == 8:
    patches = ( (0, 8, 1, 9), (1, 9, 2, 10), (2, 10, 3, 11),
                (4, 3, 12, 11), (4, 12, 5, 13), (6, 5, 14, 13),
                (7, 6, 15, 14), (0, 7, 8, 15), (8, 16, 9, 10),
                (16, 12, 10, 11), (16, 14, 12, 13), (8, 15, 16, 14) )
  elif N == 10:
    patches = ( (0, 10, 1, 11), (1, 11, 2, 12), (2, 12, 3, 13),
                (4, 3, 14, 13), (5, 4, 15, 14), (6, 5, 16, 15),
                (7, 6, 17, 16), (7, 17, 8, 18), (9, 8, 19, 18),
                (9, 19, 0, 10), (10, 20, 11, 12), (20, 14, 12, 13),
                (16, 15, 20, 14), (17, 16, 18, 20), (19, 18, 10, 20) )
  elif N == 12:
    patches = ( (0, 12, 1, 13), (1, 13, 2, 14), (3, 2, 15, 14),
                (4, 3, 16, 15), (5, 4, 17, 16), (5, 17, 6, 18),
                (6, 18, 7, 19), (7, 19, 8, 20), (9, 8, 21, 20),
                (10, 9, 22, 21), (11, 10, 23, 22), (11, 23, 0, 12),
                (12, 24, 13, 14), (16, 15, 24, 14), (17, 16, 18, 24),
                (18, 24, 19, 20), (22, 21, 24, 20), (23, 22, 12, 24) )
  else:
    raise NotImplementedError

  n = len(patchverts) - 1
  knotvector_edges = tuple( (n, nring+i) for i in range(0, nring, 2) ) + ((nring, 0),)
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def diamond_with_flaps_template(height=1, flap_extend=.5):
  assert 0 < flap_extend < 1
  A = diamond_template(height=height)
  patchverts = A.patchverts

  y_v4 = patchverts[4][1]
  x_v1, y_v1 = patchverts[1]
  x_v6, y_v6 = patchverts[6]

  v7 = (x_v1, (1 - flap_extend) * y_v1 + flap_extend * y_v4)
  v8 = (flap_extend * 0 + (1 - flap_extend) * .5, y_v4)

  v9 = (.5 * (1 - flap_extend) + 1 * flap_extend, y_v4)
  v10 = (1, y_v6 * (1 - flap_extend) + y_v4 * flap_extend)

  patches = A.patches + ((7, 8, 1, 4), (4, 9, 6, 10))
  patchverts = patchverts + (v7, v8, v9, v10)

  return MultiPatchTemplate(patches, patchverts, A._knotvector_edges + ((1, 7), (6, 10)))


def diamond_with_flaps_triangle_template(height=1, flap_extend=.5, triangle_height=None):
  if triangle_height is None:
    triangle_height = height * .5
  assert 0 < triangle_height

  A = diamond_with_flaps_template(height=height, flap_extend=flap_extend)
  patchverts = np.array(A.patchverts)
  v14 = np.array([.5, -np.sqrt(3 / 4) * triangle_height])
  v11 = (patchverts[0] + v14) / 2
  v13 = (patchverts[5] + v14) / 2
  v12 = (patchverts[0] + patchverts[5] + v14) / 3

  patchverts = np.concatenate([ patchverts, v11[None], v12[None], v13[None], v14[None] ])
  patches = A.patches + ( (11, 0, 12, 2), (12, 2, 13, 5), (11, 12, 14, 13) )
  knotvector_edges = A.knotvector_edges + ((11, 0),)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def triangle_template(height=1):

  p0, p1, p2 = np.array([0, 0]), np.array([1, 0]), np.array([.5, np.sqrt(3 / 4)]) * np.array([1, height])
  p3, p4, p5 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2
  p6 = (p0 + p1 + p2) / 3

  patchverts = p0, p1, p2, p3, p4, p5, p6
  patches = (0, 5, 3, 6), (3, 6, 1, 4), (5, 2, 6, 4)
  knotvector_edges = (0, 3), (3, 1), (0, 5)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def railroad_switch_template(height=1):
  """ width = 2 """

  patchverts = ( (0, 0), (0, .5),
                 (0, 1), (1, 0),
                 (.5, .5), (1, 1),
                 (1.5, .5), (2, 0),
                 (2, .5), (2, 1) )

  patches = ( (0, 1, 3, 4),
              (1, 2, 4, 5),
              (3, 4, 6, 5),
              (3, 6, 7, 8),
              (6, 5, 8, 9) )

  knotvector_edges = (0, 3), (0, 1), (1, 2), (3, 7)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def hat_template(height=1, triangle_height=1):

  A = singlepatch_template().refine((0, 2), .5).stretch([0, 1], height)
  B = triangle_template(height=triangle_height)

  return MultiPatchTemplate.fuse(A, B.translate([0, height]))


def hat_with_flaps_template(height=1, triangle_height=1, flap_height=None, flap_extend=.5):
  if flap_height is None:
    flap_height = height / 2

  A = hat_template(height=height, triangle_height=triangle_height)

  patches = A.patches
  patchverts = np.array(A.patchverts)
  knotvector_edges = A.knotvector_edges

  patchverts[4] -= np.array([0, flap_height])
  y_v4 = patchverts[4][1]
  x_v0, y_v0 = patchverts[0]
  x_v2, y_v2 = patchverts[2]

  v10 = np.array([0, y_v0 * (1 - flap_extend) + y_v4 * flap_extend])
  v11 = np.array([.5 * (1 - flap_extend) + flap_extend * 0, y_v4])

  v12 = np.array([.5 * (1 - flap_extend) + 1 * flap_extend, y_v4])
  v13 = np.array([1, (1 - flap_extend) * y_v2 + flap_extend * y_v4])

  patchverts = np.concatenate([patchverts, v10[None], v11[None], v12[None], v13[None]])
  patches = patches + ( (10, 0, 11, 4), (12, 4, 13, 2) )
  knotvector_edges = knotvector_edges + ((10, 0), (12, 4))

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def apply_hat(network, face_index):

  snetwork = network.take([face_index])
  indices = abs(network.face_indices[face_index])

  A = hat_template()

  index0 = abs(select_index0(A.patches, A.patchverts, snetwork))

  indices = tuple(indices.roll_to(index0))

  index3 = abs(select_index(A.patches, A.patchverts, snetwork, index=3))

  indices_right = np.array(indices[ indices.index(index0) + 2: indices.index(index3) ])
  indices_left = np.array(indices[ indices.index(index3) + 4: ])[::-1]
  indices_hat = np.array(indices[ indices.index(index3): indices.index(index3) + 2 ])

  edges_right = network.get_edges(indices_right)
  edges_left = network.get_edges(indices_left)
  edges_hat = network.get_edges(indices_hat)

  assert len(edges_right) == len(edges_left)
  n = len(edges_right) + 1

  lengths_right = np.array([edge.length for edge in edges_right])
  lengths_left = np.array([edge.length for edge in edges_left])
  lengths_hat = np.array([edge.length for edge in edges_hat])

  height = round((lengths_right.sum() + lengths_left.sum()) / 2, 5)

  left_vertices = np.stack([ np.zeros((n,)),
                             [0, *(height * lengths_left.cumsum() / lengths_left.sum())] ], axis=1)
  right_vertices = np.stack([ np.ones((n,)),
                              [0, *(height * lengths_right.cumsum() / lengths_right.sum())] ], axis=1)

  patches = tuple( (i, i+1, i+n, i+n+1) for i in range(n-1) )
  knotvector_edges = tuple( (i, i+1) for i in range(n-1) ) + ((0, n),)

  A = MultiPatchTemplate(patches=patches,
                         patchverts=np.concatenate([left_vertices, right_vertices], axis=0),
                         knotvector_edges=knotvector_edges).refine((0, n))
  B = triangle_template(height=lengths_hat.sum())

  C = MultiPatchTemplate.fuse(A, B.translate([0, height]))

  indices = network.face_indices[face_index]
  index0 = (index0 @ indices) * index0
  indices = indices.roll_to(index0)

  network.set_template(face_index, C, sides=[[i] for i in indices])


def double_hat_template(height=1, triangle_height0=1, triangle_height1=1):

  A = hat_template(height=height, triangle_height=triangle_height1)
  B = triangle_template()

  return MultiPatchTemplate.fuse(A, B.rotate(-np.pi/3).stretch([0, 1], triangle_height0))


def make_edgedict(knotvector_edges, ordered_boundary_edges, nfits, side_indices, stack_c=0):
  # ordere_boundary_edges is flattened. I.e., obe = ((elem0,), (elem1,), ...) becomes tuple(itertools.chain(*obe))
  assert stack_c >= -1
  fits = {}
  for mg_index, indices in zip(ordered_boundary_edges, side_indices):
    if (y := mg_index[::-1]) in knotvector_edges:
      mg_index, indices = y, -indices
    edge = SplineEdge.merge_SplineEdges(*[nfits[index] for index in indices]).g
    if len(indices) > 1:
      edge = edge.to_c([stack_c])
    fits[mg_index] = edge
  return fits


class FaceTemplate:

  def __init__(self, network, face_index, template, sides):
    assert set(itertools.chain(*sides)) == set(myindices := network.face_indices[face_index])
    assert len(tuple(itertools.chain(*sides))) == len(myindices)

    self.sides = tuple(map(as_IndexSequence, sides))
    assert all( side in myindices for side in self.sides )

    if len(as_IndexSequence(self.sides).ravel()) > len(self.sides):
      raise NotImplementedError('Assigning multiple edges to one side is prohibited for now.')

    self.template = template

    # re-instantiate
    self.template = MultiPatchTemplate(template.patches,
                                       template.patchverts,
                                       template.knotvector_edges)

    self.network = network
    self.face_index = tuple(face_index)

  def edit(self, **kwargs):
    return self.__class__(**{**{'network': self.network,
                                'face_index': self.face_index,
                                'template': self.template,
                                'sides': self.sides}, **kwargs})

  def tolib(self):
    """ For saving purposes """
    return {'template': self.template,
            'sides': tuple(map(tuple, self.sides))}

  @property
  def patches(self):
    return self.template.patches

  @property
  def patchverts(self):
    return self.template.patchverts

  @property
  def snetwork(self):
    return self.network.take((self.face_index,))

  def boundary_map(self):
    obe, = self.template.ordered_boundary_edges
    sides = self.sides
    return { **dict(zip(obe, sides)),
             **dict(zip([edge[::-1] for edge in obe], [-i for i in sides])) }

  def edge_pairings(self):
    boundary_map = self.boundary_map()
    pairings, matched = [], set()
    obe, = self.template.ordered_boundary_edges
    for edge in obe:
      if len({edge, edge[::-1]} & matched) == 0:
        neighbour, = [otheredge for otheredge in edge_neighbours(self.template.patches, edge) if otheredge in boundary_map ]
        pairings.append(( boundary_map[edge], boundary_map[neighbour] ))
        matched.update({edge, neighbour})
    return tuple(pairings)

  def make_MultipatchBSplineGridObject(self, stack_c=0, knotvectors=None, **kwargs):
    snetwork = self.snetwork
    indices, = list(snetwork.face_indices.values())
    from multipatch import MultiPatchBSplineGridObject
    from fit import multipatch_boundary_fit_from_univariate_gos
    obe = tuple( itertools.chain(*self.template.ordered_boundary_edges) )
    boundary_gos = make_edgedict(self.template.knotvector_edges,
                                 obe,
                                 snetwork.edgedict,
                                 self.sides,
                                 stack_c=stack_c)
    # if no dictionary of knotvectors is passed, make a knotvector containing the average
    # number of knots of the gos in boundary_gos
    if knotvectors is None:
      knotvectors = max(2, int(sum(len(val.knots[0]) for val in boundary_gos.values()) / len(boundary_gos)) - 1)
    if not isinstance(knotvectors, dict):  # None => std.KnotObject, int: uniform with n elements
      knotvectors = as_KnotObject(knotvectors)
    if isinstance(knotvectors, ko.KnotObject):
      knotvectors = dict(zip(y := self.template.knotvector_edges, [knotvectors]*len(y)))
    assert set(knotvectors.keys()) == set(self.template.knotvector_edges)
    mg = MultiPatchBSplineGridObject(self.template.patches, self.template.patchverts, knotvectors, **kwargs)
    return multipatch_boundary_fit_from_univariate_gos(mg, boundary_gos)

  make_MultiPatchBSplineGridObject = make_MultipatchBSplineGridObject

  def make_edgedict(self, **kwargs):
    obe = tuple( itertools.chain(*self.template.ordered_boundary_edges) )
    return make_edgedict(self.template.knotvector_edges,
                         obe, self.snetwork.edgedict,
                         self.sides,
                         **kwargs)


def face_template(network, face_index, template, template_args=None, sides=None):

  template_args = dict(template_args or {})

  if isinstance(template, str):
    template_constructor = {'singlepatch': singlepatch_template,
                            'diamond': diamond_template,
                            'double_diamond': double_diamond_template,
                            'n_leaf': n_leaf_template,
                            'even_n_leaf': even_n_leaf_template}[template]

    template = template_constructor(**template_args)
  else:
    assert not bool(template_args)

  assert template.__class__.__name__ == 'MultiPatchTemplate'

  obe, = template.ordered_boundary_edges
  if not len(network.face_indices[face_index]) == len(obe):
    raise NotImplementedError('The number of face edges must match the number of template boundary edges.')

  if sides is None:
    snetwork = network.take((face_index,))
    sides = select_indices(template.patches, template.patchverts, snetwork)

  return FaceTemplate(network, face_index, template, sides)
