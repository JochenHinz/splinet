from .index import as_OrientableTuple, as_OrientableFloat, Edge, as_IndexSequence, \
                   SaveLoadMixin, as_KnotObject, SplineEdge, FrozenDict
from .canvas import MatplotlibEventHandler, SelectVertices, \
                    GenerateVertexFromClick

from collections import defaultdict, deque
from functools import wraps, lru_cache, cached_property
from itertools import count, chain, product
from mapping import ko
import itertools
import numpy as np
from matplotlib import pyplot as plt
from nutils import function


# VERTEX-EDGE NETWORK


def normalize_split_index(c, i):
  return as_OrientableTuple(c), as_OrientableFloat(i)


def edge_neighbour_faces(network, edge):
  edge = as_OrientableTuple(edge)
  assert edge @ network.indices == 1
  neighbours = {edge}
  newneighbours = neighbours.copy()
  faces = defaultdict(set)
  while True:
    for neigh in neighbours:
      for face_index, patchverts in network.face_indices.items():
        pairing = neigh @ patchverts
        if pairing == 0: continue
        index = tuple(abs(patchverts)).index(abs(neigh))
        otheredge = -pairing * patchverts[(index + 2) % 4]
        newneighbours.update({ otheredge })
        faces[face_index].update({neigh, otheredge})
    if len(newneighbours) == len(neighbours): break
    neighbours = newneighbours.copy()
  return {key: tuple(value) for key, value in faces.items()}


def apply_pairing(func):

  @wraps(func)
  def wrapper(self, edge, position, *args, **kwargs):
    pairing = as_OrientableTuple(edge) @ self.indices
    assert pairing
    edge, position = pairing * as_OrientableTuple(edge), pairing * as_OrientableFloat(position)
    return func(self, edge, position, *args, **kwargs)

  return wrapper


# auxiliary function for multipatch domains


def get_edges(list_of_vertices):
  assert len(list_of_vertices) == 4
  list_of_vertices = np.asarray(list_of_vertices).reshape([2, 2])
  return tuple(map(tuple, np.concatenate([list_of_vertices, list_of_vertices.T])))


def get_all_edges(patches):
  return tuple(itertools.chain.from_iterable(list(map(get_edges, patches))))


def multipatch_boundary_edges(patches):
  alledges = get_all_edges(patches)
  return tuple( edge for edge in alledges if alledges.count(edge) + alledges.count(edge[::-1]) == 1 )


opposite_side = {0: 1, 1: 0, 2: 3, 3: 2}


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


def orient_knotvectors(knotvectors):
  """ Make sure that a dict of the form {edge: knotvector} is such that edge[1] > edge[0].
      In case edge[1] < edge[0], replace entry by (edge[1], edge[0]): knotvector.flip(). """
  ret = {}
  for (a, b), knotvector in knotvectors.items():
    if b > a: ret[(a, b)] = knotvector
    else: ret[(b, a)] = knotvector.flip()
  return ret


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
      myedge = Edge(np.stack([ patchverts[k] for k in edge ], axis=0))
      if myedge in edgedict: myindex = edgedict[myedge]
      elif -myedge in edgedict: myindex = -edgedict[-myedge]
      else:
        myindex = current_edge
        current_edge += 1
      edgedict[myedge] = myindex
      map_index_original_edge[myindex], map_index_original_edge[-myindex] = tuple(edge), tuple(edge)[::-1]
      face_indices[(i, 0)][j] = myindex if j in (0, 1) else -myindex  # bottom right are in positive direction, else negative
  edgedict = {index: edge for edge, index in edgedict.items()}
  from .network import EdgeNetwork
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
  from .canvas import select_edges
  fig, ax0 = silhouette(patches, patchverts, show=False, ax=axes[0], index0=index)
  key, = list(network.face_indices.keys())
  index0, = select_edges(network, key,
                         nclicks=1,
                         ax=axes[1],
                         title='Select the edge that corresponds to the'
                                'highlighted edge in the domain')
  return index0


select_index0 = lambda *args, **kwargs: select_index(*args, index=0, **kwargs)


def select_indices(patches, patchverts, network):
  assert len(network.face_indices) == 1
  from .canvas import select_edges
  (face_index,), (indices,) = zip(*network.face_indices.items())
  multipatch_network, _map = patches_verts_to_network(patches, patchverts)

  ordered_bindices = multipatch_network.ordered_boundary_indices[0]
  if not len(indices) == len(ordered_bindices):
    raise NotImplementedError('The number of face edges must match the number of template boundary edges.')

  unmatched_indices = indices
  sides = []
  for i, index in enumerate(ordered_bindices[:-1]):
    if i > 0 and len(unmatched_indices) == len(ordered_bindices[i:]):
      sides += [(i,) for i in unmatched_indices]
      break
    if len(unmatched_indices) < len(ordered_bindices[i:]): raise AssertionError
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig, ax0 = silhouette(patches, patchverts, index0=i, show=False, ax=axes[0])
    myindices = select_edges(network, face_index,
                             ax=axes[1],
                             indices=unmatched_indices,
                             title='Select the edges that correspond to the'
                                    'highlighted edge in the domain')
    assert as_IndexSequence(myindices) in as_IndexSequence(indices)
    sides.append(myindices)
    unmatched_indices = unmatched_indices.roll_to(myindices[0])[len(myindices):]
  else:
    sides.append(unmatched_indices)
  return tuple(map(tuple, sides))


def infer_knotvector_edges(patches, knotvector_edges):
  assert all( len(patch) == 4 for patch in patches )
  assert set(itertools.chain(*knotvector_edges)).issubset(set(itertools.chain(*patches)))
  edges, newedges = list(knotvector_edges), list(knotvector_edges)

  while True:
    for edge in edges:
      myneighbours = edge_neighbours(patches, edge)
      for neighbour in myneighbours:
        if neighbour not in newedges:
          newedges.append(neighbour)
    if len(newedges) == len(edges): break
    edges = newedges.copy()

  assert all(edge[::-1] not in newedges for edge in newedges)

  return tuple(newedges)


def get_patch_edges(patch):
  return (patch[:2], patch[2:], (patch[0], patch[2]), (patch[1], patch[3]))


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


@lru_cache(maxsize=64)
def _ordered_boundary_edges(temp):
  """ By default the ordered boundary indices are ordered in such a way that the first edge edge0 = [i0, i1]
      is the one with the property that i0 equals the smallest index in the sequence. """
  network, map_index_original_edge = temp.to_network()
  bindices = network.ordered_boundary_indices
  ret_ = tuple(tuple(map_index_original_edge[index] for index in bindex) for bindex in bindices)
  ret = []
  for subseq in ret_:
    minindex = min(chain(*subseq))
    index, = [i for i, item in enumerate(subseq) if item[0] == minindex]
    ret.append(subseq[index:] + subseq[:index])
  return tuple(ret)


class OrderedBoundaryEdgesMixin:

  @property
  def ordered_boundary_edges(self):
    return _ordered_boundary_edges(self)


class MultiPatchTemplate(SaveLoadMixin, OrderedBoundaryEdgesMixin):

  @staticmethod
  def fuse(template0, template1, tol=1e-7):
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

    return MultiPatchTemplate(patches, patchverts, knotvector_edges).repair_orientation()

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
    from .aux import angle_between_vectors_

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
    n = count(len(template0.patchverts))

    for i in range(len(template1.patchverts)):
      if i not in edge1:
        map_verts[i] = n.__next__()

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
            'knotvector_edges': self._knotvector_edges}

  def __init__(self, patches, patchverts, knotvector_edges):
    assert all( len(patch) == 4 for patch in patches )
    assert all( len(vert) == 2 for vert in patchverts )
    assert set(itertools.chain(*patches)) == set(range(len(patchverts)))
    self.patches = tuple(map(tuple, patches))
    self.patchverts = tuple(map(lambda x: tuple(np.round(x, 5)), patchverts))
    assert set(itertools.chain(*knotvector_edges)).issubset(set(range(len(patchverts))))
    self._knotvector_edges = tuple(map(tuple, knotvector_edges))
    self.knotvector_edges = infer_knotvector_edges(self.patches, self._knotvector_edges)

  def __hash__(self):
    return hash((self.patches, self.patchverts, self.knotvector_edges))

  def __eq__(self, other):
    return self.patches == other.patches and \
           self.patchverts == other.patchverts and \
           self.knotvector_edges == other.knotvector_edges

  @cached_property
  def all_edges(self):
    return tuple(itertools.chain(*[list(map(tuple, map(sorted, get_patch_edges(patch)))) for patch in self.patches]))

  def to_MultiPatchBSplineGridObject(self, knotvectors, **kwargs):
    if isinstance(knotvectors, int):
      knotvectors = ko.KnotObject(np.linspace(0, 1, knotvectors))
    if isinstance(knotvectors, ko.KnotObject):
      knotvectors = dict(zip(self.knotvector_edges, [knotvectors]*len(self.knotvector_edges)))
    from .multipatch import MultiPatchBSplineGridObject
    return MultiPatchBSplineGridObject(self.patches, self.patchverts, knotvectors, **kwargs)

  def to_network(self):
    return patches_verts_to_network(self.patches, self.patchverts)

  def silhouette(self, **kwargs):
    return silhouette(self.patches, self.patchverts, **kwargs)

  def translate(self, x):
    return self.__class__(self.patches,
                          np.asarray(self.patchverts) + np.asarray(x)[None],
                          self.knotvector_edges)

  def rotate(self, theta):
    A = np.array([ [np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)] ])
    return self.__class__(self.patches,
                          (A @ np.array(self.patchverts).T).T,
                          self.knotvector_edges)

  def stretch(self, vec, magnitude, x0=None):
    if x0 is None:
      x0 = np.array([0, 0])
    vec, x0 = np.asarray(vec), np.asarray(x0)
    patchverts = (np.array(self.patchverts) - x0[None])
    patchverts = (patchverts + (magnitude - 1) * (patchverts * vec[None])) + \
                 (magnitude - 1) * (vec * x0)[None]
    return self.__class__(self.patches, patchverts, self.knotvector_edges)

  def translate_vertex(self, index, vec):
    patchverts = np.array(self.patchverts, dtype=float)
    patchverts[index] += np.asarray(vec)
    return self.__class__(self.patches, patchverts, self.knotvector_edges)

  @cached_property
  def edge2patch(self):
    ret = {}
    for i, patch in enumerate(self.patches):
      for edge in (patch[:2], patch[2:], patch[::2], patch[1::2]):
        ret.setdefault(edge, []).append(i)
        ret.setdefault(edge[::-1], []).append(i)

    return FrozenDict({key: tuple(val) for key, val in ret.items()})

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

  def refine(self, edge, positions=.5):
    if np.isscalar(positions):
      positions = positions,

    if edge[0] > edge[1]:
      edge = edge[::-1]
      positions = tuple(1 - pos for pos in positions)

    position, *positions = sorted(positions)

    edge = tuple(edge)
    position = round(position, 5)
    assert edge in self.all_edges

    # get all sorted neighbours
    # neighbours = tuple(map(tuple, [edge] + list(map(sorted, self.edge_neighbours(edge)))))
    neighbours = tuple(map(tuple, [edge] + list(self.edge_neighbours(edge))))
    # refine_positions = { edge0: position if edge0 == edge1 else 1 - position for edge0, edge1 in zip(neighbours, [edge] + list(self.edge_neighbours(edge))) }
    # refine_positions.update({edge[::-1]: 1 - pos for edge, pos in refine_positions.items()})

    patches = {}
    m = len(self.patches)
    fac = np.array([1 - position, position])[:, None]

    for i, patch in enumerate(self.patches):
      myvertices = tuple(self.patchverts[i] for i in patch)
      myedges = tuple(map(tuple, get_patch_edges(patch)))
      for neigh in neighbours:
        if neigh in myedges: index = myedges.index(neigh)
        elif neigh[::-1] in myedges: index = myedges.index(neigh[::-1])
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
      else:
        patches[i] = myvertices

    all_vertices = list(set(itertools.chain(*list(patches.values()))))
    map_vertex_index = dict(zip(self.patchverts, range(len(self.patchverts))))

    n = len(map_vertex_index)
    for vert in all_vertices:
      if vert not in map_vertex_index:
        map_vertex_index[vert] = n
        n += 1

    patches = [ [map_vertex_index[vert] for vert in patches[i]] for i in range(len(patches)) ]
    map_index_vertex = {value: key for key, value in map_vertex_index.items()}
    patchverts = [ map_index_vertex[ind] for ind in range(len(map_index_vertex)) ]

    original_edge = edge

    knotvector_edges = []
    for edge in self._knotvector_edges:
      if edge in neighbours or edge[::-1] in neighbours:
        if edge[::-1] in neighbours:
          myfac = 1 - fac
        else:
          myfac = fac
        v0 = tuple(np.round((np.stack([self.patchverts[e] for e in edge]) * myfac).sum(0), 5))
        knotvector_edges.append( (edge[0], map_vertex_index[v0]) )
        knotvector_edges.append( (map_vertex_index[v0], edge[1]) )
      else:
        knotvector_edges.append(edge)

    ret = self.__class__(patches, patchverts, knotvector_edges)
    if len(positions) == 0:
      return ret

    v0 = tuple(np.round((np.stack([self.patchverts[e] for e in original_edge]) * np.array([1 - position, position])[:, None]).sum(0), 5))
    newedge = (map_vertex_index[v0], original_edge[1])
    positions = tuple( (pos - position) / (1 - position) for pos in positions )
    return ret.refine(newedge, positions=positions)

  def qplot(self, show=True, show_axis=True, c='b', fontsize=10, **kwargs):
    fig, ax = plt.subplots()
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
      ax.fill(*XY.T, color=c, alpha=.1)
      vertices = set(patch) - mapped_verts
      for index in vertices:
        vertex = np.asarray(self.patchverts[index])
        position = center + 1.1 * (vertex - center)
        if fontsize > 0:
          ax.text(*position, str(index), fontsize=fontsize, color='k', zorder=10)
        ax.scatter(*vertex, s=20, color='r', zorder=4)
        mapped_verts.update({index})
      if fontsize > 0:
        ax.text(*center, str([i]), fontsize=fontsize, color='k')
    if not show:
      return fig, ax
    plt.show()

  def qplot_(self):
    self.to_network()[0].qplot()

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

  def normalize(self):
    """ Regauge such that vol(tempalte) = 1 and center of mass(template) = [0, 0]. """
    mg = self.to_MultiPatchBSplineGridObject(knotvectors=2)
    volume = mg.integrate(function.J(mg.geom))
    patchverts = np.asarray(self.patchverts) / np.sqrt(volume)
    obi = np.asarray([edge[0] for edge in self.ordered_boundary_edges[0]], dtype=int)
    com = patchverts[obi].sum(0) / len(obi)
    return MultiPatchTemplate(self.patches, patchverts - com[None], self._knotvector_edges)

  def renumber(self):
    """ Renumber such that obi = [edge[0] for edge in self.obe] is an ascending
        sequence [0, 1, ...] """
    obe = self.ordered_boundary_edges
    obi = [ [edge[0] for edge in subseq] for subseq in obe ]
    assert all( len(set(obi0) & set(obi1)) == 0 for i, obi0 in enumerate(obi[:-1]) for obi1 in obi[i+1:] ), NotImplementedError
    index_map = {}
    counter = count()
    for myobi in obi:
      for index in myobi:
        index_map[index] = counter.__next__()
    for index in sorted(set(range(len(self.patchverts))) - set(index_map.keys())):
      index_map[index] = counter.__next__()
    inv = {value: key for key, value in index_map.items()}
    patches = [ [index_map[i] for i in patch] for patch in self.patches ]
    patchverts = [ self.patchverts[inv[i]] for i in range(len(self.patchverts)) ]
    knotvector_edges = [ [index_map[i] for i in edge] for edge in self._knotvector_edges ]
    return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def singlepatch_template():
  patches = ((0, 1, 2, 3),)
  patchverts = ((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5))
  knotvector_edges = ( (0, 1), (0, 2) )
  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


def optionally_normalize(fn):

  @wraps(fn)
  def wrapper(*args, normalize=True, **kwargs):
    ret = fn(*args, **kwargs)
    if normalize:
      ret = ret.normalize()
    return ret

  return wrapper


@optionally_normalize
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


@optionally_normalize
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


@optionally_normalize
def extended_trampoline_template():
  """ extend the trampoline template by two sides running from 1 and 6 in the direction of (5, 7)"""

  A = trampoline_template()
  patches, patchverts, knotvector_edges = A.patches, A.patchverts, A.knotvector_edges
  patches += ((1, 8, 7, 9), (6, 7, 10, 9))
  patchverts += ((0, 5), (5, 5), (5, 0))
  knotvector_edges += ((1, 8),)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


@optionally_normalize
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


@optionally_normalize
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


@optionally_normalize
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


@optionally_normalize
def even_n_leaf_template(N=6):
  """
    ASCII art forthcoming.
  """
  assert N % 2 == 0
  assert N >= 6
  xi = np.linspace(0, 2*np.pi, N+1)[:-1]
  ring = np.stack([np.cos(xi), np.sin(xi)], axis=1)
  nring = len(xi)
  patchverts = tuple(map(tuple, 2 * ring)) + tuple(map(tuple, ring)) + ((0, 0),)
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


@optionally_normalize
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


@optionally_normalize
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


@optionally_normalize
def triangle_template(height=1):

  p0, p1, p2 = np.array([0, 0]), np.array([1, 0]), np.array([.5, np.sqrt(3 / 4)]) * np.array([1, height])
  p3, p4, p5 = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p0) / 2
  p6 = (p0 + p1 + p2) / 3

  patchverts = p0, p1, p2, p3, p4, p5, p6
  patches = (0, 5, 3, 6), (3, 6, 1, 4), (5, 2, 6, 4)
  knotvector_edges = (0, 3), (3, 1), (0, 5)

  return MultiPatchTemplate(patches, patchverts, knotvector_edges)


@optionally_normalize
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


@optionally_normalize
def hat_template(height=1, triangle_height=1):

  A = singlepatch_template().refine((0, 2), .5).stretch([0, 1], height)
  B = triangle_template(height=triangle_height)

  return MultiPatchTemplate.fuse(A, B.translate([0, height]))


@optionally_normalize
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


def apply_double_hat(network, face_index):
  snetwork = network.take([face_index])
  indices = abs(network.face_indices[face_index])

  A = double_hat_template()

  index0 = abs(select_index0(A.patches, A.patchverts, snetwork))

  indices = tuple(indices.roll_to(index0))

  index5 = abs(select_index(A.patches, A.patchverts, snetwork, index=5))

  indices_hat0 = np.array(indices[:2])
  indices_right = np.array(indices[4: indices.index(index5)])
  indices_hat1 = np.array(indices[ indices.index(index5): indices.index(index5) + 2 ])
  indices_left = np.array(indices[ indices.index(index5) + 4: ])[::-1]

  edges_right = network.get_edges(indices_right)
  lengths_right = np.array([edge.length for edge in edges_right])

  edges_left = network.get_edges(indices_left)
  lengths_left = np.array([edge.length for edge in edges_left])

  assert len(edges_right) == len(edges_left)
  n = len(edges_right) + 1

  edges_hat0 = network.get_edges(indices_hat0)
  lengths_hat0 = np.array([ edge.length for edge in edges_hat0 ])

  edges_hat1 = network.get_edges(indices_hat1)
  lengths_hat1 = np.array([ edge.length for edge in edges_hat1 ])

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
  B = triangle_template().rotate(-np.pi/3).stretch([0, 1], lengths_hat0.sum())
  C = triangle_template(height=lengths_hat1.sum()).translate([0, height])

  # first edge: (0, 3)
  D = MultiPatchTemplate.fuse(B, MultiPatchTemplate.fuse(A, C))

  indices = network.face_indices[face_index]
  index0 = (index0 @ indices) * index0
  indices = indices.roll_to(index0)

  network.set_template(face_index, D, sides=[[i] for i in indices])


def apply_singlepatch_template(network, face_index):

  snetwork = network.take([face_index])
  indices = abs(network.face_indices[face_index])

  A = singlepatch_template()

  index0 = abs(select_index(A.patches, A.patchverts, snetwork, index=3))

  indices = tuple(indices.roll_to(index0))

  index1 = abs(select_index(A.patches, A.patchverts, snetwork, index=1))

  indices_bottom = np.array(indices[ indices.index(index0) + 1: indices.index(index1) ])
  indices_top = np.array(indices[indices.index(index1) + 1:])[::-1]
  edges_bottom = network.get_edges(indices_bottom)
  edges_top = network.get_edges(indices_top)

  edge_left = network.get_edges(indices[0])
  edge_right = network.get_edges(index1)

  length_left = edge_left.length
  length_right = edge_right.length

  lengths_bottom = np.array([edge.length for edge in edges_bottom])
  lengths_top = np.array([edge.length for edge in edges_top])

  p0 = np.array([0, 0])
  p1 = np.array([1, 0])
  p2 = np.array([1, length_right / length_left])
  p3 = np.array([0, 1])

  p_bottom = np.stack([ p0 * (1 - xi) + p1 * xi for xi in [0, *(lengths_bottom.cumsum() / lengths_bottom.sum())] ], axis=0)
  n = len(p_bottom)
  p_top = np.stack([ p3 * (1 - xi) + p2 * xi for xi in [0, *(lengths_top.cumsum() / lengths_top.sum())] ], axis=0)

  patchverts = np.concatenate([p_bottom, p_top], axis=0)

  patches = []
  knotvector_edges = [(0, n)]
  for i in range(n - 1):
    patches.append((i, i+n, i+1, i+1+n))
    knotvector_edges.append((i, i+1))

  A = MultiPatchTemplate(patchverts=patchverts, patches=patches, knotvector_edges=knotvector_edges)
  obe = A.ordered_boundary_edges[0]
  index_n0 = obe.index((n, 0))

  indices = network.face_indices[face_index]
  indices = indices.roll_to((index0 @ indices) * index0).roll(index_n0)

  network.set_template(face_index, A, sides=[[i] for i in indices])


def create_corner(network, face_index):

  snetwork = network.take([face_index])

  with SelectVertices(snetwork, nclicks=1, title="Select the corner's vertex.") as handler:
    vertex, = handler.clicked_vertices

  A = diamond_template()
  fig, axes = plt.subplots(nrows=1, ncols=2)

  title = 'Click on the position on the edge to create the highlighted edge.'
  fig, ax0 = silhouette(A.patches, A.patchverts, show=False, ax=axes[0], index0=3)
  with GenerateVertexFromClick(snetwork, title=title, ax=axes[1]) as handler:
    e, i = handler.ei

  indices = snetwork.face_indices[face_index]
  pairing = e @ indices
  if pairing == -1:
    e = -e
    i = len(network.get_edges(e).points) - 1 - i

  firstedge = network.get_edges(e)
  edge0, edge1 = firstedge.split(i)
  i = edge0.length / firstedge.length

  network = network.trefine(e, i)

  indices = indices.roll_to(e)

  nextindex = indices[1]
  nextedge = network.get_edges(nextindex)
  cumlength = nextedge.dx.cumsum()
  nextposition = nextedge.verts[np.argmin((cumlength - edge1.length)**2)]

  return network.trefine(nextindex, nextposition)


def create_triangle_corner(network, face_index):

  snetwork = network.take([face_index])

  with SelectVertices(snetwork, nclicks=1, title="Select the corner's vertex.") as handler:
    vertex, = handler.clicked_vertices

  A = diamond_template()
  fig, axes = plt.subplots(nrows=1, ncols=2)

  title = 'Click on the position on the edge to create the highlighted edge.'
  fig, ax0 = silhouette(A.patches, A.patchverts, show=False, ax=axes[0], index0=3)
  with GenerateVertexFromClick(snetwork, title=title, ax=axes[1]) as handler:
    e, i = handler.ei

  indices = snetwork.face_indices[face_index]
  pairing = e @ indices
  if pairing == -1:
    e = -e
    i = len(network.get_edges(e).points) - 1 - i

  firstedge = network.get_edges(e)
  edge0, edge1 = firstedge.split(i)
  i = edge0.length / firstedge.length

  network, indexmap = network.trefine(e, i, return_indexmap=True)
  # last edge before corner
  lebc = indexmap[e] if e > 0 else e

  network, indexmap = network.trefine(abs(lebc), .5, return_indexmap=True)
  lebc = indexmap[lebc] if lebc > 0 else lebc

  indices = network.face_indices[face_index]
  indices = indices.roll_to( (lebc @ indices) * lebc )

  nextindex = indices[1]
  nextedge = network.get_edges(nextindex)
  cumlength = nextedge.dx.cumsum()
  nextposition = nextedge.verts[np.argmin((cumlength - edge1.length)**2)]

  network, indexmap = network.trefine(nextindex, nextposition, return_indexmap=True)
  # first edge after corner
  feac = nextindex if nextindex > 0 else indexmap[nextindex]

  return network.trefine(feac, .5)


def gmsh_template(network, face_index, verbose=True):
  snetwork = network.take([face_index])
  face_indices = network.face_indices[face_index]
  snetwork_linearized = snetwork.linearize()
  vertices = np.stack([ snetwork_linearized.get_edges(edge).vertices[0]
                                             for edge in face_indices ], axis=0)

  from .aux import angle_between_vectors

  def para_angles(x):
    x = x.reshape([-1, 2])
    x = np.concatenate([ x[-1:], x, x[:1]])
    # v0 = x[1: -1] - x[:-2]
    v0 = x[2:] - x[1: -1]
    v1 = x[:-2] - x[1: -1]
    return angle_between_vectors(v0, v1, positive=True)

  def lengths(x):
    x = x.reshape([-1, 2])
    x = np.concatenate([x, x[:1]])
    return ((x[1:] - x[:-1])**2).sum(1) ** .5

  vertices = vertices - np.average(vertices, axis=0)[None]
  area = snetwork.polygons[face_index].area
  vertices = vertices / np.sqrt(area)

  verts = vertices.ravel()
  lengths0 = lengths(verts)

  costfunc = lambda x: ( (lengths(x) - lengths0)**2 ).sum()

  from scipy import spatial, optimize

  convex_verts = np.sort(spatial.ConvexHull(verts.reshape([-1, 2])).vertices)
  roll = convex_verts[0]

  pieces = [ np.concatenate([vertices]*2)[a: b+1] for a, b in zip(convex_verts, np.concatenate([convex_verts[1:], [len(vertices) + roll]])) ]
  pieces = [ piece[:-1] if len(piece) < 3 else piece[:1] + np.linspace(0, 1, len(piece))[:len(piece) - 1, None] * (piece[-1] - piece[0])[None] for piece in pieces ]

  x0 = np.roll(np.concatenate(pieces, axis=0), roll, axis=0).ravel()

  constraints = ({'type': 'ineq', 'fun': lambda x: np.pi - para_angles(x)},)

  x = optimize.minimize(costfunc, x0, constraints=constraints, method='SLSQP')

  vertices = x.x.reshape([-1, 2])

  if verbose:
    plt.plot( *np.concatenate([vertices, vertices[:1]]).T )

  if not x.success:
    raise AssertionError('Failed to find a suitable bounding box.')

  raise NotImplementedError


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
    from .multipatch import MultiPatchBSplineGridObject
    from .fit import multipatch_boundary_fit_from_univariate_gos
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


def map_pol_onto_other_pol(A: MultiPatchTemplate, B: MultiPatchTemplate, i0=None, p=1, nelems=5):
  """ map one polygon harmonically onto another one
      parameters
      ----------
      i0: forthcoming
      p: degree with which the harmonic map is approximated
      nelems: number of elements per patch per direction
  """
  obe0, = A.ordered_boundary_edges
  obe1, = B.ordered_boundary_edges

  obi1 = np.array([edge[0] for edge in obe1])

  assert len(obe0) == len(obe1)

  kv = ko.KnotObject(np.linspace(0, 1, nelems+1))
  g0 = A.to_MultiPatchBSplineGridObject({edge: kv for edge in A._knotvector_edges})

  otherverts = np.asarray(A.patchverts).copy()
  otherverts[obi1] = np.asarray(B.patchverts)[obi1]

  g1 = A.__class__(A.patches, otherverts, A._knotvector_edges).to_MultiPatchBSplineGridObject({edge: kv for edge in A._knotvector_edges})

  g0.cons = g0.project(g1.geom, domain=g0.domain.boundary)

  from mapping import sol
  sol.forward_laplace(g0)

  sample = g0.domain.locate(g0.geom, np.asarray(A.patchverts), eps=1e-8)

  return A.__class__(A.patches, sample.eval(g0.mapping), A._knotvector_edges)


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
