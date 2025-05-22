from .network import EdgeNetwork

from shapely import ops, geometry

from itertools import count, chain
from functools import lru_cache

import networkx as nx
import numpy as np

from matplotlib import pyplot as plt

import pygmsh
from mapping.sol import make_matrix, MKL_from_scipy
from mapping import pc
from nutils import util

from scipy import optimize


def freeze(arr):
  arr = np.asarray(arr)
  arr.flags.writeable = False
  return arr


def refine_verts(verts, n=1):
  if n == 0: return verts
  return refine_verts(np.insert(verts, np.arange(1, len(verts)), (verts[1:] + verts[:-1]) / 2), n=n-1)


@lru_cache(maxsize=32)
def floater(Xi, X, ref=3):
  Xi, X = map(np.asarray, (Xi, X))

  XiPc = pc.PointCloud(np.concatenate([Xi, Xi[:1]]))
  XPc = pc.PointCloud(np.concatenate([X, X[:1]]))
  Xi = XiPc.toInterpolatedUnivariateSpline(k=1)(refine_verts(np.concatenate([XiPc.verts, [1]]), n=ref))[:-1]
  X = XPc.toInterpolatedUnivariateSpline(k=1)(refine_verts(np.concatenate([XPc.verts, [1]]), n=ref))[:-1]

  max_dist = np.linalg.norm(X[1:] - X[:-1], axis=1).max()

  with pygmsh.geo.Geometry() as geom:
    geom.add_polygon(X, mesh_size=1.1*max_dist)
    tri = geom.generate_mesh(algorithm=5, dim=2, order=1)

  W = make_matrix(tri.cells_dict['triangle'], tri.points)

  N = W.shape[0] // 2
  cons = util.NanVec((2 * N,))
  npoints = len(X)
  cons[:npoints] = Xi[:, 0]
  cons[N: N + npoints] = Xi[:, 1]

  M = MKL_from_scipy(W)

  vertices_param = M.solve(constrain=cons).reshape([2, -1]).T
  vertices_geom = tri.points[:, :2].copy()

  return freeze(vertices_param), freeze(vertices_geom)


def compute_vertex_positions(Xi, X, verts, ref=3):

  vertices_param, vertices_geom = floater(tuple(map(tuple, Xi)), tuple(map(tuple, X)), ref=ref)

  Xs = []

  for vert in verts:
    dist = np.linalg.norm(vertices_param - vert[None], axis=1)
    n = np.argmin(dist)
    Xs.append(vertices_geom[n])

    vertices_param = np.concatenate([vertices_param[:n], vertices_param[n+1:]])
    vertices_geom = np.concatenate([vertices_geom[:n], vertices_geom[n+1:]])

  return np.stack(Xs)


def untangle(template, patchverts, mu=.1):

  patchverts = np.asarray(patchverts)
  assert patchverts.shape[1:] == (2,) and len(patchverts) == len(template.patchverts)

  ex = np.concatenate([[patch[::2] for patch in template.patches],
                       [patch[1::2] for patch in template.patches] ])
  ey = np.concatenate([[patch[:2] for patch in template.patches],
                       [patch[2:] for patch in template.patches] ])

  obe, = template.ordered_boundary_edges
  obi = set(chain(*obe))

  dofindices = np.asarray([i for i in range(len(patchverts)) if i not in obi], dtype=int)

  if len(dofindices) == 0:
    return patchverts

  cons = patchverts.copy().reshape([-1, 2])

  def func(c):
    c = c.reshape([-1, 2])
    cons[dofindices] = c

    us = cons[ex[:, 1]] - cons[ex[:, 0]]
    us /= np.linalg.norm(us, axis=1)[:, None]

    vs = cons[ey[:, 1]] - cons[ey[:, 0]]
    vs /= np.linalg.norm(vs, axis=1)[:, None]

    x = np.cross(us, vs)
    ret = np.piecewise(x, [x < mu, x >= mu], [lambda x: mu - x, lambda x: np.zeros(len(x))]).sum()
    return ret

  x = optimize.minimize(func, patchverts[dofindices].ravel())

  if not np.abs(func(x.x)) < 1e-10:
    return patchverts

  cons[dofindices] = x.x.reshape([-1, 2])

  return cons


def draw_skeletton(network: EdgeNetwork, ref=2, ax=None, show=True, fontsize=8, templates_only=False):
  if templates_only:
    subnetwork = network.take(list(network.templates.keys()))
  else:
    subnetwork = network

  nodes = count(1)
  map_coord_index = {}
  edges = set()
  coords = {}

  for edge in subnetwork.edges:
    myverts = list(map(tuple, np.round(edge.vertices, 7)))

    for vi in myverts:
      if vi not in map_coord_index:
        map_coord_index[vi] = nodes.__next__()
      coords[map_coord_index[vi]] = vi

    edges.update({tuple(sorted([map_coord_index[vi] for vi in myverts]))})

  fixed_indices = set()
  face_vertices = {}

  for face, facetemplate in subnetwork.templates.items():
    X = []
    Xi = []

    template = facetemplate.template

    patchverts = np.asarray(template.patchverts)
    mymap = facetemplate.boundary_map()

    obe, = template.ordered_boundary_edges
    obi = tuple(edge[0] for edge in obe)

    map_local_index_glob = {}

    for local_edge in obe:
      index_glob = mymap.get(local_edge)
      edge, = subnetwork.get_edges(index_glob)
      v0, v1 = map(tuple, np.round(edge.vertices, 7))
      X.append(v0)
      Xi.append(patchverts[local_edge[0]])

      # for vi in (v0, v1):
      #   if vi not in map_coord_index:
      #     map_coord_index[vi] = nodes.__next__()

      map_local_index_glob[local_edge[0]] = map_coord_index[v0]
      map_local_index_glob[local_edge[1]] = map_coord_index[v1]

    myvertices = compute_vertex_positions(Xi, X, patchverts, ref=ref)
    myvertices = untangle(template, myvertices)

    for local_index in (set(range(len(template.patchverts))) - set(obi)):
      n = nodes.__next__()
      map_local_index_glob[local_index] = n

    for i, j in map_local_index_glob.items():
      coords[j] = myvertices[i]

    for patch in template.patches:
      for local_edge in (patch[:2], patch[2:], patch[::2], patch[1::2]):
        glob_edge = tuple(sorted([map_local_index_glob[i] for i in local_edge]))
        edges.update({glob_edge})

    myvertices = []

    for i in obi:
      glob_index = map_local_index_glob[i]
      fixed_indices.update({glob_index})
      myvertices.append(glob_index)

    face_vertices[face] = tuple(myvertices)

  colors = subnetwork._colors

  indices = np.arange(1, len(coords) + 1)

  G = nx.Graph()
  G.add_nodes_from(indices)
  nx.set_node_attributes(G, coords, 'pos')

  G.add_edges_from(list(edges))

  if ax is None:
    fig, ax = plt.subplots()
  else:
    fig = ax.figure

  nx.draw(G, coords, ax=ax, node_size=0)
  ax.set_aspect('equal')

  Xblack = np.stack([coords[i] for i in range(1, len(coords) + 1) if i not in fixed_indices])
  Xred = np.stack([coords[i] for i in fixed_indices])

  ax.scatter(*Xblack.T, s=12, c='k')
  ax.scatter(*Xred.T, s=16, c='r')

  for face_index, indices in face_vertices.items():
    mypol = np.stack([coords[i] for i in indices])
    ax.fill(*mypol.T, color=colors[face_index], alpha=.1)

    point = np.concatenate(ops.polylabel(geometry.Polygon(mypol)).coords.xy)
    ax.text(*point, str(list(face_index)), fontsize=fontsize, color=colors[face_index], zorder=10)

  if show:
    plt.show()

  return fig, ax