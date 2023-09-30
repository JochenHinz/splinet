from .template import MultiPatchTemplate, \
                      OrderedBoundaryEdgesMixin, get_edges, multipatch_boundary_edges, \
                      patches_verts_to_network, silhouette

from mapping import mul
from nutils import mesh, log, numeric, util, topology
import itertools
from functools import cached_property
import numpy as np


index_to_side = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}


def _get_edges(patch):
  patch = tuple(patch)
  return ( patch[:2], patch[2:], patch[::2], patch[1::2] )


def default_ids(patches):
  return tuple( _get_edges(patch) for patch in map(tuple, patches) )


@log.withcontext
def multipatch(patches, knotvectors, patchverts=None, name='multipatch'):
  from nutils.types import frozenarray
  patches = np.array(patches)
  if patches.dtype != int:
    raise ValueError('`patches` should be an array of ints.')
  if patches.ndim < 2 or patches.ndim == 2 and patches.shape[-1] % 2 != 0:
    raise ValueError('`patches` should be an array with shape (npatches,2,...,2) or (npatches,2*ndims).')
  elif patches.ndim > 2 and patches.shape[1:] != (2,) * (patches.ndim - 1):
    raise ValueError('`patches` should be an array with shape (npatches,2,...,2) or (npatches,2*ndims).')
  patches = patches.reshape(patches.shape[0], -1)

  # determine topological dimension of patches

  ndims = 0
  while 2**ndims < patches.shape[1]:
    ndims += 1
  if 2**ndims > patches.shape[1]:
    raise ValueError('Only hyperrectangular patches are supported: ' \
      'number of patch vertices should be a power of two.')
  patches = patches.reshape([patches.shape[0]] + [2]*ndims)

  # group all common patch edges (and/or boundaries?)

  assert isinstance(knotvectors, dict)
  # knotvectors = {frozenset(key): item for key, item in knotvectors.items()}

  # create patch topologies, geometries

  if patchverts is not None:
    patchverts = np.array(patchverts)
    indices = set(patches.flat)
    if tuple(sorted(indices)) != tuple(range(len(indices))):
      raise ValueError('Patch vertices in `patches` should be numbered consecutively, starting at 0.')
    if len(patchverts) != len(indices):
      raise ValueError('Number of `patchverts` does not equal number of vertices specified in `patches`.')
    if len(patchverts.shape) != 2:
      raise ValueError('Every patch vertex should be an array of dimension 1.')

  topos = []
  coords = []
  localcoords = []
  for i, patch in enumerate(patches):
    # find shape of patch and local patch coordinates
    shape = []
    for dim in range(ndims):
      nelems_sides = []
      sides = [(0,1)]*ndims
      sides[dim] = slice(None),
      for side in itertools.product(*sides):
        # sideverts = frozenset(patch[side])
        sideverts = tuple(patch[side])
        if sideverts[::-1] in knotvectors:
          assert sideverts not in knotvectors
          nelems_sides.append(frozenarray(knotvectors[sideverts[::-1]].flip().knots))
        elif sideverts in knotvectors:
          # nelems_sides.append(knotvectors[sideverts])
          nelems_sides.append(frozenarray(knotvectors[sideverts].knots))
        else:
          raise
      if len(set(nelems_sides)) != 1:
        raise ValueError('duplicate number of elements specified for patch {} in dimension {}'.format(i, dim))
      shape.append(nelems_sides[0])

    # myknotvector = np.prod(shape)
    # create patch topology
    # topos.append(mesh.rectilinear(myknotvector.knots, name='{}{}'.format(name, i))[0])
    topos.append(mesh.rectilinear(shape, name='{}{}'.format(name, i))[0])
    # compute patch geometry
    # patchcoords = myknotvector.knots
    patchcoords = shape
    localpatchcoords = numeric.meshgrid(*patchcoords).reshape(ndims, -1)
    if patchverts is not None:
      patchcoords = np.array([
        sum(
          patchverts[j]*util.product(c if s else 1-c for c, s in zip(coord, side))
          for j, side in zip(patch.flat, itertools.product(*[[0,1]]*ndims))
       )
        for coord in localpatchcoords.T
      ]).T
    coords.append(patchcoords)
    localcoords.append(localpatchcoords)

  # build patch boundary data

  boundarydata = topology.MultipatchTopology.build_boundarydata(patches)

  # join patch topologies, geometries

  topo = topology.MultipatchTopology(tuple(map(topology.Patch, topos, patches, boundarydata)))
  # funcsp = topo.basis('spline', degree=1, patchcontinuous=False)
  knotvectors1 = {key: knotvector.to_p(1).to_c(0) for key, knotvector in knotvectors.items()}
  funcsp = topo.basis_spline(degree=1,
                             patchcontinuous=False,
                             knotvalues={key: kv.knots for key, kv in knotvectors1.items()},
                             knotmultiplicities={key: kv.knotmultiplicities for key, kv in knotvectors1.items()})
  geom = (funcsp * np.concatenate(coords, axis=1)).sum(-1)
  localgeom = (funcsp * np.concatenate(localcoords, axis=1)).sum(-1)

  return topo, geom, localgeom


class MultiPatchBSplineGridObject(mul.MultiPatchBSplineGridObject, OrderedBoundaryEdgesMixin):

  def to_network(self):
    return patches_verts_to_network(self.patches, self.patchverts)

  @property
  def boundary_edges(self):
    return multipatch_boundary_edges(self.patches)

  def silhouette(self, **kwargs):
    return silhouette(self.patches, self.patchverts, **kwargs)

  def plot_patches(self):
    func = self.domain.basis_patch().dot(np.arange(len(self.patches)))
    self.qplot(func=func)

  def to_MultiPatchTemplate(self):
    return MultiPatchTemplate(self.patches, self.patchverts, tuple(sorted(self.knotvectors.keys())))

  @cached_property
  def all_edges(self):
    return default_ids(self.patches)

  def project_edge(self, edge, func, *args, **kwargs):
    edge = tuple(edge)
    all_edges = set(self.all_edges)
    edge = edge if edge in all_edges else edge[::-1]
    assert edge in all_edges
    (ipatch, index), *ignore = [(i, ids.index(edge)) for i, ids in enumerate(self._reference_ids) if edge in ids]
    side = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}[index]
    return self.project(func, *args, domain=self.domain.patches[ipatch].topo.boundary[side], **kwargs)


def singlepatch(knotvectors, *args, patches=None, patchverts=None, **kwargs):
  patches = patches or ((0, 1, 2, 3),)
  patchverts = patchverts or ((0, 0), (0, 1), (1, 0), (1, 1))
  assert set(patches[0]) == {0, 1, 2, 3}
  return MultiPatchBSplineGridObject(patches, patchverts, knotvectors, *args, **kwargs)


def stack_MultiPatchBSplineGridObjects(mg0, mg1, edge0, edge1, constrain_interface=True):
  template0, template1 = mg0.to_MultiPatchTemplate(), mg1.to_MultiPatchTemplate()
  template, vertex_map = MultiPatchTemplate.stack(template0, template1, edge0, edge1, return_vertex_map=True)
  edge_map = lambda x: tuple(vertex_map[i] for i in x)
  vec = []

  for i, (mypatch, g) in enumerate(zip(mg0.patches, mg0.break_apart())):
    patch = template.patches[i]
    if patch == mypatch:
      vec.append(g.x)
    # patch orientation got flipped during repair stage. This can only mean mypatch became mypatch[::-1]
    elif patch == mypatch[::-1]:
      vec.append( g.x.tensor[::-1][:, ::-1].ravel() )
    else:
      raise AssertionError

  knotvectors = {key: value for key, value in mg0.knotvectors.items() if key in template._knotvector_edges}

  for i, (mypatch_old, g) in enumerate(zip(mg1.patches, mg1.break_apart()), len(mg0.patches)):
    mypatch = edge_map(mypatch_old)
    patch = template.patches[i]
    if patch == mypatch:
      vec.append(g.x)
    # patch orientation got flipped during repair stage. This can only mean mypatch became mypatch[::-1]
    elif patch == mypatch[::-1]:
      vec.append( g.x.tensor[::-1][:, ::-1].ravel() )
    else:
      raise AssertionError

  knotvectors.update({edge_map(key): value for key, value in mg1.knotvectors.items() if edge_map(key) in template._knotvector_edges})

  mg = template.to_MultiPatchBSplineGridObject(knotvectors)
  basis_disc = mg.make_basis(patchcontinuous=False)
  mg.x = mg.project(basis_disc.vector(2).dot(np.concatenate(vec)))

  f = mg.g_geom()

  mg.set_cons_from_x()
  f.set_cons_from_x()

  if constrain_interface:
    found = False
    for i, patch in enumerate(mg.patches):
      for side, edge in zip(('left', 'right', 'bottom', 'top'), get_edges(patch)):
        if edge == edge0 or edge == edge0[::-1]:
          domain = mg.domain.patches[i].topo.boundary[side]
          found = True
          break
      if found: break
    else:
      raise AssertionError
    mg.cons |= mg.project( mg.mapping, domain=domain )
    f.cons |= f.project( f.mapping, domain=domain )

  return mg, f
