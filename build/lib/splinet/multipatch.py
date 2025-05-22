from .template import MultiPatchTemplate, \
                      OrderedBoundaryEdgesMixin, get_edges, multipatch_boundary_edges, \
                      patches_verts_to_network, silhouette
from .aux import default_ids

from mapping import mul
import itertools
from functools import cached_property
import numpy as np


index_to_side = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}


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

  def MultiPatchTemplate_from_mapping(self):
    patchverts = np.asarray(self.patchverts)
    sample = self.domain.locate(self.geom, patchverts, eps=1e-8)
    return MultiPatchTemplate(self.patches, sample.eval(self.mapping), tuple(sorted(self.knotvectors.keys())))

  @cached_property
  def all_edges(self):
    return default_ids(self.patches)

  def get_edge_domain(self, edge):
    edge = tuple(edge)
    all_edges = set(itertools.chain(*self.all_edges))
    edge = edge if edge in all_edges else edge[::-1]
    assert edge in all_edges
    (ipatch, index), *ignore = [(i, ids.index(edge)) for i, ids in enumerate(self.all_edges) if edge in ids]
    side = {0: 'left', 1: 'right', 2: 'bottom', 3: 'top'}[index]
    return self.domain.patches[ipatch].topo.boundary[side]

  def project_edge(self, edge, func, *args, **kwargs):
    topo = self.get_edge_domain(edge)
    return self.project(func, *args, domain=topo, **kwargs)

  to_template = to_MultiPatchTemplate


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
