import numpy as np
from mapping import sol, aux
from nutils import function
from genus import MultiPatchBSplineGridObject
from sol import Blechschmidt
import template
import itertools


def apply_transfinite_to_all_patches(mg):

  gs = mg.break_apart()
  basis_disc = mg.make_basis(patchcontinuous=False)

  for g in gs:
    g.set_cons_from_x()
    sol.transfinite_interpolation(g)

  x = np.concatenate([g.x for g in gs])
  func = basis_disc.vector(2).dot(x)

  mg.x = mg.project(func)


def transfinite_from_analytic_functions(funcs, corners, geom, mu=0):
  """ funcs = [left, right, bottom, top]
      corners = [P00, P01, P10, P11]
  """

  x, y = geom
  left, right, bottom, top = funcs
  P00, P01, P10, P11 = corners

  # reference bilinear parameterisations of all sides
  left_ref = (1 - y) * P00 + y * P01
  right_ref = (1 - y) * P10 + y * P11
  bottom_ref = (1 - x) * P00 + x * P10
  top_ref = (1 - x) * P01 + x * P11

  # the blending satisfies: bl(0) = 1, bl(1) = 1 and bl(x) < 1 on (0, 1) for mu > 0
  blending1 = (1 / ( 1 + np.exp(-mu))) * ( function.exp(-mu*y) + function.exp(-mu * (1 - y)) )
  blending0 = (1 / ( 1 + np.exp(-mu))) * ( function.exp(-mu*x) + function.exp(-mu * (1 - x)) )

  # the blending between the original bilinear map and the adjusted map
  blending = blending0 * blending1

  expression_real = left * (1 - x) + right * x + bottom * (1 - y) + top * y
  expression_ref = left_ref * (1 - x) + right_ref * x + bottom_ref * (1 - y) + top_ref * y

  # blend the reference map and the concave-corner map
  expression = blending * expression_real + (1 - blending) * expression_ref
  expression += -(1 - x) * (1 - y) * P00 \
                - x * y * P11 \
                - x * (1 - y) * P10 \
                - (1 - x) * y * P01

  return expression


def make_controlmap_for_concave_corners(mg, concave_indices, exp=2/3, mu=0.5, inner_mu=2):
  """ mg: genus.MultiPatchBSplineGridObject
      concave_indices: list of indices that need to be ``concaved``
      mu: decay rate at which the reparameterised edge goes back to the original edge from the concave index
          mu small => edge stays largely reparameterised along the whole axis
          mu big => edge goes back to linear quickly
      inner_mu: for the transfinite interpolation
                inner_mu big => the interpolated patch map goes back to the bilinear map quickly in the interior
                inner_mu small => the interpolated patch map stays largely nonbilinear in the interior
  """

  assert 0 < exp < 1
  assert mu >= 0 and inner_mu >= 0

  concave_indices = set(concave_indices)

  basis_patch = mg.domain.basis_patch()

  # get boundary edges (and reverses)
  boundary_edges = set(mg.boundary_edges().keys())
  boundary_edges.update(set([edge[::-1] for edge in boundary_edges]))

  interior_edges = set(mg.all_edges) - boundary_edges
  assert set(concave_indices).issubset(set(itertools.chain(*boundary_edges)))

  # get the UNIQUE edge in the interior that contains the concave index
  map_edge_cindex = {}
  for index in concave_indices:
    myedges = [edge for edge in interior_edges if index in edge]
    assert len(myedges) == 1, NotImplementedError
    map_edge_cindex[myedges[0]] = index

  x, y = mg.localgeom
  patchverts = np.array(mg.patchverts)

  # build the new controlmap
  controlmap = 0
  for one, edges in zip(basis_patch, mg._reference_ids):
    mycorners = [ patchverts[i] for i in itertools.chain(*edges[:2]) ]
    myfuncs = []
    # build left, right, bottom, top funcs
    for s, edge in zip([y, y, x, x], edges):

      # see if edge maps to a concave index
      cindex = map_edge_cindex.get(edge, None)

      # no ? the parameterisation stays linear along the edge
      if cindex is None:
        blending = s

      # yes? build a concave corner function
      else:
        local_index = edge.index(cindex)

        # blend s**(exp) from bottom to top
        if local_index == 0:
          blending = s**exp * function.exp(-mu*s) + (1 - function.exp(-mu*s)) * s

        # blend from top to bottom
        else:
          blending = (1 - (1 - s)**exp) * function.exp(-mu*(1 - s)) + (1 - function.exp(-mu*(1-s))) * s

      P0, P1 = patchverts[ list(edge) ]
      myfuncs.append( (1 - blending) * P0 + blending * P1 )

    controlmap += transfinite_from_analytic_functions(myfuncs, mycorners, mg.localgeom, mu=inner_mu) * one

  return controlmap


def test_controlmap():
  A = template.even_n_leaf_template(8)
  cindices = np.unique(list(itertools.chain(*A.boundary_edges)))[::2]

  mg = MultiPatchBSplineGridObject.from_template(A, knotvectors=10)

  controlmap = make_controlmap_for_concave_corners(mg, cindices, exp=(1/2), inner_mu=10)

  aux.plot(mg.domain, controlmap)


def test_L():
  A = template.singlepatch_template().refine((0, 2), .5)
  mg = MultiPatchBSplineGridObject.from_template(A, knotvectors=15)

  x, y = mg.geom

  top = function.stack([x, function.piecewise(x, [.5], 1 - 1.5 * x, 1.5 * x - .5)])
  cons = mg.project_edge((1, 5), top)
  cons |= mg.project_edge((5, 3), top)

  cons |= mg.project(mg.geom, domain=mg.domain.boundary)
  mg.cons = cons
  mg.x = cons | 0

  controlmap = make_controlmap_for_concave_corners(mg, [5])

  cm_cons = mg.project(controlmap, domain=mg.domain.boundary)
  mg.controlmap = mg.basis.vector(2).dot(mg.project(controlmap, constrain=cm_cons))
  sol.forward_laplace(mg)
  mg.qplot()

  # mg.g_controlmap().qplot()

  from sol import LakkisPryer

  solv = LakkisPryer(mg, gamma='trace')

  mg.x = solv.newton()

  import ipdb
  ipdb.set_trace()


if __name__ == '__main__':
  test_L()
