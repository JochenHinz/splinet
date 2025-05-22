from genus import MultiPatchBSplineGridObject
from nutils import function


def compute_Area(mg: MultiPatchBSplineGridObject):
  J = mg.mapping.grad(mg.localgeom)
  return mg.integrate(function.determinant(J)**2 * function.J(mg.localgeom))
