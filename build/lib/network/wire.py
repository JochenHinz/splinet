import aux
from network import EdgeNetwork
import numpy as np


def rot_matrix(theta):

  return np.array([ [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta) ] ])


def circle_pieces(breaks, npoints=201):
  breaks = tuple(breaks)
  assert all( th1 > th0 for th0, th1 in zip(breaks, breaks[1:]) )

  ret = []

  for th0, th1 in zip(breaks, breaks[1:]):
    ret.append( np.stack([ np.cos(np.linspace(th0, th1, npoints)),
                           np.sin(np.linspace(th0, th1, npoints)) ], axis=1) )

  return ret


def compute_outer_circle_offsets(y0, r):

  y = np.array([0, y0])

  wire_circle = [ r * elem for elem in circle_pieces(np.linspace(0, 1, 5) * 2 * np.pi - np.pi / 4) ]
  ms = [wire_circle[i][-1] for i in range(2)]

  thetas = []

  for m in ms:
    p = 2 * (m * y).sum() / ((m**2).sum())
    q = ( (y**2).sum() - 1 ) / ((m**2).sum())
    s = - p / 2 + np.sqrt(p**2 / 4 - q)
    vec0 = y + s * m
    thetas.append(aux.angle_between_vectors(np.array([1, 0]), vec0))

  for theta in np.linspace(0, 2*np.pi, 4)[1:-1]:
    thetas.append(thetas[0] + theta)
    thetas.append(thetas[1] + theta)

  return tuple(thetas + [thetas[0] + 2*np.pi])


def straight_line(v0, v1, npoints=201):
  xi = np.linspace(0, 1, npoints)
  return v0[None] * (1 - xi)[:, None] + v1[None] * xi[:, None]


def wire(y0=0.5, r=0.4, nelems=8):

  wire_circle = [ r * elem for elem in circle_pieces(np.linspace(0, 1, 5) * 2 * np.pi - np.pi / 4) ]

  outer_circle = circle_pieces( compute_outer_circle_offsets(y0, r) )
  y = np.array([0, y0])

  edges = outer_circle

  for theta in np.linspace(0, 2*np.pi, 4)[:-1]:
    mat = rot_matrix(theta)
    for circ in wire_circle:
      edges.append((mat @ (y[:, None] + circ.T)).T)

  for i, j in zip(range(8, 20, 4), range(1, 7, 2)):
    edges.append(straight_line(edges[i-1][0], edges[j-1][0]))
    edges.append(straight_line(edges[i-1][-1], edges[j-1][-1]))

  face_indices = {}

  face_indices[1] = [19, 1, -20, -8]
  face_indices[2] = [21, 3, -22, -12]
  face_indices[3] = [23, 5, -24, -16]

  face_indices[4] = list(range(7, 11))
  face_indices[5] = list(range(11, 15))
  face_indices[6] = list(range(15, 19))

  face_indices[7] = [6, -19, -7, -10, -9, 20, 2, -21, -11, -14,
                                -13, 22, 4, -23, -15, -18, -17, 24]

  network = EdgeNetwork(list(range(1, len(edges)+1)), edges, face_indices)

  network.is_connected
  snetwork = network.take([1, 2, 3, 4, 5, 7])

  obi = snetwork.ordered_boundary_indices

  print(obi)

  snetwork = network.take([1, 2, 3, 4, 5, 6])

  snetwork.edge_neighbours

  network.boundary_network().qplot()

  snetwork = network.take([1, 2, 3, 4, 5, 7])

  print(snetwork.select_faces())


if __name__ == '__main__':
  wire()
