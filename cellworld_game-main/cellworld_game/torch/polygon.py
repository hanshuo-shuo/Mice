import math

import torch
import typing
from ..interfaces import IPolygon
from ..util import Point
from .device import default_device


class Polygon(IPolygon):
    def __init__(self, vertices):
        if isinstance(vertices, torch.Tensor):
            self.vertices = vertices.to(default_device)
        else:
            self.vertices = torch.tensor(vertices, device=default_device)
        self._sides = len(self.vertices)
        self._vertices_x: torch.tensor = None
        self._vertices_y: torch.tensor = None
        self._edges: torch.tensor = None
        self._bounds = None

    def sides(self):
        return self._sides

    def bounds(self) -> typing.Tuple[float, float, float, float]:
        if self._bounds is None:
            self._bounds = tuple(self.vertices.min(dim=0)[0].tolist()) + tuple(self.vertices.max(dim=0)[0].tolist())
        return self._bounds

    @property
    def vertices_x(self) -> torch.tensor:
        if self._vertices_x is None:
            self._vertices_x = self.vertices[:, 0]
        return self._vertices_x

    @property
    def vertices_y(self) -> torch.tensor:
        if self._vertices_y is None:
            self._vertices_y = self.vertices[:, 1]
        return self._vertices_y

    @property
    def edges(self) -> torch.tensor:
        if self._edges is None:
            self._edges = self.vertices[(torch.arange(self._sides) + 1) % self._sides] - self.vertices
        return self._edges

    def contains(self, points):

        if isinstance(points, Polygon):
            contain_vertices = self.contains(points.vertices)
            return torch.all(contain_vertices)

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, device=default_device)

        n_points = points.shape[0]
        n_vertices = self.vertices.shape[0]

        # Extract x and y coordinates of the points
        points_x = points[:, 0]
        points_y = points[:, 1]

        # Repeat polygon coordinates for each point
        poly_x_repeated = self.vertices_x.repeat(n_points, 1)
        poly_y_repeated = self.vertices_y.repeat(n_points, 1)

        # Calculate if edges cross the ray extending to the right of each point
        j = torch.arange(n_vertices) - 1
        vertex_1_y = poly_y_repeated[:, j]
        vertex_2_y = poly_y_repeated
        condition_1 = (vertex_1_y > points_y.unsqueeze(1)) != (vertex_2_y > points_y.unsqueeze(1))
        slope = (poly_x_repeated[:, j] - poly_x_repeated) / (poly_y_repeated[:, j] - poly_y_repeated)
        intercept_x = poly_x_repeated + slope * (points_y.unsqueeze(1) - poly_y_repeated)
        condition_2 = points_x.unsqueeze(1) < intercept_x
        # Determine if the number of crossings is odd or even
        crossings = (condition_1 & condition_2).sum(dim=1)
        inside = crossings % 2 == 1
        return inside

    def intersects(self, other: "Polygon") -> bool:
        """Proper polygon-polygon intersection test.

        The previous implementation only checked if any vertex of `other`
        lay inside `self`, which misses three very common cases for small
        agent bodies vs world occlusions:
          (1) a vertex of `self` is inside `other` (e.g. the agent body
              sitting fully inside a large occlusion)
          (2) `self` is fully contained in `other` with none of its
              vertices coinciding with `other`'s vertices
          (3) edges cross without any vertex being inside either polygon
              (common for thin, elongated bodies clipping a wall corner)

        We now test:
          - any `other`-vertex inside `self`
          - any `self`-vertex inside `other`
          - any edge of `self` crossing any edge of `other`
        which correctly detects intersection for arbitrary convex or
        concave simple polygons.
        """
        # --- case 1: other's vertices inside self ---
        if bool(self.contains(points=other.vertices).any()):
            return True
        # --- case 2: self's vertices inside other ---
        if bool(other.contains(points=self.vertices).any()):
            return True
        # --- case 3: vectorized edge-edge crossing test ---
        # For each edge pair (a1->a2 in self, b1->b2 in other) we use the
        # standard CCW sign test: segments strictly cross iff the two
        # endpoints of one segment lie on opposite sides of the other and
        # vice versa. Collinear-touching is treated as non-intersecting,
        # which matches shapely's default behavior for the "just kissing"
        # case and is what the old rotate-retry loop relied on.
        a = self.vertices  # (n, 2)
        b = other.vertices  # (m, 2)
        n = a.shape[0]
        m = b.shape[0]
        if n < 2 or m < 2:
            return False
        a1 = a
        a2 = a[(torch.arange(n, device=a.device) + 1) % n]
        b1 = b
        b2 = b[(torch.arange(m, device=b.device) + 1) % m]
        # broadcast to shape (n, m, 2)
        A1 = a1.unsqueeze(1).expand(n, m, 2)
        A2 = a2.unsqueeze(1).expand(n, m, 2)
        B1 = b1.unsqueeze(0).expand(n, m, 2)
        B2 = b2.unsqueeze(0).expand(n, m, 2)

        def _ccw(p, q, r):
            return (r[..., 1] - p[..., 1]) * (q[..., 0] - p[..., 0]) - \
                   (q[..., 1] - p[..., 1]) * (r[..., 0] - p[..., 0])

        d1 = _ccw(B1, B2, A1)
        d2 = _ccw(B1, B2, A2)
        d3 = _ccw(A1, A2, B1)
        d4 = _ccw(A1, A2, B2)
        crosses = ((d1 * d2) < 0) & ((d3 * d4) < 0)
        return bool(crosses.any())

    def __getitem__(self, item) -> typing.Tuple[float, float]:
        return tuple(self.vertices[item, :].tolist())

    def translate_rotate(self,
                         translation: Point.type,
                         rotation: float,
                         rotation_center: Point.type = (0, 0)) -> "Polygon":

        t = torch.tensor(translation, device=default_device)
        rc = torch.tensor(rotation_center, device=default_device)
        r = torch.tensor(math.radians(-rotation), device=default_device)
        vertices = self.vertices.clone() - rc
        rotation_matrix = torch.tensor([
            [torch.cos(r), -torch.sin(r)],
            [torch.sin(r), torch.cos(r)]
        ], dtype=torch.float32, device=default_device)
        rotated_points = torch.matmul(vertices, rotation_matrix)
        rotated_points += rc + t
        return Polygon(vertices=rotated_points)

    def area(self) -> float:
        # Extract x and y coordinates
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        # Apply the Shoelace formula
        area = 0.5 * torch.abs(torch.sum(x[:-1] * y[1:]) + x[-1] * y[0] - torch.sum(y[:-1] * x[1:]) - y[-1] * x[0])
        return float(area)
    