from dataclasses import dataclass

@dataclass
class Metric:
    p: int
    q: int
    name: str = "custom"

    @property
    def n(self):
        return self.p + self.q

class EuclideanMetric(Metric):
    def __init__(self, n: int):
        super().__init__(p=n, q=0, name=f"E{n}")

class SpacetimeMetric(Metric):
    def __init__(self, spatial_dims: int = 3, time_dims: int = 1):
        super().__init__(p=spatial_dims, q=time_dims, name=f"R{spatial_dims},{time_dims}")

class ConformalMetric(Metric):
    def __init__(self, euclidean_dims: int):
        # CGA adds 1 positive (infinity) and 1 negative (origin) dimension
        super().__init__(p=euclidean_dims + 1, q=1, name=f"CGA{euclidean_dims}")
