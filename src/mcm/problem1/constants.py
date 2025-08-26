from dataclasses import dataclass
import math

@dataclass(frozen=True)
class SpiralParams:
    pitch: float = 0.55  # m, 螺距 p
    b: float = pitch / (2 * math.pi)  # r = b * theta

@dataclass(frozen=True)
class ChainParams:
    n_total: int = 223  # 板凳总数
    len_head: float = 3.41
    len_body: float = 2.20
    hole_offset: float = 0.275
    width: float = 0.30
    v_head: float = 1.0  # m/s
    first_interval: float = len_head - 2 * hole_offset  # between head front & first body front handle
    body_interval: float = len_body - 2 * hole_offset

    @property
    def handle_count(self) -> int:
        # front handles per bench + tail rear handle => n_total + 1
        return self.n_total + 1

    def distance_between_handles(self, idx_from: int) -> float:
        """距离: front handle i 到 front handle i+1 (i 从0开始, 0为龙头前把手)
        最后一个 front handle (尾节) 到尾节后把手的距离为 len_body - hole_offset (后孔距前端) = len_body - hole_offset.
        """
        # idx_from in [0, n_total-1] mapping bench index
        if idx_from == 0:
            return self.first_interval
        if idx_from < self.n_total - 1:
            return self.body_interval
        # last segment to tail rear handle
        return self.len_body - self.hole_offset  # 2.2 - 0.275 = 1.925
