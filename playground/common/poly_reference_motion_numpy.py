import numpy as np
import pickle


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str):
        data = pickle.load(open(polynomial_coefficients, "rb"))
        self.dx_range = [0, 0]
        self.dy_range = [0, 0]
        self.dtheta_range = [0, 0]
        self.dxs = []
        self.dys = []
        self.dthetas = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period = None

        self.process(data)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = data[name]["period"]
                self.fps = data[name]["fps"]
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = data[name][
                    "startend_double_support_ratio"
                ]
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)

            if dy not in self.dys:
                self.dys.append(dy)

            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}

            if dy not in _data[dx]:
                _data[dx][dy] = {}

            if dtheta not in _data[dx][dy]:
                _data[dx][dy][dtheta] = data[name]

            # 녹화 한 주기(nb_steps_in_period x 40)를 그대로 쓴다 (fit_poly.py).
            _data[dx][dy][dtheta] = data[name]["frames"]

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = _data[dx][dy][dtheta]

        self.data_array = np.array(self.data_array)

        print("[Poly ref data] Done processing")

    def _bracket(self, grid, v):
        """격자에서 v 를 감싸는 아래 칸 번호와 위 칸 쪽 가중치."""
        g = np.asarray(grid)
        v = np.clip(v, g[0], g[-1])
        k = np.clip(np.searchsorted(g, v, side="right") - 1, 0, len(grid) - 2)
        return k, (v - g[k]) / (g[k + 1] - g[k])

    def get_reference_motion(self, dx, dy, dtheta, i):
        """명령을 감싸는 격자 8 점을 세 축 선형 보간한다.

        가장 가까운 한 점을 고르면 명령이 격자 중간을 넘는 순간 기준 관절각이 최대
        0.25 rad 튀었다(research/04 1장).
        """
        (ix, wx), (iy, wy), (it, wt) = (
            self._bracket(self.dxs, dx),
            self._bracket(self.dys, dy),
            self._bracket(self.dthetas, dtheta),
        )
        k = int(i) % self.nb_steps_in_period
        out = 0.0
        for a, fa in ((0, 1 - wx), (1, wx)):
            for b, fb in ((0, 1 - wy), (1, wy)):
                for c, fc in ((0, 1 - wt), (1, wt)):
                    out = out + fa * fb * fc * self.data_array[ix + a, iy + b, it + c, k]
        return out


if __name__ == "__main__":

    PRM = PolyReferenceMotion(
        "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
    )
    vals = []
    select_dim = -1
    for i in range(PRM.nb_steps_in_period):
        vals.append(PRM.get_reference_motion(0.0, -0.05, -0.1, i)[select_dim])

    # plot
    import matplotlib.pyplot as plt
    import numpy as np

    ts = np.arange(0, PRM.nb_steps_in_period)
    plt.plot(ts, vals)
    plt.show()
