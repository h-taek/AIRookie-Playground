"""ONNX 걷기 정책을 JAX 함수로 옮긴다.

정책은 (관측 − 평균) × 1/표준편차 → 은닉 4 층(swish) → 28 → 앞 14 의 tanh 다(`export_onnx.py`).
환경 수천 개를 한 번에 추론하려고 onnxruntime 대신 같은 식을 JAX 로 쓴다.
"""
from pathlib import Path

import jax.numpy as jp
import numpy as np
import onnx
from onnx import numpy_helper


def load(path: str | Path):
    """ONNX 파일에서 가중치를 읽어 `act(obs) -> action` 을 돌려준다. obs 는 (..., 101)."""
    m = onnx.load(str(path))
    w = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    ops = [n.op_type for n in m.graph.node]
    assert ops == ["Sub", "Mul", "Gemm", "Sigmoid", "Mul", "Gemm", "Sigmoid", "Mul", "Gemm", "Sigmoid", "Mul",
                   "Gemm", "Split", "Tanh"], ops
    mean = next(v for k, v in w.items() if k.endswith("sub/ReadVariableOp:0"))
    inv_std = next(v for k, v in w.items() if "truediv_recip" in k)
    layers = []
    for i in range(4):
        kern = next(v for k, v in w.items() if f"hidden_{i}_1/Cast" in k)
        bias = next(v for k, v in w.items() if f"hidden_{i}_1/BiasAdd" in k)
        layers.append((jp.asarray(kern), jp.asarray(bias)))
    # Gemm 의 transB 를 확인한다. 가중치가 (입력, 출력) 이 아니면 식이 틀린다.
    for n in m.graph.node:
        if n.op_type == "Gemm":
            tb = next((a.i for a in n.attribute if a.name == "transB"), 0)
            assert tb == 0, "transB=1 인 Gemm 은 다루지 않는다"
    nu = layers[-1][0].shape[1] // 2
    mean, inv_std = jp.asarray(mean), jp.asarray(inv_std)

    def act(obs):
        x = (obs - mean) * inv_std
        for j, (k, b) in enumerate(layers):
            x = x @ k + b
            if j < len(layers) - 1:
                x = x * (1.0 / (1.0 + jp.exp(-x)))
        return jp.tanh(x[..., :nu])

    return act


def load_many(paths: list) -> callable:
    """정책 여럿. `act(obs, pid)` 는 환경마다 pid 번째 정책의 행동을 낸다. 모두 계산하고 고른다."""
    fns = [load(p) for p in paths]

    def act(obs, pid):
        out = jp.stack([f(obs) for f in fns], axis=0)  # (정책, N, 14)
        return jp.take_along_axis(out, pid[None, :, None], axis=0)[0]

    return act


def check(path: str | Path, n: int = 64) -> float:
    """onnxruntime 출력과의 최대 차."""
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs = np.random.default_rng(0).normal(size=(n, 101)).astype(np.float32)
    ref = np.concatenate([s.run(None, {"obs": o[None]})[0] for o in obs])
    out = np.asarray(load(path)(jp.asarray(obs)))
    return float(np.abs(out - ref).max())


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(Path(p).name, f"최대 차 {check(p):.2e}")
