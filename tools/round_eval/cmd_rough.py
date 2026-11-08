"""무릎 명령 거칠기(research/10 26-7 지표): 2 차 변화 RMS(°), 방향 뒤집힘 비율, 3 Hz 이상 파워 비중."""
import numpy as np
def rough(x, dt=0.02):
    x = np.asarray(x, float)
    d1 = np.diff(x); d2 = np.diff(x, 2)
    flip = np.mean(np.sign(d1[1:]) * np.sign(d1[:-1]) < 0)
    y = x - x.mean(); P = np.abs(np.fft.rfft(y)) ** 2; f = np.fft.rfftfreq(len(y), dt)
    hf = P[f >= 3.0].sum() / P[f > 0].sum()
    return np.sqrt(np.mean(d2 ** 2)), flip, hf
