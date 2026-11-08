# AIRookie_Playground

> 소형 이족 동반 로봇 AI Rookie 의 걷기 정책을 학습하는 강화학습 환경이다.

[English](README.md) | 한국어

## 무엇을 해결하는가

AI Rookie 는 손으로 맞춘 걸음새가 아니라 신경망 정책으로 걷는다. 이 정책은 서보가 늦게 따라가고, 외장에 유격이 있고, 질량을 실측으로만 아는 실물에서 돌아야 한다. 이 저장소는 MuJoCo 에서 정책을 학습하고, 실물 측정값을 시뮬에 되먹여 다음 회차가 실물에 더 가깝게 시작하게 한다.

## 누구를 위한 것인가

- Open Duck Mini v2 계열로 소형 이족 로봇을 만들며, 외장을 바꾼 채 바로 돌릴 학습 환경이 필요한 사람
- 학생 로봇 프로젝트에서 시뮬과 실물의 차이(서보 지연, 유격, 무게중심 오차)를 어떻게 다뤘는지 궁금한 사람

## 기능

- MuJoCo MJX 또는 MJWarp(`--impl warp`, CUDA) 위의 PPO 학습과 도메인 무작위화
- 내장 시범 동작 생성기가 만든 기준 걸음새로 모방 보상
- AI Rookie 외장 로봇 모델. 다리 10축, 머리 4축, 서보 유격 포함
- 학습한 정책을 ONNX 로 내보내 `policies/` 에 둔다
- `real_fit/`: 실물 걷기 기록을 재현하는 무게중심 · 외장 유격 · 머리 유격 · 바닥 마찰을 찾는다
- 평가 도구: 회차마다 자세 · 토크 · 머리 부하 · 느린 걷기 검사

## 동작 방식

1. 시범 동작 생성기(placo)가 걸음 한 주기를 표로 녹화해 `playground/open_duck_mini_v2/data/polynomial_coefficients.pkl` 에 둔다.
2. `runner.py` 가 `flat_terrain_backlash` 장면에서 그 기준을 따라 조종 명령을 받는 걷기 정책을 학습한다.
3. 정책을 ONNX 로 내보낸다. 온보드 SW([AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime))가 같은 관측 배치로 50 Hz 에 돌린다.
4. 실물 걷기 기록을 `real_fit/data/records.npz` 로 묶고, 맞춘 물리값을 다음 회차에 넣는다.

## 요구 사항과 사용법

- Python 3.12, [uv](https://docs.astral.sh/uv/)
- 학습: 기본 MJWarp 백엔드는 CUDA 가 되는 NVIDIA GPU 가 필요하다. CUDA 가 없으면 `--impl jax` 를 쓴다.

```bash
uv sync
uv run playground/open_duck_mini_v2/runner.py --task flat_terrain_backlash --num_timesteps 300000000
uv run tensorboard --logdir=checkpoints
```

학습한 정책을 MuJoCo 뷰어에서 돌린다.

```bash
uv run playground/open_duck_mini_v2/mujoco_infer.py -o policies/v2/v2_48_300M.onnx
```

학습 서버에 붙는 도구(`tools/round_eval/run.sh`)는 이 저장소 한 단계 위 `.env` 에서 `SERVER_USER` · `SERVER_HOST` · `SERVER_PORT` · `KEY_PATH` 를 읽는다.

## 관련 저장소

| 저장소 | 역할 |
|---|---|
| [AIRookie_Runtime](https://github.com/h-taek/AIRookie-Runtime) | 젯슨에서 도는 온보드 SW |
| [AIRookie_Sim](https://github.com/h-taek/AIRookie-Sim) | 온보드 SW 를 안에서 돌리는 시뮬레이터 앱 |

## 출처와 라이선스

이 저장소는 Antoine Pirrone, Steve Nguyen 외 기여자의 [Open Duck Playground](https://github.com/apirrone/Open_Duck_Playground), [Open Duck reference motion generator](https://github.com/apirrone/Open_Duck_reference_motion_generator), [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) 에서 출발했다. 자세한 내용은 [NOTICE](NOTICE) 에 있다. Open Duck Mini 에서 온 부분은 Apache License 2.0([LICENSE-APACHE](LICENSE-APACHE))을 따른다.

AI Rookie 가 더한 부분에는 아직 오픈소스 라이선스를 정하지 않았다. 모든 권리를 보유한다.
