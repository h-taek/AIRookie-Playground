#!/bin/bash
# 회차 평가. 서버 체크포인트 이름을 checkpoints_<회차>_300M 으로 바꾸고 마지막 정책을 가져와 평가 셋을 돌린다.
# 사용: tools/round_eval/run.sh v2_N   (Playground 루트에서). 결과는 eval_out/<회차>/ 에 남는다.
set -e
V=$1; T=tools/round_eval; O=eval_out/$V; mkdir -p $O
# 서버 접속 정보는 작업 폴더 루트의 .env 에서 읽는다 (SERVER_USER · SERVER_HOST · SERVER_PORT · KEY_PATH).
set -a; . ../.env; set +a
K="-i ${KEY_PATH/#\~/$HOME}"; H=$SERVER_USER@$SERVER_HOST
SSH="ssh $K -p $SERVER_PORT $H"; R=/data/work/playground/AIRookie_Playground
$SSH "cd $R && [ -d checkpoints ] && mv checkpoints checkpoints_${V}_300M || true"
LAST=$($SSH "ls $R/checkpoints_${V}_300M/*.onnx | sort -t_ -k6 -n | tail -1")
# 정책은 policies/ 한 곳에 둔다 (policies/README.md).
scp -q $K -P $SERVER_PORT $H:$LAST policies/${V%%_*}/${V}_300M.onnx
$SSH "grep STEP /data/work/playground/train_$V.log | tail -1" > $O/train_last.txt
P=policies/${V%%_*}/${V}_300M.onnx
.venv/bin/python tools/eval_policy_behavior.py -o $P --head-from-command > $O/behavior.txt 2>&1 &
.venv/bin/python $T/torque.py $P > $O/torque.txt 2>&1 &
.venv/bin/python $T/head_stress.py $P 1 2>&1 | grep -v "backlash\|Poly\|actuator names" > $O/stress.txt &
.venv/bin/python $T/head_stress_280.py $P 1 2>&1 | grep -v "backlash\|Poly\|actuator names" > $O/stress_280.txt &
.venv/bin/python $T/slow_walk.py $P 1 2>&1 | grep vx > $O/slow.txt &
wait
echo "== $V $(cat $O/train_last.txt)"
grep -E "^=====|뒤집힘 True" $O/behavior.txt | grep -B1 "뒤집힘 True" || echo "평가 명령 뒤집힘 없음"
grep -E "^=====|몸통 기준 속도|yaw 각속도" $O/behavior.txt | paste - - - | sed 's/  */ /g' | cut -c1-200 | head -9
sed -n '/명령  .*Σ|tau|/,/^옆/p;/월드 기준/,$p' $O/torque.txt
python3 $T/stress_summary.py $O/stress.txt | head -1
awk -F'\t' '{split($3,a,"/"); f+=a[1]; n+=a[2]} END{print "머리 시험 280 넘어짐 " f " / " n}' $O/stress_280.txt
cat $O/slow.txt
