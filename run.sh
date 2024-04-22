BASE='results'
for ITERATION in 0 2 4 6; do #0 2 4 6
  for RUN in 1 2; do  # Depth = 3xoptimal solution
    for TASK in 'UC-hadamard-q1-d9' 'SP-random-q2-d12'; do # Base 
    # for TASK in 'SP-ghz3-q3-d15'  'UC-toffoli-q3-d63'; do # Advanced
    # for TASK in 'SP-bell-q2-d12' 'UC-random-q2-d12'; do # Additional 
      for ALG in 'PPO' 'SAC' 'A2C' 'TD3'; do
        SEED=$(($ITERATION + $RUN))
        O="$BASE/out/$TASK/$ALG"; mkdir -p "$O"
        echo "Running $ALG in $TASK [SEED $SEED]"
        python -m train $ALG -e $TASK -s $SEED --sparse --punish --path $BASE &> "$O/$SEED.out" &
        sleep 5 
      done
    done
  done
done