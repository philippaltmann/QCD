for ITERATION in 0 2 4 6; do
  for RUN in 1 2; do  # Depth = 3xoptimal solution
    for CHALLENGE in 'UC-hadamard-q1-d9' 'SP-bell-q2-d12' 'SP-ghz3-q3-d15' 'UC-toffoli-q3-d63'; do
      for ALG in 'PPO' 'A2C' 'SAC' 'TD3'; do
        SEED=$(($ITERATION + $RUN))
        O="results/out/$CHALLENGE/$ALG"; mkdir -p "$O"
        echo "Running $ALG in $CHALLENGE [SEED $SEED]"
        python -m train $ALG -e $CHALLENGE -s $SEED &> "$O/$SEED.out" &
        sleep 5 
      done
    done
  done
  sleep 4200
done