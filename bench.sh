for SEED in 1 2; do
  for ALG in 'PPO' 'SAC' ; do # 'TD3' 'A2C'
    for ENV in 'SP-random' 'SP-bell' 'SP-ghz2' 'UC-random' 'UC-hadamard' 'UC-toffoli';do # 'SP-ghz1' "Maze${SIZE}Target"
      for ETA in 4 8; do #2 
        for DELTA in 10; do # 5, 20
          O="results/out/$ENV-$ETA-$DELTA/$ALG"; mkdir -p "$O"
          nohup echo "Running $ALG in $ENV ($ETA | $DETLA) [SEED $SEED]" &> "$O/$SEED.out" &
          nohup python -m run $ALG -e $ENV-q$ETA-d$DELTA -s $SEED &> "$O/$RUN.out" &
          # sleep 1s
        done
        # sleep 20m
      done
      # sleep 20m
    done
    # sleep 30m
  done
done
