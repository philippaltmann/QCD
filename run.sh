for SEED in 1 2 3 4; do
  for ALG in 'DQN'; do
    for ENV in 'SP-random' 'SP-bell' 'SP-ghz2' 'SP-ghz1' 'UC-random' 'UC-hadamard' 'UC-toffoli';do
      for ETA in 4 8; do
        for DELTA in 8; do
          O="results/out/$ENV-$ETA-$DELTA/$ALG"; mkdir -p "$O"
          echo "Running $ALG in $ENV ($ETA | $DETLA) [SEED $SEED]" &> "$O/$SEED.out"
          python -m train $ALG -e $ENV-q$ETA-d$DELTA -s $SEED &> "$O/$SEED.out" &
          sleep 5 
        done
      done
    done
  done
done

sleep 4000

for SEED in 3 4; do
  for ALG in 'SAC' 'PPO' 'TD3'; do # 'PPO' 'SAC' 'TD3' 'DQN' #'A2C'
    for ENV in 'SP-random' 'SP-bell' 'SP-ghz2' 'SP-ghz1' 'UC-random' 'UC-hadamard' 'UC-toffoli';do
      for ETA in 4 8; do #2
        for DELTA in 8; do # 5, 20
          O="results/out/$ENV-$ETA-$DELTA/$ALG"; mkdir -p "$O"
          echo "Running $ALG in $ENV ($ETA | $DETLA) [SEED $SEED]" &> "$O/$SEED.out"
          python -m train $ALG -e $ENV-q$ETA-d$DELTA -s $SEED &> "$O/$SEED.out" &
          sleep 5 
        done
      done
    done
  done
done