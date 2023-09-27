#!/bin/bash

SEED=$1
ETA=$2

for ALG in 'PPO' 'SAC' ; do # 'TD3' 'A2C' 'DQN'
  for ENV in 'SP-random' 'SP-bell' 'SP-ghz2' 'UC-random' 'UC-hadamard' 'UC-toffoli';do # 'SP-ghz1' "Maze${SIZE}Target"
    for DELTA in 8; do # 5, 20
      O="results/out/$ENV-$ETA-$DELTA/$ALG"; mkdir -p "$O"
      echo "Running $ALG in $ENV ($ETA | $DETLA) [SEED $SEED]" &> "$O/$SEED.out"
      python -m train $ALG -e $ENV-q$ETA-d$DELTA -s $SEED &> "$O/$RUN.out"
    done
  done
done
