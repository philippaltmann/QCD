for SEED in 1 2 3 4; do  # Depth = 3xoptimal solution
  for CHALLENGE in 'UC-hadamard-q1-d9', 'SP-bell-q2-d12', 'SP-ghz3-q3-d15', 'UC-toffoli-q3-d63'
    for ALG in 'SAC' 'PPO' 'TD3' 'DQN'; do # 'PPO' 'SAC' 'TD3'  #'A2C'
          O="results/out/$CHALLENGE/$ALG"; mkdir -p "$O"
          echo "Running $ALG in $CHALLENGE [SEED $SEED]" &> "$O/$SEED.out"
          python -m train $ALG -e $CHALLENGE -s $SEED &> "$O/$SEED.out" &
          sleep 5 
        done
      done
    done
  done
done