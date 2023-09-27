#!/bin/bash

for SEED in 1 2; do
    for ETA in 4 8; do #2
        echo "$SEED-$ETA"
        tmux new-session -t "run-$SEED-$ETA" -d
        tmux send-keys -t "run-$SEED-$ETA" "jobs/job.sh $SEED $ETA" C-m
        # tmux kill-session -t "run-$SEED-$ETA"
    done
done