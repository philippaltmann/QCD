#!/bin/bash

for SEED in 1 2; do
    for ETA in 4 8; do #2
    tmux new-session -t "$SEED-$ETA" -d
    tmux send-keys -t "$SEED-$ETA" "jobs/exp.sh $SEED $ETA" C-m
    # tmux kill-session -t "$SEED-$ETA"
    done
done