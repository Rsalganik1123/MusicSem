#!/bin/bash

LOG_FILE="mureka_restart.log"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

echo "[$(timestamp)] Starting monitoring script" >> $LOG_FILE

count=0
max_attempts=30

# "$@" forwards all arguments passed to this script to mureka.py
while [ $count -lt $max_attempts ]; do
    count=$((count+1))
    echo "[$(timestamp)] Starting mureka run (Attempt $count of $max_attempts)" >> $LOG_FILE
    python3 frozen_main.py "$@"
    echo "[$(timestamp)] mureka run stopped, restarting in 10 seconds..." >> $LOG_FILE

    if [ $count -lt $max_attempts ]; then
        sleep 10
    else
        echo "[$(timestamp)] Reached maximum number of attempts ($max_attempts). Exiting." >> $LOG_FILE
    fi
done