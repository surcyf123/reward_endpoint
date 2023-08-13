#!/bin/bash

START_PORT=10400
NUM_INSTANCES=3
NAME=$START_PORT

# Start NUM_INSTANCES instances
for ((i=0; i<$NUM_INSTANCES; i++))
do
    # Calculate the GPU ID and port for this instance
    GPU_ID=$i
    PORT=$((START_PORT + i))

    # Start the process with pm2
    pm2 start --name "${PORT}" --interpreter=python3 ~/reward_endpoint/reward_endpoint2.py -- --gpu $GPU_ID --port $PORT
done

