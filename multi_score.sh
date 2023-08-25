#!/bin/bash

START_PORT=30000
NUM_INSTANCES=8
NAME=$START_PORT

# Start NUM_INSTANCES instances
for ((i=0; i<$NUM_INSTANCES; i++))
do
    # Calculate the GPU ID and port for this instance
    GPU_ID=$i
    PORT=$((START_PORT + i))

    # Start the process with pm2
    pm2 start --name "${PORT}" --interpreter=python3 /root/reward_endpoint/openvalidators/reward_endpoint2.py -- --gpu $GPU_ID --port $PORT
done

