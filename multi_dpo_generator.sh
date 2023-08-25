#!/bin/bash
occupied_gpus=4
START_PORT=30000
NUM_INSTANCES=4
NAME=$START_PORT

# Start NUM_INSTANCES instances
for ((i=$occupied_gpus; i<($occupied_gpus+$NUM_INSTANCES); i++))
do
    # Calculate the GPU ID and port for this instance
    GPU_ID=$i
    PORT=$((START_PORT + i))

    # Start the process with pm2
    pm2 start --name "${PORT}" --time --interpreter=python3 /root/reward_endpoint/dpo_generator.py -- --gpu $GPU_ID --port $PORT
done
