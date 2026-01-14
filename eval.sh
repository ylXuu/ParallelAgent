#!/bin/bash

SGLANG_ENV="/path/to/your/sglang/env/bin/activate"
EVAL_ENV="/path/to/your/eval/env/bin/activate"
ENV_FILE=".env"
MODEL_NAME="/path/to/your/model"
PORT=30000
LOG_PATH="/path/to/your/log"


set -a
source "$ENV_FILE"
set +a


echo ">>> Activating SGLANG environment"
source "$SGLANG_ENV"


SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 python -m sglang.launch_server \
    --model-path ${MODEL_NAME} \
    --tp-size 8 \
    --ep-size 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000 \
    --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}' \
    --context-length 131072 \
    --reasoning-parser deepseek-r1 \
    --api-key token-abc123 > "$LOG_PATH" 2>&1 &
SERVER_PID=$!


echo ">>> Waiting for server on port $PORT to be ready..."
until curl -s "http://localhost:$PORT/v1/models" >/dev/null; do
    sleep 5
done


echo ">>> Switching to evaluation environment"
source "$EVAL_ENV"


echo ">>> Running testing for self_manager ($MODEL_NAME)..."
python -m inference.run_eval --agent_name self_manager --model_name "$MODEL_NAME" --dataset "deepresearch_bench"
echo ">>> Testing completed for self_manager ($MODEL_NAME)."


echo ">>> Evaluation complete. Shutting down server..."
kill $SERVER_PID
sleep 20
if ps -p $SERVER_PID > /dev/null; then
    kill -9 $SERVER_PID
fi
