export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MODEL_PATH="/usr/models/audio2video/a2v_opensource"
PROMPT="A young girl's eyes kept following the camera, singing with a mysterious expression. Surrounded by rich blooming florals, the camera cranes up slowly to reveal the full surreal, luxurious scene."
INPUT_IMAGE="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single1.png"
INPUT_AUDIO="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/talking_avatar_video/single_actor/huahai_5s.mp3"

python generate_video.py \
    --task_type talking_avatar \
    --model_id $MODEL_PATH \
    --prompt "$PROMPT" \
    --seed 42 \
    --low_vram \
    --input_image "$INPUT_IMAGE" \
    --input_audio "$INPUT_AUDIO"
