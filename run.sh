#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供一个参数"
    exit 1
fi

# 存储参数
SCENE_NAME=$1

# 执行三个命令
blender/blender --background --python new_run.py -- "$SCENE_NAME"
python new_optimize.py "$SCENE_NAME"
blender/blender --background --python render.py -- "$SCENE_NAME"
