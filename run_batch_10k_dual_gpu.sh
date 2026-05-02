#!/bin/bash
# 双GPU并行生成风机叶片缺陷数据集 (多样性增强+防记忆)
# GPU 0: crack, erosion, coating_damage, surface_damage, contamination (5种, ~556张)
# GPU 1: peeling, hole, burn_damage, delamination (4种, ~444张)
# 总计约10000张

PYTHON=/home/cn/.conda/envs/diffusers/bin/python
SCRIPT=/home/cn/yolo/AnomalyAny/generate_batch_10k.py
OUTPUT=/home/cn/yolo/AnomalyAny/dataset_blade_defect_10k

mkdir -p $OUTPUT

echo "=========================================="
echo "双GPU并行生成风机叶片缺陷数据集"
echo "多样性增强: 随机prompt/位置/严重度/光照/纹理"
echo "防记忆: 随机guidance/steps + 输出噪声"
echo "GPU 0: crack, erosion, coating_damage, surface_damage, contamination"
echo "GPU 1: peeling, hole, burn_damage, delamination"
echo "总计: ~10000张"
echo "=========================================="

# GPU 0: 5种缺陷, 每种约1112张 = 5560张
CUDA_VISIBLE_DEVICES=0 $PYTHON $SCRIPT \
    --output-dir $OUTPUT \
    --total 5560 \
    --steps 0 \
    --guidance 0 \
    --seed 42 \
    --noise-strength 0.02 \
    --defect crack,erosion,coating_damage,surface_damage,contamination \
    > $OUTPUT/gpu0.log 2>&1 &
PID0=$!

# GPU 1: 4种缺陷, 每种约1111张 = 4444张
CUDA_VISIBLE_DEVICES=1 $PYTHON $SCRIPT \
    --output-dir $OUTPUT \
    --total 4444 \
    --steps 0 \
    --guidance 0 \
    --seed 12345 \
    --noise-strength 0.02 \
    --defect peeling,hole,burn_damage,delamination \
    > $OUTPUT/gpu1.log 2>&1 &
PID1=$!

echo "GPU 0 进程 PID: $PID0"
echo "GPU 1 进程 PID: $PID1"
echo "日志文件:"
echo "  $OUTPUT/gpu0.log"
echo "  $OUTPUT/gpu1.log"
echo ""
echo "监控命令:"
echo "  tail -f $OUTPUT/gpu0.log"
echo "  tail -f $OUTPUT/gpu1.log"
echo ""
echo "等待两个进程完成..."

wait $PID0
echo "GPU 0 进程完成 (退出码: $?)"

wait $PID1
echo "GPU 1 进程完成 (退出码: $?)"

echo ""
echo "=========================================="
echo "全部完成! 查看结果:"
echo "  for d in $OUTPUT/*/; do echo \"\$(basename \$d): \$(ls \$d/*.png 2>/dev/null | wc -l)\"; done"
echo "=========================================="
