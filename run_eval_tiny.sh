#!/bin/bash
# ============================================================
# ConvNeXt COCO 评测：剩余步骤一键脚本（本机已执行过）
# 环境: conda convnext_det | 数据: COCO val2017 | 模型: Tiny Cascade
# ============================================================
set -e
source /root/miniconda3/etc/profile.d/conda.sh

# ---------- 1) 环境（若已创建可跳过） ----------
# conda create -n convnext_det python=3.8 -y
conda activate convnext_det

# ---------- 2) 依赖（若已安装可跳过） ----------
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
#   --index-url https://download.pytorch.org/whl/cu118
# pip install https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/mmcv_full-1.7.2-cp38-cp38-manylinux1_x86_64.whl
# pip install timm==0.3.2 terminaltables pycocotools==2.0.7 opencv-python-headless openmim
# cd /root/autodl-tmp/Swin-Transformer-Object-Detection && pip install -e . --no-deps

# ---------- 3) 数据（若已解压可跳过） ----------
# mkdir -p /root/autodl-tmp/data/coco /root/autodl-tmp/checkpoints /root/autodl-tmp/logs
# cd /root/autodl-tmp/data/coco
# unzip -n /autodl-pub/data/COCO2017/val2017.zip
# unzip -n /autodl-pub/data/COCO2017/annotations_trainval2017.zip
# mkdir -p /root/autodl-tmp/Swin-Transformer-Object-Detection/data
# ln -sfn /root/autodl-tmp/data/coco /root/autodl-tmp/Swin-Transformer-Object-Detection/data/coco

# ---------- 4) 权重（若已下载可跳过） ----------
# cd /root/autodl-tmp/checkpoints
# wget -c https://dl.fbaipublicfiles.com/convnext/coco/cascade_mask_rcnn_convnext_tiny_1k_3x.pth

# ---------- 5) 评测（保存日志） ----------
cd /root/autodl-tmp/Swin-Transformer-Object-Detection
export PYTHONPATH=/root/autodl-tmp/Swin-Transformer-Object-Detection:$PYTHONPATH

CONFIG=configs/convnext/cascade_mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k.py
CKPT=/root/autodl-tmp/checkpoints/cascade_mask_rcnn_convnext_tiny_1k_3x.pth
OUT=/root/autodl-tmp/results_tiny.pkl
LOG=/root/autodl-tmp/logs/eval_tiny.log

mkdir -p /root/autodl-tmp/logs

python tools/test.py "$CONFIG" "$CKPT" \
  --eval bbox segm \
  --out "$OUT" \
  2>&1 | tee "$LOG"

echo "日志: $LOG"
echo "预期: box mAP ~50.4 / mask mAP ~43.7"
