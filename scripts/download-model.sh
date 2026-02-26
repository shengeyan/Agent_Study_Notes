#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Qwen1.5-0.5B-Chat 模型下载脚本${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. 创建目录
CACHE_DIR="$HOME/.cache/huggingface/hub/models--Xenova--Qwen1.5-0.5B-Chat/snapshots/main"
echo -e "${YELLOW}[1/6]${NC} 创建缓存目录..."
mkdir -p "$CACHE_DIR/onnx"
echo -e "${GREEN}✓${NC} 目录创建完成: $CACHE_DIR\n"

# 进入目录
cd "$CACHE_DIR"

# 2. 下载小文件
echo -e "${YELLOW}[2/6]${NC} 下载 config.json..."
curl -# -L -o config.json https://huggingface.co/Xenova/Qwen1.5-0.5B-Chat/resolve/main/config.json
echo -e "${GREEN}✓${NC} config.json 下载完成\n"

echo -e "${YELLOW}[3/6]${NC} 下载 generation_config.json..."
curl -# -L -o generation_config.json https://huggingface.co/Xenova/Qwen1.5-0.5B-Chat/resolve/main/generation_config.json
echo -e "${GREEN}✓${NC} generation_config.json 下载完成\n"

echo -e "${YELLOW}[4/6]${NC} 下载 tokenizer_config.json..."
curl -# -L -o tokenizer_config.json https://huggingface.co/Xenova/Qwen1.5-0.5B-Chat/resolve/main/tokenizer_config.json
echo -e "${GREEN}✓${NC} tokenizer_config.json 下载完成\n"

echo -e "${YELLOW}[5/6]${NC} 下载 tokenizer.json (~2MB)..."
curl -# -L -o tokenizer.json https://huggingface.co/Xenova/Qwen1.5-0.5B-Chat/resolve/main/tokenizer.json
echo -e "${GREEN}✓${NC} tokenizer.json 下载完成\n"

# 3. 下载大文件（模型）
echo -e "${YELLOW}[6/6]${NC} 下载模型文件 decoder_model_merged_quantized.onnx (~500MB)..."
echo -e "${BLUE}这可能需要 5-10 分钟，请耐心等待...${NC}\n"
curl -# -L -o onnx/decoder_model_merged_quantized.onnx https://huggingface.co/Xenova/Qwen1.5-0.5B-Chat/resolve/main/onnx/decoder_model_merged_quantized.onnx
echo -e "${GREEN}✓${NC} 模型文件下载完成\n"

# 4. 验证文件
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  下载完成！文件列表：${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "主目录文件："
ls -lh | grep -v "^d"

echo -e "\nONNX 目录文件："
ls -lh onnx/

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  ✓ 所有文件下载成功！${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "现在可以运行："
echo -e "${BLUE}node src/1/1-3-Transformers.js${NC}\n"
