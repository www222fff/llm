# 使用官方 Python 3.9 slim 镜像作为基础
FROM python:3.9-slim

# 安装必要的工具
RUN apt-get update && apt-get install -y git git-lfs && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制 Python 脚本
COPY dataset.json .
COPY step2-train-cpu.py .
COPY step2-train-gpu.py .

# 设置环境变量，确保 Python 输出实时显示
ENV PYTHONUNBUFFERED=1

# 运行 Python 脚本
CMD ["python", "step2-train-cpu.py"]
