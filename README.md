### 第一步：安装 Python 3.10 本体
Codespace 的默认镜像通常不带 3.10。先执行这个，否则你建不了 3.10 的虚拟环境：
```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

```
### 第二步：重建虚拟环境并激活
这里我们要强行指定 3.10 来创建，确保不被系统默认的 3.12 干扰：
```bash
# 如果有旧的 venv 文件夹先删掉
rm -rf venv

# 创建并激活
python3.10 -m venv venv
source venv/bin/activate

```
*激活后，请确认终端左侧出现了 (venv)。*
### 第三步：一键安装 Mediapipe 及其配套依赖
我们直接用你最稳的那套版本组合：
```bash
# 1. 基础工具升级
pip install --upgrade pip

# 2. 安装所有库（注意使用 headless 版避免 libGL 报错）
pip install numpy==1.23.5 tqdm mediapipe==0.10.11 protobuf==3.20.3 opencv-python-headless==4.8.0.74

```
### 第四步：补充系统依赖 (补丁)
即便是新账号的 Codespace，底层 Linux 依然可能缺库：
```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

```
### 第五步：验证环境
运行这行，如果看到 **Python: 3.10** 且没有报错，就说明你“复活”成功了：
```bash
python -c "import cv2, mediapipe as mp; print('Python:', __import__('sys').version.split()[0]); print('OpenCV & MediaPipe OK')"

如有报错，运行下面之后再次验证
# 1. 彻底卸载当前的 OpenCV 
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

# 2. 安装无头版 (它不需要 libGL)
pip install opencv-python-headless==4.8.0.74

运行：python scripts/extract_pose.python

shape：
sample1: (1091, 33, 2) (1091, 20)
sample2: (1615, 33, 2) (1615, 20)
sample3: (1604, 33, 2) (1604, 20)
sample4: (3921, 33, 2) (3921, 20)


# 1. 基础步行场景（用来建立坐标波动的 Baseline）
curl -L -o sample1.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4

# 下载几个公开的姿态/表情测试视频 (这些是轻量级的)
curl -L -o sample2.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4
curl -L -o sample3.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4

# 4. 交互类
curl -L -o sample4.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4



