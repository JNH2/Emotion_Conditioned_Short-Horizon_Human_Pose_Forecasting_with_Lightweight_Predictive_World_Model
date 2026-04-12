快速启动环境（Python 3.10）
如果codespace重启或环境损坏，请按顺序执行：
1.系统级修复（解决libGL报错）：
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
强行指定310重建虚拟环境（纺织系统默认跳回3.12）：
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
一键安装所有库：
pip install --upgrade pip
pip install -r requirements.txt

shape：
sample1: (1091, 33, 2)
sample2: (1091, 33, 2)
sample3: (1615, 33, 2)
sample4: (1091, 33, 2)
sample5: (1615, 33, 2)
sample6: (647, 33, 2)
sample7: (3921, 33, 2)
sample8: (596, 33, 2)
sample9: (1604, 33, 2)
samle10: (1604, 33, 2)


# 进入 data 文件夹
cd data

# 1. 基础步行场景（用来建立坐标波动的 Baseline）
curl -L -o sample1.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4

# 下载几个公开的姿态/表情测试视频 (这些是轻量级的)
curl -L -o sample2.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking-and-pause.mp4
curl -L -o sample3.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4

# 4. 演讲类（上半身动作丰富）
curl -L -o sample4.mp4 https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/face-demographics-walking-and-pause.mp4

# 5. 交互类
curl -L -o sample5.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-female.mp4


# 6行人正面（全身坐标测试）
curl -L -o sample6.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4

# 7. 多人场景（测试 MediaPipe 在复杂背景下的单人锁定，通常会抓画面中心的人）
curl -L -o sample7.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4

# 8. 室内监控视角（用于模拟 SEMAINE 这种固定摄像头的交互）
curl -L -o sample8.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4


# 9. 面部/头部偏转（专门测 Pose 里的面部 0-10 号关键点）
curl -L -o sample9.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-male.mp4


# 10. 男性面部特写（增加不同性别/特征的泛化能力）
curl -L -o sample10.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/head-pose-face-detection-male.mp4

