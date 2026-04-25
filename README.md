🧠 Emotion-Conditioned Short-Horizon Human Pose Forecasting

A lightweight multimodal predictive world model for short-horizon human motion forecasting, integrating facial-expression-derived affective embeddings with pose dynamics.

👥 Authors: 
Jingni Huang
📧 jingni.huang@kellogg.ox.ac.uk/ jingnih@gmail.com, Peter Bloodsworth
📧 peter.bloodsworth@cs.ox.ac.uk

Citation: 
@misc{huang2025emotionpose,
  title={Emotion-Conditioned Short-Horizon Human Pose Forecasting with a Lightweight Predictive World Model},
  author={Jingni Huang and Peter Bloodsworth},
  year={2025},
  note={GitHub repository},
  howpublished={\url{https://github.com/your-username/your-repo}}
}

📚 Related Work:
This work is inspired by the predictive world model paradigm proposed by LeCun et al., where intelligent systems learn compact latent representations to capture the dynamics of the physical world rather than directly predicting observations.
Predictive world models focus on learning temporally consistent latent state transitions that support reasoning, planning, and long-horizon prediction. Instead of optimizing short-term geometric accuracy, these models aim to capture the evolution of the environment through internal representations.
Our approach adopts this perspective and applies it to short-horizon multimodal pose prediction by integrating affective signals into the latent dynamics.

Unlike large-scale world model architectures that require extensive computational resources and Transformer-based backbones, our method focuses on lightweight multimodal fusion and autoregressive rollout prediction. This enables efficient deployment while preserving the core idea of predictive state representation learning.

Recent work on JEPA-based world models further explores stable latent representation learning for long-horizon reasoning, reinforcing the importance of predictive latent dynamics over direct observation prediction.


🚀 Overview:
Human motion is not purely kinematic—it is strongly influenced by latent affective and intentional states.
This project explores:
   •   How facial-expression-derived emotion signals influence motion prediction
   •   Whether multimodal fusion improves short-horizon forecasting
   •   How lightweight world models behave under affect perturbations
We propose a lightweight emotion-conditioned predictive world model that:
   •   Extracts pose (33×2) and facial embeddings (20D)
   •   Learns a gated multimodal representation
   •   Performs short-horizon rollout prediction (15 frames)
   •   Evaluates robustness via counterfactual perturbation


🏗️ System Architecture:
Pipeline:
Video → Pose Extraction → Emotion Extraction → Fusion → Predictor → Rollout → Evaluation  → Counterfactual 
Modules:
   •   extract_pose.py → MediaPipe Pose (33 joints)
   •   extract_emotion.py → Face Mesh (facial landmarks subset)
   •   train_pose_baseline.py → Pose-only LSTM baseline
   •   train_fusion_predictor.py → Multimodal fusion model
   •   train_world_model_rollout.py → Autoregressive world model
   •   evaluate_model.py → MPJPE + denormalized evaluation
   •   counterfactual.py → robustness analysis


🧪 Key Results(Details see paper):
Dataset II (in-the-wild affect-driven sequences)
Model	Normalization	Gate	Test Loss	MPJPE
Pose Baseline	✓	-	0.0072	0.0334
Fusion (naive)	✗	-	0.0070	0.0389
Fusion + norm + α	✓	-	0.2776	N/A
Fusion + norm + gate	✓	0.098	0.2636	0.0232 ↓
World Model	✓	0.115	0.4164	N/A
Insights
   •   Naive fusion hurts performance → modality imbalance
   •   Gated fusion improves MPJPE significantly (↓30%)
   •   Emotion acts as auxiliary signal, not dominant driver
   •   World model shows stable rollout but higher loss



🔬 Counterfactual Analysis:
Model	Gate	Counterfactual Difference
Fusion	0.090	0.3077
World Model	0.109	0.0332 ↓
👉 World model is significantly more robust to perturbations


🧠 Key Contributions:
   •  Lightweight multimodal pose forecasting (runs on low-resource setup)
   •   Emotion-conditioned latent fusion with learned gating
   •   Empirical evidence: emotion improves prediction only when properly integrated
   •   First exploration (lightweight setting) of:
      •   affect-conditioned world models
      •   counterfactual robustness in pose forecasting


⚙️ Design Philosophy:
Inspired by LeCun’s world model paradigm, this project focuses on:
	Learning latent dynamics rather than direct prediction.
However:
   •   We use a minimal LSTM rollout model
   •   Designed for short-horizon forecasting (0.5–3s)
   •   Prioritizes efficiency and interpretability


📦 Dataset:
	Dataset I: Intel OpenVINO demo videos (controlled motion)
   	Dataset II: In-the-wild YouTube sequences (affect-driven motion)
👉 Only Dataset II shows clear benefit from emotion conditioning.


🧪 Metrics:
   MPJPE (Mean Per Joint Position Error)
→ computed in denormalized space
   Training/Test Loss
→ computed in normalized space (MSE)


🛠️ Tech Stack:
   •   PyTorch
   •   MediaPipe (Pose + Face Mesh)
   •   NumPy
   •   OpenCV


⚙️ Environment Setup: 
Step 1: Install Python 3.10
Codespaces usually do not include Python 3.10 by default.
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
Step 2: Rebuild Virtual Environment
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
pip install --upgrade pip
pip install numpy==1.23.5 tqdm mediapipe==0.10.11 protobuf==3.20.3 opencv-python-headless==4.8.0.74
Step 4: System Dependencies
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
Step 5: Verify Environment
python -c "import cv2, mediapipe as mp; print('Python:', __import__('sys').version.split()[0]); print('OpenCV & MediaPipe OK')"
If error:
pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
pip install opencv-python-headless==4.8.0.74



📊 Data Preparation:
Extract Pose
python scripts/extract_pose.py
Dataset I Shapes
sampleA: (1091, 33, 2) (1091, 20)
sampleB: (1615, 33, 2) (1615, 20)
sampleC: (1604, 33, 2) (1604, 20)
sampleD: (3921, 33, 2) (3921, 20)
Dataset II shape:
SampleE: (567, 33, 2) (567, 20)
SampleF: (95, 33, 2) (95, 20)
SampleG: (345, 33, 2) (345, 20)
SampleH: (851, 33, 2) (851, 20)
SampleI: (710, 33, 2) (710, 20)


📈 Future Work:
   •   Transformer-based world model
   •   Longer horizon prediction
   •   Better emotion representation (beyond landmarks)
   •   Cross-subject generalization







