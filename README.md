# 🧠 Federated Learning on Kubernetes (K3s) with Flower 🌸

Privacy-Preserving Federated Learning system using **K3s**, **TensorFlow**, and **Flower**. Designed for distributed, resource-constrained environments like Raspberry Pi, GPU servers, and edge devices. Trains medical image models (e.g. chest X-rays) with strict data privacy using cutting-edge FL, DP, and encryption technologies.

---

## 🚀 Features

- ⚙️ **Federated Learning Orchestration** with K3s
- 🌸 **Flower Framework** with FedAdam / FedProx strategies
- 🔐 **Privacy-Preserving** via:
  - Differential Privacy (TensorFlow Privacy)
  - Homomorphic Encryption (TenSEAL)
  - HTTPS/TLS Communication
  - OAuth2/JWT Authentication
- 📦 **Model Compression**: Mixed Precision, Quantization (TF Lite)
- 📈 **Real-Time Monitoring** via Prometheus + Grafana
- 🧑‍⚕️ **Medical Domain**: Chest X-ray image classification
- 🔄 **Kubernetes Jobs** for on-demand training
- 📊 **Federated Metrics Aggregation**

---

## 🧱 Architecture

```plaintext
+-------------------+       +--------------------+
|  Client 1 (Pi)    | <---> |                    |
|  InceptionV3      |       |                    |
|                   |                    |
+-------------------+       |                    |
                            |                    |
+-------------------+       |                    |
|  Client 2 (GPU)   | <---> |   Aggregator Node  |
|  InceptionV3      |       |   (K3s Master)     |
|                   |       |                    |
+-------------------+       |                    |
                            |    Flower Server   |   
                           +--------------------+

🏁 Getting Started
1. Clone the repo

git clone https://github.com/sagnik3788/federated-k8s-demo.git
cd federated-k8s-demo

2. Set up virtual environment

python3 -m venv federated-env
source federated-env/bin/activate
pip install -r requirements.txt

3. Spin up K3s cluster

Install K3s on your server and join your clients (e.g. GPU server, Raspberry Pi, etc).

curl -sfL https://get.k3s.io | sh -

4. Deploy with Kubernetes

Apply the YAML files inside the kubernetes/ directory:

kubectl apply -f kubernetes/

5. Run Federated Server

cd server/
python server.py

6. Start Clients

Each client will run:

cd client1/  # or client2/, client3/
python client.py

📊 Monitoring

You can view metrics via Grafana dashboard:

    Accuracy over Rounds

    Loss per Client

    Communication Overhead

Deployed via:

kubectl apply -f kubernetes/prometheus.yaml
kubectl apply -f kubernetes/grafana.yaml

📂 Directory Structure

.
├── client1/                 # Raspberry Pi client (MobileNetV3)
├── client2/                 # GPU client (VGG19)
├── client3/                 # GPU client (InceptionV3)
├── server/                  # Federated Learning server (Flower)
│   ├── final_model.h5       # (ignored)
│   ├── final_inception_model.keras (ignored)
│   └── full_dataset_vectors.h5 (ignored)
├── kubernetes/              # K3s deployment YAMLs
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignore large datasets and models
└── README.md

📁 .gitignore (Important!)

This repo ignores:

    Large models (*.keras, *.h5)

    Datasets (*/dataset/)

    Virtual environment (federated-env/)

Make sure large files are not pushed to GitHub.
📈 Performance Targets

    ✅ >92% validation accuracy on chest X-ray

    ⏱️ <2 hours per FL round (10 clients, 100 epochs)

    📦 <50 MB client update size (compressed)

    🛡️ DP Budget ε < 2.0

⚙️ Technologies Used

    🐍 Python 3.10

    📦 TensorFlow, Keras

    🌸 Flower

    🛡️ TensorFlow Privacy, TenSEAL

    🐳 Docker + K3s (Kubernetes Lightweight)

    📊 Prometheus, Grafana

    ⚡ TF Lite, FP16

    🌐 HTTPS, JWT

📜 License

MIT © Sagnik Das
🙌 Acknowledgements

    Flower

    TensorFlow Privacy

    TenSEAL

    K3s