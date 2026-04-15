# MLOps MLflow Practice Report – Kubernetes Pipeline  
**MLOps Fundamentals - LTAT.02.038**

## Overview

In this practice, I built a full local MLOps pipeline using Kubernetes (Minikube), MLflow, and a FastAPI application. The goal was to move from simple containerization towards a more realistic deployment setup where model training, tracking, and serving are handled as separate components inside a cluster.

The system consists of three main parts:
- MLflow for experiment tracking and model storage  
- a training job for building models  
- a FastAPI application for serving predictions  

---

## Steps Performedx

### 1. Setting up Kubernetes Environment
- I installed and configured Minikube.
- Learned basic Kubernetes commands:
  - `kubectl apply`
  - `kubectl get pods`
  - `kubectl logs`
- Created a namespace `mlops`.

---

### 2. Deploying MLflow
- Built a custom Docker image for MLflow.
- Deployed MLflow as a Kubernetes Deployment.
- Exposed it via a Service.
- Added persistent storage (PVC) to store artifacts.

---

### 3. Creating Training Job
- Implemented a training script using `RandomForestRegressor`.
- Loaded taxi dataset (parquet format).
- Logged metrics and model to MLflow.
- Created a Kubernetes Job to run training.

---

### 4. Building Prediction App
- Developed a FastAPI application.
- Added endpoints:
  - `/`
  - `/predict`
  - `/health`
  - `/model-info`
- Integrated MLflow to load the best model at startup.

---

### 5. Model Selection Logic
- Queried MLflow runs.
- Selected best model based on lowest RMSE.
- Stored model URI as:
  runs:/<run_id>/model

---

### 6. Connecting Components
- Used Kubernetes Services:
  - `mlflow-service`
  - `taxi-app-service`
- Enabled internal communication via service names.

---

## Challenges and Fixes

### Memory issues (OOMKilled)
- Training job failed due to memory limits.
- Fix:
  - reduced dataset (single parquet file)
  - reduced model size (`n_estimators=50`, `n_jobs=1`)

---

### Docker & Minikube mismatch
- Images not visible inside cluster.
- Fix:
```bash
eval $(minikube docker-env)
```

---

### MLflow artifact issues
- Model could not be loaded.
- Fix:
  - introduced shared PVC
  - mounted `/mlflow` in both MLflow and app

---

### Model URI issues
- `models:/...` did not work reliably.
- Fix:
  - switched to `runs:/<run_id>/model`
  - stored URI as tag

---

### App not updating model
- Model URI showed `None`.
- Fix:
  - restart deployment:
```bash
kubectl rollout restart deployment/taxi-app -n mlops
```

---

### Port-forward issues
- `kubectl port-forward` unreliable.
- Fix:
```bash
minikube service taxi-app-service --url
```

---

## Learnings

- Kubernetes enables modular system design.
- MLflow integration requires careful handling of artifacts.
- Resource limits strongly affect ML workloads.
- Debugging distributed systems is mostly environment-related.

---

## Screenshots

(Add screenshots here)

- Running FastAPI UI  
- MLflow UI with experiments  
- Kubernetes pods view  
- Health endpoint response  

---

## Ingress

/to be updated/

---

## Current Status

- MLflow running in Kubernetes  
- Training job successful  
- App loads best model  
- Predictions working  
- UI accessible  

---

## Conclusion

I successfully built a working MLOps pipeline using Kubernetes, MLflow, and FastAPI. Despite several issues related to environment setup and resource limits, I was able to resolve them and understand how real-world MLOps systems operate.