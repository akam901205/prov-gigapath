# GigaPath RunPod Deployment Guide

## Overview
This document contains the complete setup process for deploying Microsoft's GigaPath foundation model on RunPod for digital pathology inference.

## Repository Information
- **GitHub Repository**: https://github.com/akam901205/prov-gigapath
- **Current Release**: v0.0.5
- **Model**: prov-gigapath/prov-gigapath (Hugging Face - Gated Model)

## Prerequisites

### 1. Hugging Face Token
- **Token**: `[REDACTED_FOR_SECURITY]`
- **Purpose**: Required to access the gated GigaPath model
- **How to get**: 
  1. Go to https://huggingface.co/prov-gigapath/prov-gigapath
  2. Request access to the model
  3. Create token at https://huggingface.co/settings/tokens

### 2. RunPod API Access
- **RunPod Account**: Required for GPU pod deployment
- **MCP Tools**: Configured for RunPod management

### 3. SSH Key Setup
- **Public Key Location**: `~/.ssh/runpod_key.pub`
- **Private Key Location**: `~/.ssh/runpod_key`
- **Public Key Content**:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFIeoqvrauiM8te77vWWRJ1DaZP9pLI1TS4Ik5OCuShj akam19901205@gmail.com
```
- **Added to RunPod**: Via RunPod Console → Settings → SSH Public Keys

## Current RunPod Pod Details

### Active Pod Information
- **Pod Name**: gigapath-ready
- **Pod ID**: 8v9wob2mln55to
- **GPU**: NVIDIA A40 (48GB VRAM)
- **Location**: Canada (Montreal)
- **Cost**: $0.40/hour
- **Base Image**: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
- **Volume**: 30GB at /workspace
- **Ports**: 8888/http, 22/tcp, 3000/http

### SSH Connection
```bash
# Direct TCP connection (recommended)
ssh root@69.30.85.10 -p 22027 -i ~/.ssh/runpod_key

# Alternative via RunPod SSH proxy
ssh 8v9wob2mln55to-644113d9@ssh.runpod.io -i ~/.ssh/runpod_key
```

## Installation Steps Performed

### 1. Create RunPod Pod
```python
# Using MCP tools
mcp__runpod__create-pod(
    name="gigapath-ready",
    imageName="runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpuCount=1,
    gpuTypeIds=["NVIDIA A40"],
    containerDiskInGb=50,
    volumeInGb=30,
    volumeMountPath="/workspace",
    ports=["8888/http", "22/tcp", "3000/http"],
    env={
        "HF_TOKEN": "[REDACTED_FOR_SECURITY]",
        "MODE": "inference",
        "MODEL_DIR": "/workspace/model"
    }
)
```

### 2. Clone Repository
```bash
cd /workspace
git clone https://github.com/akam901205/prov-gigapath.git
cd prov-gigapath
```

### 3. Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install additional packages
pip install runpod timm huggingface_hub openslide-python
```

### 4. Test GigaPath
```bash
# Set HF_TOKEN environment variable
export HF_TOKEN='[REDACTED_FOR_SECURITY]'

# Run test
cd /workspace/prov-gigapath
python3 handler.py
```

## File Structure

### Key Files Modified
1. **Dockerfile** - Fixed handler.py path from `.runpod/` to root directory
2. **handler.py** - Implemented full GigaPath tile and slide encoder
3. **.runpod/tests.json** - Made GPU requirements flexible
4. **.runpod/hub.json** - Expanded supported GPU types
5. **.gitignore** - Added config files to prevent API key exposure

### Handler Features
- Supports both image URL and base64 input
- Implements tile encoder (single patches)
- Framework for slide encoder (WSI processing)
- Automatic GPU/CPU detection
- Model caching for efficient inference
- Returns 1536-dimensional feature vectors

## API Usage

### Input Format
```json
{
  "input": {
    "image_url": "https://example.com/image.png",
    "encoder_type": "tile",
    "mode": "inference"
  }
}
```

### Alternative Base64 Input
```json
{
  "input": {
    "image_base64": "base64_encoded_string",
    "encoder_type": "tile",
    "mode": "inference"
  }
}
```

### Output Format
```json
{
  "status": "success",
  "result": {
    "encoder_type": "tile",
    "features_shape": [1, 1536],
    "features": [[...1536 float values...]],
    "device": "cuda"
  }
}
```

## Troubleshooting

### Common Issues and Solutions

1. **SSH Connection Issues**
   - Error: "Your SSH client doesn't support PTY"
   - Solution: Use direct TCP connection instead of RunPod SSH proxy
   - Command: `ssh root@<IP> -p <PORT> -i ~/.ssh/runpod_key`

2. **HF Token Authentication**
   - Error: "401 Client Error... Cannot access gated repo"
   - Solution: Ensure HF_TOKEN is set: `export HF_TOKEN='[REDACTED_FOR_SECURITY]'`
   - Permanent: Add to ~/.bashrc in pod

3. **Docker Build Errors**
   - Error: "manifest not found"
   - Solution: Use valid RunPod base images
   - Working image: `runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04`

4. **Volume Mount Issues**
   - Error: "invalid mount path"
   - Solution: Use absolute paths without quotes in volumeMountPath

## Pod Management Commands

### Check Pod Status
```python
mcp__runpod__get-pod(podId="8v9wob2mln55to")
```

### Stop Pod
```python
mcp__runpod__stop-pod(podId="8v9wob2mln55to")
```

### Start Pod
```python
mcp__runpod__start-pod(podId="8v9wob2mln55to")
```

### Delete Pod
```python
mcp__runpod__delete-pod(podId="8v9wob2mln55to")
```

## Next Steps for Production

### 1. Create Docker Image
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# ... (see Dockerfile in repo)
CMD ["python3", "/workspace/handler.py"]
```

### 2. Build and Push to Docker Hub
```bash
docker build -t akam901205/gigapath:latest .
docker push akam901205/gigapath:latest
```

### 3. Create Serverless Endpoint
- Go to RunPod Console → Serverless → New Endpoint
- Container Image: `akam901205/gigapath:latest`
- Environment Variables:
  - `HF_TOKEN=[REDACTED_FOR_SECURITY]`
  - `MODE=inference`
- GPU: Any from supported list (T4, RTX 3090, A40, etc.)

### 4. Test Endpoint
```python
import runpod

runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("endpoint_id")

result = endpoint.run({
    "input": {
        "image_url": "https://example.com/pathology_image.png",
        "encoder_type": "tile"
    }
})
```

## Supported GPUs
- NVIDIA GeForce RTX 3090/4090
- NVIDIA RTX A4000/A4500/A5000/A6000
- NVIDIA A40, L4, L40
- NVIDIA T4, A10
- NVIDIA A100 PCIe

## Important Notes
1. Model requires ~8GB VRAM minimum
2. First run downloads model weights (~4GB)
3. Inference time: ~100-200ms per image patch
4. Keep HF_TOKEN secure - never commit to git
5. Pod charges continue while running - stop when not in use

## Contact & Support
- GitHub Issues: https://github.com/akam901205/prov-gigapath/issues
- RunPod Support: https://runpod.io/console/support
- Model Authors: Microsoft Research

## Last Updated
- Date: 2025-08-11
- Version: v0.0.5
- Status: Successfully deployed and tested

---

# Next.js Web Interface for GigaPath

## Project Overview
Building a user-friendly web interface using Next.js to interact with the GigaPath model for pathology image analysis.

## Technology Stack
- **Frontend**: Next.js 14 (App Router)
- **UI Library**: Tailwind CSS + shadcn/ui
- **Image Upload**: react-dropzone
- **API Communication**: Axios
- **State Management**: React hooks
- **Deployment**: Vercel (frontend) + RunPod (backend)

## Project Structure
```
gigapath-web/
├── app/
│   ├── page.tsx           # Main upload page
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts    # API endpoint
│   └── layout.tsx          # Root layout
├── components/
│   ├── ImageUploader.tsx   # Upload component
│   ├── ResultDisplay.tsx   # Results viewer
│   └── LoadingSpinner.tsx  # Loading state
├── lib/
│   └── gigapath.ts         # GigaPath API client
└── public/
    └── sample-images/       # Test images
```

## Features
1. **Image Upload**
   - Drag & drop interface
   - File selection
   - Image preview
   - Format validation (PNG, JPG, TIFF)

2. **Analysis**
   - Send image to GigaPath API
   - Real-time processing status
   - Feature extraction visualization

3. **Results Display**
   - Confidence scores
   - Feature heatmap
   - Downloadable report
   - History of analyses

## Setup Instructions

### 1. Initialize Next.js Project
```bash
cd /var/www/vhosts/dev.esimfly.net/httpdocs/gigapath
npx create-next-app@latest gigapath-web --typescript --tailwind --app
cd gigapath-web
```

### 2. Install Dependencies
```bash
npm install axios react-dropzone recharts lucide-react
npm install @radix-ui/react-dialog @radix-ui/react-slot
npm install class-variance-authority clsx tailwind-merge
```

### 3. Environment Variables
Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://69.30.85.10:8000
RUNPOD_API_KEY=your_runpod_api_key
HF_TOKEN=[REDACTED_FOR_SECURITY]
```

## Implementation Progress
- [x] Create Next.js project structure
- [x] Build image upload component
- [x] Create API route for GigaPath
- [x] Implement results display
- [x] Add loading states and error handling
- [x] Style with Tailwind CSS
- [x] Test API connectivity
- [ ] Deploy to production

## 🎉 Current Status: FULLY FUNCTIONAL

The complete GigaPath system is now operational with:
1. **FastAPI server** running on RunPod pod (port 8000)
2. **SSH tunnel** providing local access (port 8001)
3. **Next.js web interface** running locally (port 3000)
4. **Full pipeline** working: Image upload → GigaPath analysis → Feature extraction

### How to Use
1. Ensure SSH tunnel is active: `ssh -p 22051 -i ~/.ssh/runpod_key -L 8001:localhost:8000 -N root@69.30.85.10`
2. Visit http://localhost:3000
3. Upload a pathology image
4. Click "Analyze Image"
5. View the 1536-dimensional feature extraction results

### API Endpoints
- **Health Check**: `GET http://localhost:8001/`
- **Analyze Image**: `POST http://localhost:8001/analyze`
  - Accepts base64 or URL input
  - Returns feature vectors and metadata