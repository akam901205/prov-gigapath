# CLAUDE - GigaPath RunPod Connection Instructions

## Quick Fix When RunPod Restarts

When the user says RunPod restarted or connection is broken, follow these steps:

### 1. Get Current Pod Status
```bash
# Use MCP to get pod info
mcp__runpod__get-pod with podId: 8v9wob2mln55to
```

### 2. Extract SSH Port
Look for `"portMappings": {"22": XXXXX}` in the response. The XXXXX is the new SSH port.

### 3. Kill Old SSH Tunnels
```bash
pkill -f "ssh.*8001:localhost:8000"
pkill -f "ssh.*8002:localhost:8002"
pkill -f "ssh.*8003:localhost:8003"
pkill -f "ssh.*8004:localhost:8004"
pkill -f "ssh.*8005:localhost:8005"
```

### 4. Start RunPod Services
```bash
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key root@69.30.85.10 "/workspace/start_all_services.sh"
```

### 5. Install Python Packages (if needed)
If services fail to start, install packages:
```bash
ssh -p [NEW_SSH_PORT] -i ~/.ssh/runpod_key root@69.30.85.10 "pip install fastapi uvicorn torch torchvision timm pillow requests pydantic 'numpy<2' python-multipart scikit-learn matplotlib faiss-cpu --no-deps"
```

### 6. Set Up New SSH Tunnels
Run these in background:
```bash
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8001:localhost:8000 -N root@69.30.85.10 &
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8002:localhost:8002 -N root@69.30.85.10 &
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8003:localhost:8003 -N root@69.30.85.10 &
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8004:localhost:8004 -N root@69.30.85.10 &
ssh -p [NEW_SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8005:localhost:8005 -N root@69.30.85.10 &
```

### 7. Test Connection
```bash
curl http://localhost:8001/
```

## Important Info

- **Pod ID**: 8v9wob2mln55to
- **RunPod IP**: 69.30.85.10 (usually stays the same)
- **SSH Port**: Changes every restart (check with MCP)
- **Local Ports**:
  - 8001 → RunPod 8000 (Main API)
  - 8002 → RunPod 8002 (Fine-tuning)
  - 8003 → RunPod 8003 (Similarity)
  - 8004 → RunPod 8004 (Heatmap)
  - 8005 → RunPod 8005 (Differential)

## Common Issues

### NumPy Version Conflict
If you see NumPy errors, fix with:
```bash
ssh -p [SSH_PORT] -i ~/.ssh/runpod_key root@69.30.85.10 "pip install 'numpy<2'"
```

### Services Not Starting
Check logs:
```bash
ssh -p [SSH_PORT] -i ~/.ssh/runpod_key root@69.30.85.10 "tail -20 /workspace/api.log"
```

## Full Automation Code Block
When user reports connection issues, run this sequence (replace [SSH_PORT] with actual port):

```bash
# Get pod info first to find SSH port
mcp__runpod__get-pod podId=8v9wob2mln55to

# Then run these with the correct port:
pkill -f "ssh.*localhost:800"
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key root@69.30.85.10 "/workspace/start_all_services.sh"
sleep 5
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8001:localhost:8000 -N root@69.30.85.10 &
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8002:localhost:8002 -N root@69.30.85.10 &
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8003:localhost:8003 -N root@69.30.85.10 &
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8004:localhost:8004 -N root@69.30.85.10 &
ssh -p [SSH_PORT] -o StrictHostKeyChecking=no -i ~/.ssh/runpod_key -L 8005:localhost:8005 -N root@69.30.85.10 &
```