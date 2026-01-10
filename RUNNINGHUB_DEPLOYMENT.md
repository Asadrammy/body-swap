# Runninghub GPU Deployment Guide

This guide provides step-by-step instructions for deploying the Face-Body Swap pipeline on Runninghub GPU instances.

## Prerequisites

- Runninghub GPU instance with CUDA support
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed
- Access to Runninghub dashboard

## Step 1: Prepare Your Runninghub Instance

### 1.1 Verify GPU Availability

```bash
# Check NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 1.2 Install Docker Compose (if not installed)

```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 1.3 Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Step 2: Clone and Prepare Project

### 2.1 Clone Repository

```bash
cd /path/to/your/projects
git clone <your-repo-url> face-body-swap
cd face-body-swap
```

### 2.2 Create Environment File

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# GPU Configuration
DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Paths
MODELS_DIR=/app/models
OUTPUTS_DIR=/app/outputs
TEMP_DIR=/app/temp
LOGS_DIR=/app/logs

# Model Configuration
FACE_DETECTOR=insightface
POSE_DETECTOR=mediapipe
GENERATOR_DEVICE=cuda
```

## Step 3: Build Docker Image

### 3.1 Build Image

```bash
docker-compose build
```

This will:
- Install all Python dependencies
- Download base models (InsightFace, MediaPipe)
- Set up the application environment

**Note**: First build may take 15-30 minutes depending on network speed.

### 3.2 Verify Build

```bash
docker images | grep face-body-swap
```

## Step 4: Start Services

### 4.1 Start with Docker Compose

```bash
docker-compose up -d
```

### 4.2 Check Container Status

```bash
docker-compose ps
docker-compose logs -f face-body-swap
```

### 4.3 Verify Health

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics
```

Expected response:
```json
{
  "status": "healthy",
  "gpu": {
    "available": true,
    "count": 1
  },
  "system": {
    "cpu_percent": 5.2,
    "memory_percent": 45.3
  }
}
```

## Step 5: Configure Runninghub Networking

### 5.1 Expose Port (if needed)

If Runninghub requires port configuration:
- Go to Runninghub dashboard
- Configure port forwarding: `8000 -> 8000`
- Note the public URL provided

### 5.2 Test API Access

```bash
# Test from local machine
curl http://your-runninghub-url:8000/health

# Test API endpoint
curl http://your-runninghub-url:8000/api/v1/templates
```

## Step 6: Verify GPU Usage

### 6.1 Check GPU in Container

```bash
docker exec face-body-swap-api nvidia-smi
```

### 6.2 Monitor GPU During Processing

```bash
# In one terminal
watch -n 1 nvidia-smi

# In another terminal, trigger a test job
curl -X POST http://localhost:8000/api/v1/swap \
  -F "customer_photos=@test_image.jpg" \
  -F "template_id=tpl_individual_casual_001"
```

You should see GPU memory usage increase during processing.

## Step 7: Production Configuration

### 7.1 Update Configuration

Edit `configs/production.yaml`:

```yaml
models:
  generator:
    device: cuda
    lora_paths: []  # Add your LoRA paths here

api:
  host: 0.0.0.0
  port: 8000
  timeout: 600  # Increase for large images

processing:
  image_size: 512
  max_image_size: 1024
  num_inference_steps: 50
```

### 7.2 Set Resource Limits

Update `docker-compose.yml`:

```yaml
services:
  face-body-swap:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Step 8: Monitoring and Maintenance

### 8.1 Health Checks

The container includes automatic health checks:
- Endpoint: `/health`
- Interval: 30 seconds
- Timeout: 10 seconds

Monitor health:
```bash
watch -n 5 'curl -s http://localhost:8000/health | jq'
```

### 8.2 Metrics Endpoint

Get detailed metrics:
```bash
curl http://localhost:8000/metrics | jq
```

Metrics include:
- CPU and memory usage
- GPU utilization
- Active job counts
- Disk usage

### 8.3 Log Monitoring

```bash
# View logs
docker-compose logs -f face-body-swap

# View last 100 lines
docker-compose logs --tail=100 face-body-swap

# View logs from file
tail -f logs/app.log
```

### 8.4 Restart Services

```bash
# Restart container
docker-compose restart face-body-swap

# Rebuild and restart
docker-compose up -d --build
```

## Step 9: Performance Optimization

### 9.1 Enable xformers (if available)

The pipeline automatically uses xformers if installed. Verify:

```bash
docker exec face-body-swap-api python -c "import xformers; print('xformers available')"
```

### 9.2 Configure LoRA Models

Add LoRA paths to `configs/production.yaml`:

```yaml
models:
  generator:
    lora_paths:
      - path: "models/lora/style_lora.safetensors"
        weight: 1.0
        name: "style_lora"
```

### 9.3 Batch Processing

For batch jobs, use the CLI:

```bash
docker exec face-body-swap-api python -m src.api.cli \
  --customer-dir /app/inputs \
  --template-dir /app/templates \
  --output-dir /app/outputs \
  --batch
```

## Step 10: Troubleshooting

### 10.1 Container Won't Start

```bash
# Check logs
docker-compose logs face-body-swap

# Check GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 10.2 Out of Memory Errors

- Reduce `max_image_size` in config
- Reduce `num_inference_steps`
- Enable attention slicing (already enabled)
- Process one image at a time

### 10.3 Slow Processing

- Verify GPU is being used: `nvidia-smi`
- Check if models are loaded: Check logs for "✓" messages
- Increase batch size if processing multiple images
- Use smaller image sizes for faster processing

### 10.4 Health Check Failing

```bash
# Check if API is responding
curl http://localhost:8000/health

# Check container status
docker ps | grep face-body-swap

# Restart if needed
docker-compose restart face-body-swap
```

### 10.5 Model Download Issues

If models fail to download:
- Check internet connection in container
- Manually download models to `models/` directory
- Use HuggingFace cache if available

## Step 11: Backup and Recovery

### 11.1 Backup Important Data

```bash
# Backup outputs
tar -czf outputs_backup.tar.gz outputs/

# Backup models (if custom)
tar -czf models_backup.tar.gz models/

# Backup configuration
cp configs/production.yaml configs/production.yaml.backup
```

### 11.2 Restore from Backup

```bash
# Restore outputs
tar -xzf outputs_backup.tar.gz

# Restore models
tar -xzf models_backup.tar.gz
```

## Step 12: Scaling (Optional)

### 12.1 Multiple GPU Instances

If you have multiple GPUs:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all  # Use all available GPUs
          capabilities: [gpu]
```

### 12.2 Load Balancing

For high traffic, use a load balancer:

```nginx
upstream face_swap_backend {
    server localhost:8000;
    # Add more instances if needed
}

server {
    listen 80;
    location / {
        proxy_pass http://face_swap_backend;
    }
}
```

## Quick Reference Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Check health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics

# Test API
curl http://localhost:8000/api/v1/templates

# Execute command in container
docker exec -it face-body-swap-api bash

# Rebuild after code changes
docker-compose up -d --build
```

## Support

For issues specific to Runninghub:
1. Check Runninghub documentation
2. Review container logs: `docker-compose logs`
3. Check health endpoint: `/health`
4. Review metrics: `/metrics`

For pipeline-specific issues:
1. Check `TROUBLESHOOTING_GUIDE.md`
2. Review `WORKFLOW_DOCUMENTATION.md`
3. Check logs in `logs/app.log`

## Next Steps

After successful deployment:
1. ✅ Run test set: `python scripts/generate_test_set.py`
2. ✅ Test with sample images
3. ✅ Configure LoRA models if needed
4. ✅ Set up monitoring alerts
5. ✅ Document your specific configuration

---

**Deployment Status**: ✅ Ready for Production

**Last Updated**: Current Date
**Tested On**: Runninghub GPU Instance

