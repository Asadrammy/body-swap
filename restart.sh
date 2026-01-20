#!/bin/bash
# Linux restart script for Face-Body Swap API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Face-Body Swap API - Restarting${NC}"
echo -e "${GREEN}========================================${NC}"

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Restart services
echo -e "${YELLOW}Restarting Docker containers...${NC}"
$DOCKER_COMPOSE restart

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Check health
echo -e "${GREEN}Checking service health...${NC}"
if curl -f http://localhost:8000/health &> /dev/null; then
    echo -e "${GREEN}✓ Service is healthy!${NC}"
else
    echo -e "${YELLOW}⚠ Service may still be starting. Check logs with: $DOCKER_COMPOSE logs -f${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}API is running at: http://localhost:8000${NC}"
echo -e "${GREEN}========================================${NC}"

