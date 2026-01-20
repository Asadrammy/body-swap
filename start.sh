#!/bin/bash
# Linux startup script for Face-Body Swap API
# Usage: ./start.sh [--build] [--logs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Face-Body Swap API - Starting${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from env.example...${NC}"
    if [ -f env.example ]; then
        cp env.example .env
        echo -e "${GREEN}Created .env file. Please edit it with your configuration.${NC}"
    else
        echo -e "${RED}Error: env.example not found!${NC}"
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed!${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed!${NC}"
    exit 1
fi

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Check for --build flag
BUILD_FLAG=""
if [[ "$*" == *"--build"* ]]; then
    BUILD_FLAG="--build"
    echo -e "${YELLOW}Building Docker images...${NC}"
fi

# Start services
echo -e "${GREEN}Starting Docker containers...${NC}"
$DOCKER_COMPOSE up -d $BUILD_FLAG

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 5

# Check health
echo -e "${GREEN}Checking service health...${NC}"
MAX_RETRIES=12
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo -e "${GREEN}✓ Service is healthy!${NC}"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo -e "${YELLOW}Waiting for service... (${RETRY_COUNT}/${MAX_RETRIES})${NC}"
            sleep 5
        else
            echo -e "${RED}✗ Service health check failed after ${MAX_RETRIES} attempts${NC}"
            echo -e "${YELLOW}Check logs with: $DOCKER_COMPOSE logs -f${NC}"
            exit 1
        fi
    fi
done

# Show status
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Service Status:${NC}"
echo -e "${GREEN}========================================${NC}"
$DOCKER_COMPOSE ps

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}API is running at: http://localhost:8000${NC}"
echo -e "${GREEN}API Docs: http://localhost:8000/docs${NC}"
echo -e "${GREEN}Health Check: http://localhost:8000/health${NC}"
echo -e "${GREEN}========================================${NC}"

# Show logs if --logs flag is set
if [[ "$*" == *"--logs"* ]]; then
    echo -e "${YELLOW}Showing logs (Ctrl+C to exit)...${NC}"
    $DOCKER_COMPOSE logs -f
fi

