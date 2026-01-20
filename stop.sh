#!/bin/bash
# Linux stop script for Face-Body Swap API

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Face-Body Swap API - Stopping${NC}"
echo -e "${GREEN}========================================${NC}"

# Use docker compose (newer) or docker-compose (older)
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Stop services
echo -e "${YELLOW}Stopping Docker containers...${NC}"
$DOCKER_COMPOSE down

echo -e "${GREEN}✓ Services stopped${NC}"

