# Deployment Guide

## Frontend + Backend Deployment

### Quick Start (Development)

1. **Start Backend**:
   ```bash
   python -m src.api.main
   ```

2. **Access Frontend**:
   - Open browser: `http://localhost:8000`
   - Frontend is automatically served by FastAPI

### Production Deployment

#### Option 1: Single Server (Recommended for Small Scale)

**Using Docker Compose:**

```bash
docker-compose up -d
```

The frontend is included in the Docker image and served automatically.

#### Option 2: Separate Frontend Hosting

**Frontend (Static Hosting):**
- Deploy `frontend/` folder to Netlify, Vercel, or GitHub Pages
- Update `API_BASE_URL` in `frontend/script.js` to your backend URL

**Backend (VPS/Cloud):**
- Deploy backend to Runninghub, AWS, Azure, etc.
- Ensure CORS is configured correctly
- Set API URL in frontend config

#### Option 3: Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    # Frontend
    location / {
        root /path/to/frontend;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Environment Configuration

**Backend (.env):**
```env
API_HOST=0.0.0.0
API_PORT=8000
DEVICE=cuda  # or cpu
```

**Frontend (api-config.js):**
```javascript
const API_BASE_URL = 'https://your-api-domain.com/api/v1';
```

### Security Considerations

1. **CORS**: Configure allowed origins in production
2. **HTTPS**: Use SSL certificates
3. **Rate Limiting**: Add rate limiting to API
4. **Authentication**: Add user authentication if needed
5. **File Size Limits**: Configure max upload size

### Monitoring

- Check API logs: `logs/app.log`
- Monitor job queue
- Set up health checks: `/health` endpoint
- Track processing times

### Scaling

For high traffic:
- Use load balancer
- Multiple API instances
- Queue system (Redis/RabbitMQ)
- Database for job storage
- CDN for frontend assets

