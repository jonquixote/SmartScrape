# SmartScrape Production Deployment Guide

This guide outlines the steps to deploy SmartScrape in a production environment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Security Considerations](#security-considerations)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Deployment Options](#deployment-options)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)
- [Backup and Disaster Recovery](#backup-and-disaster-recovery)

## Prerequisites

Before deploying SmartScrape to production, ensure you have the following:

- Docker and Docker Compose (v1.27.0+)
- Access to a secure server with at least 4GB RAM and 2 CPUs
- Domain name (optional, but recommended)
- SSL certificates (can be generated with Let's Encrypt)
- API keys for required services (Google API key for Gemini, etc.)
- Sufficient disk space (at least 20GB recommended)

## Security Considerations

Security is a priority in production environments. Take these measures:

1. **Environment Variables**: Never commit `.env.production` to version control.
2. **API Keys**: Use appropriate permissions and rotate regularly.
3. **Secrets Management**: Consider a secrets manager for production environments.
4. **Network Security**: Use HTTPS, secure headers, and firewall rules.
5. **Access Control**: Implement proper authentication and authorization.
6. **Data Protection**: Encrypt sensitive data at rest and in transit.

## Environment Setup

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/smartscrape.git
   cd smartscrape
   ```

2. **Run Setup Script**:
   ```bash
   ./setup_environment.sh prod
   ```

3. **Edit Production Environment File**:
   Open `.env.production` and update all required values including:
   - API keys
   - Database connection strings
   - Secret key
   - Rate limiting settings
   - Proxy configuration (if applicable)

## Configuration

### 1. Enable Security Features

Ensure the following settings are configured in `.env.production`:

```
SECRET_KEY=your_secure_secret_key
CORS_ORIGINS=https://yourdomain.com
RATE_LIMIT_REQUESTS_PER_MINUTE=60
API_KEY_HEADER=X-API-Key
```

### 2. Configure Nginx SSL (if using the provided Nginx)

1. Place your SSL certificates in `nginx/ssl/`:
   - `server.crt`: SSL certificate
   - `server.key`: Private key

2. Update the domain name in `docker/nginx/nginx.conf` if needed.

### 3. Configure Database (if using external database)

Set the appropriate `DATABASE_URL` in `.env.production`:

```
# PostgreSQL example
DATABASE_URL=postgresql://username:password@postgres-host:5432/smartscrape

# MySQL example
DATABASE_URL=mysql://username:password@mysql-host:3306/smartscrape
```

### 4. Configure Monitoring

For Sentry integration, add your DSN:

```
SENTRY_DSN=https://your-sentry-dsn
METRICS_ENABLED=true
```

## Deployment Options

### Docker Compose (Recommended for Most Deployments)

1. Deploy the application:
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

2. Verify the deployment:
   ```bash
   docker-compose ps
   curl http://localhost:8001/health
   ```

### Kubernetes Deployment (For Large-Scale Deployments)

For large-scale deployments, use the Kubernetes manifests in the `k8s/` directory.

1. Apply the Kubernetes configurations:
   ```bash
   kubectl apply -f k8s/namespace.yaml
   kubectl apply -f k8s/configmap.yaml
   kubectl apply -f k8s/secret.yaml
   kubectl apply -f k8s/deployment.yaml
   kubectl apply -f k8s/service.yaml
   kubectl apply -f k8s/ingress.yaml
   ```

2. Verify the deployment:
   ```bash
   kubectl get pods -n smartscrape
   ```

## Monitoring and Maintenance

### Health Checks

The application provides a health endpoint at `/health` that returns detailed status information.

### Metrics

Prometheus metrics are available at `/metrics`. These can be integrated with monitoring systems like Grafana.

### Logs

Logs are written to the `logs/` directory inside the container and mounted to the host. For Docker Compose deployments, view logs with:

```bash
docker-compose logs -f app
```

### Resource Monitoring

Monitor system resources using:

```bash
docker stats
```

## Scaling

### Horizontal Scaling

For higher throughput, you can scale horizontally:

```bash
docker-compose up -d --scale app=3
```

Note: This requires a load balancer to distribute traffic.

### Vertical Scaling

For handling larger scraping operations, increase resource limits in `docker-compose.yml`:

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

## Troubleshooting

### Common Issues

1. **Application fails to start**:
   - Check logs: `docker-compose logs app`
   - Verify environment variables
   - Ensure all required services are running

2. **Memory issues**:
   - Adjust `MAX_MEMORY_MB` in `.env.production`
   - Increase Docker container memory limits

3. **Rate limiting errors**:
   - Check `RATE_LIMIT_REQUESTS_PER_MINUTE` configuration
   - Implement a proxy rotation strategy if needed

### Debugging

Enable debug logs temporarily in production:

```bash
docker-compose exec app python -c "import logging; logging.getLogger().setLevel(logging.DEBUG)"
```

## Backup and Disaster Recovery

### Database Backups

If using an external database, configure regular backups:

```bash
# PostgreSQL example
docker-compose exec db pg_dump -U username dbname > backup.sql
```

### Configuration Backups

Regularly backup your configuration files:

```bash
tar -czf smartscrape-config-backup.tar.gz .env.production docker/ monitoring/
```

### Recovery Plan

1. Restore configuration from backups
2. Rebuild and redeploy containers
3. Restore database if applicable
4. Verify system health with `/health` endpoint

## Support

For production support, contact the SmartScrape team or open an issue on GitHub.
