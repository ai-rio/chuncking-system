# NextJS-FastAPI Integration Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Cloud Platform Deployments](#cloud-platform-deployments)
6. [Environment Configuration](#environment-configuration)
7. [Database Migration](#database-migration)
8. [Monitoring Setup](#monitoring-setup)
9. [Backup & Recovery](#backup--recovery)
10. [Troubleshooting](#troubleshooting)

---

## 1. Overview

This guide provides step-by-step instructions for deploying the NextJS-FastAPI integrated chunking system across different environments, from local development to production cloud deployments.

### Architecture Overview

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[Nginx/ALB]
    end
    
    subgraph "Frontend Tier"
        FE1[NextJS App 1]
        FE2[NextJS App 2]
    end
    
    subgraph "Backend Tier"
        BE1[FastAPI App 1]
        BE2[FastAPI App 2]
    end
    
    subgraph "Worker Tier"
        W1[Celery Worker 1]
        W2[Celery Worker 2]
    end
    
    subgraph "Data Tier"
        DB[(PostgreSQL)]
        REDIS[(Redis)]
        S3[(File Storage)]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        LOGS[ELK Stack]
    end
    
    LB --> FE1
    LB --> FE2
    LB --> BE1
    LB --> BE2
    
    BE1 --> DB
    BE2 --> DB
    BE1 --> REDIS
    BE2 --> REDIS
    BE1 --> S3
    BE2 --> S3
    
    W1 --> DB
    W2 --> DB
    W1 --> REDIS
    W2 --> REDIS
    W1 --> S3
    W2 --> S3
    
    BE1 --> PROM
    BE2 --> PROM
    PROM --> GRAF
    BE1 --> LOGS
    BE2 --> LOGS
```

---

## 2. Prerequisites

### System Requirements

- **CPU**: Minimum 4 cores (8 cores recommended for production)
- **RAM**: Minimum 8GB (16GB+ recommended for production)
- **Storage**: Minimum 50GB SSD (100GB+ recommended for production)
- **Network**: Stable internet connection with adequate bandwidth

### Software Dependencies

```bash
# Required software
- Docker 24.0+
- Docker Compose 2.0+
- Node.js 18+
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Nginx (for production)
```

### Development Tools

```bash
# Install development dependencies
npm install -g yarn pnpm
pip install poetry

# Optional but recommended
apt-get install -y curl wget git vim htop
```

---

## 3. Local Development Setup

### 3.1 Quick Start with Docker Compose

```bash
# Clone the repository
git clone <repository-url>
cd chunking-system

# Create environment file
cp .env.example .env

# Edit environment variables
vim .env

# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3.2 Manual Development Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run database migrations
alembic upgrade head

# Start FastAPI development server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
yarn install

# Start development server
yarn dev
```

#### Worker Setup

```bash
# In a new terminal, navigate to backend
cd backend
poetry shell

# Start Celery worker
celery -A src.worker.celery_app worker --loglevel=info

# Start Celery beat (for scheduled tasks)
celery -A src.worker.celery_app beat --loglevel=info
```

### 3.3 Development Environment Variables

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chunking_system_dev
REDIS_URL=redis://localhost:6379/0

# API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Security
JWT_SECRET_KEY=your_jwt_secret_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=30

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# File Upload
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=100MB

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
```

---

## 4. Production Deployment

### 4.1 Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
    networks:
      - app-network

  # PostgreSQL with replication
  postgres-primary:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_REPLICATION_USER: ${POSTGRES_REPLICATION_USER}
      POSTGRES_REPLICATION_PASSWORD: ${POSTGRES_REPLICATION_PASSWORD}
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
      - ./postgres/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./postgres/pg_hba.conf:/etc/postgresql/pg_hba.conf
      - ./postgres/init-replication.sql:/docker-entrypoint-initdb.d/init-replication.sql
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    restart: unless-stopped
    networks:
      - app-network

  # Redis Cluster
  redis-master:
    image: redis:7-alpine
    command: redis-server --appendonly yes --replica-read-only no
    volumes:
      - redis_master_data:/data
    restart: unless-stopped
    networks:
      - app-network

  redis-replica:
    image: redis:7-alpine
    command: redis-server --appendonly yes --replicaof redis-master 6379
    volumes:
      - redis_replica_data:/data
    depends_on:
      - redis-master
    restart: unless-stopped
    networks:
      - app-network

  # FastAPI Backend (multiple instances)
  backend:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
      target: production
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-primary:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis-master:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - app_uploads:/app/uploads
    depends_on:
      - postgres-primary
      - redis-master
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped
    networks:
      - app-network

  # Celery Workers
  worker:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
      target: production
    command: poetry run celery -A src.worker.celery_app worker --loglevel=info --concurrency=4
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-primary:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis-master:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - app_uploads:/app/uploads
    depends_on:
      - postgres-primary
      - redis-master
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    restart: unless-stopped
    networks:
      - app-network

  # Celery Beat Scheduler
  beat:
    build:
      context: .
      dockerfile: Dockerfile.fastapi
      target: production
    command: poetry run celery -A src.worker.celery_app beat --loglevel=info
    environment:
      - ENVIRONMENT=production
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres-primary:5432/${POSTGRES_DB}
      - REDIS_URL=redis://redis-master:6379/0
    depends_on:
      - postgres-primary
      - redis-master
    restart: unless-stopped
    networks:
      - app-network

  # Next.js Frontend
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.nextjs
      target: production
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://your-domain.com
      - NEXT_PUBLIC_WS_URL=wss://your-domain.com
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '0.5'
          memory: 1G
        reservations:
          cpus: '0.25'
          memory: 512M
    restart: unless-stopped
    networks:
      - app-network

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - app-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    restart: unless-stopped
    networks:
      - app-network

volumes:
  postgres_primary_data:
  redis_master_data:
  redis_replica_data:
  app_uploads:
  prometheus_data:
  grafana_data:

networks:
  app-network:
    driver: bridge
```

### 4.2 Production Environment Variables

```bash
# .env.production
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=info

# Database
POSTGRES_DB=chunking_system_prod
POSTGRES_USER=chunking_user
POSTGRES_PASSWORD=secure_password_here
POSTGRES_REPLICATION_USER=replicator
POSTGRES_REPLICATION_PASSWORD=replication_password_here

# Redis
REDIS_PASSWORD=redis_password_here

# API Keys (use secure key management)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Security
JWT_SECRET_KEY=very_secure_jwt_secret_key_here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# SSL/TLS
SSL_CERT_PATH=/etc/nginx/ssl/cert.pem
SSL_KEY_PATH=/etc/nginx/ssl/key.pem

# Monitoring
GRAFANA_PASSWORD=secure_grafana_password
PROMETHEUS_RETENTION=30d

# Backup
BACKUP_S3_BUCKET=chunking-system-backups
BACKUP_SCHEDULE=0 2 * * *

# Performance
WORKER_CONCURRENCY=4
MAX_CONNECTIONS=100
POOL_SIZE=20
```

### 4.3 Production Deployment Script

```bash
#!/bin/bash
# deploy-production.sh

set -e

echo "Starting production deployment..."

# Backup current deployment
echo "Creating backup..."
docker-compose -f docker-compose.prod.yml exec postgres-primary pg_dump -U $POSTGRES_USER $POSTGRES_DB > backup_$(date +%Y%m%d_%H%M%S).sql

# Pull latest images
echo "Pulling latest images..."
docker-compose -f docker-compose.prod.yml pull

# Build new images
echo "Building new images..."
docker-compose -f docker-compose.prod.yml build --no-cache

# Run database migrations
echo "Running database migrations..."
docker-compose -f docker-compose.prod.yml run --rm backend poetry run alembic upgrade head

# Deploy with zero downtime
echo "Deploying services..."
docker-compose -f docker-compose.prod.yml up -d --remove-orphans

# Health check
echo "Performing health checks..."
sleep 30

for i in {1..10}; do
    if curl -f http://localhost/api/health; then
        echo "Health check passed!"
        break
    else
        echo "Health check failed, attempt $i/10"
        sleep 10
    fi
done

# Cleanup old images
echo "Cleaning up old images..."
docker image prune -f

echo "Deployment completed successfully!"
```

---

## 5. Cloud Platform Deployments

### 5.1 AWS Deployment

#### ECS Fargate Deployment

```yaml
# aws-ecs-task-definition.json
{
  "family": "chunking-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/chunking-system-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/db"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chunking-system",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    },
    {
      "name": "frontend",
      "image": "your-account.dkr.ecr.region.amazonaws.com/chunking-system-frontend:latest",
      "portMappings": [
        {
          "containerPort": 3000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NODE_ENV",
          "value": "production"
        },
        {
          "name": "NEXT_PUBLIC_API_URL",
          "value": "https://api.your-domain.com"
        }
      ]
    }
  ]
}
```

#### Terraform Infrastructure

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "chunking-system-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = var.environment
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "postgres" {
  identifier = "chunking-system-db"
  
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "chunking-system-final-snapshot"
  
  tags = {
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "main" {
  name       = "chunking-system-cache-subnet"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "chunking-system-redis"
  description                = "Redis cluster for chunking system"
  
  node_type            = "cache.t3.micro"
  port                 = 6379
  parameter_group_name = "default.redis7"
  
  num_cache_clusters = 2
  
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  security_group_ids = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  
  tags = {
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "chunking-system"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "chunking-system-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  tags = {
    Environment = var.environment
  }
}
```

### 5.2 Google Cloud Platform Deployment

#### GKE Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chunking-system-backend
  labels:
    app: chunking-system
    component: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chunking-system
      component: backend
  template:
    metadata:
      labels:
        app: chunking-system
        component: backend
    spec:
      containers:
      - name: backend
        image: gcr.io/your-project/chunking-system-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: chunking-system-secrets
              key: database-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: chunking-system-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: chunking-system-backend-service
spec:
  selector:
    app: chunking-system
    component: backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### 5.3 Azure Deployment

#### Azure Container Instances

```yaml
# azure-container-group.yaml
apiVersion: 2021-03-01
location: eastus
name: chunking-system
properties:
  containers:
  - name: backend
    properties:
      image: your-registry.azurecr.io/chunking-system-backend:latest
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 2.0
      ports:
      - port: 8000
        protocol: TCP
      environmentVariables:
      - name: ENVIRONMENT
        value: production
      - name: DATABASE_URL
        secureValue: postgresql://user:pass@server:5432/db
  - name: frontend
    properties:
      image: your-registry.azurecr.io/chunking-system-frontend:latest
      resources:
        requests:
          cpu: 0.5
          memoryInGb: 1.0
      ports:
      - port: 3000
        protocol: TCP
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - protocol: TCP
      port: 80
    - protocol: TCP
      port: 443
    dnsNameLabel: chunking-system
type: Microsoft.ContainerInstance/containerGroups
```

---

## 6. Environment Configuration

### 6.1 Configuration Management

```python
# src/core/config.py
from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class Settings(BaseSettings):
    """Application settings with environment-specific configurations."""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=20, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=30, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(..., env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    
    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, env="JWT_EXPIRE_MINUTES")
    
    # API Keys
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(..., env="ANTHROPIC_API_KEY")
    
    # File Upload
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    allowed_file_types: List[str] = Field(
        default=["application/pdf", "text/markdown", "text/plain"],
        env="ALLOWED_FILE_TYPES"
    )
    
    # Performance
    worker_concurrency: int = Field(default=4, env="WORKER_CONCURRENCY")
    max_connections: int = Field(default=100, env="MAX_CONNECTIONS")
    
    # Monitoring
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # External Services
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    aws_region: Optional[str] = Field(default=None, env="AWS_REGION")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

settings = Settings()
```

### 6.2 Environment-Specific Configurations

```bash
# environments/development.env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=debug
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chunking_system_dev
REDIS_URL=redis://localhost:6379/0
JWT_EXPIRE_MINUTES=60
PROMETHEUS_ENABLED=true

# environments/staging.env
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=info
DATABASE_URL=postgresql://user:pass@staging-db:5432/chunking_system_staging
REDIS_URL=redis://staging-redis:6379/0
JWT_EXPIRE_MINUTES=30
PROMETHEUS_ENABLED=true

# environments/production.env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=warning
DATABASE_URL=postgresql://user:pass@prod-db:5432/chunking_system_prod
REDIS_URL=redis://prod-redis:6379/0
JWT_EXPIRE_MINUTES=15
PROMETHEUS_ENABLED=true
MAX_CONNECTIONS=200
WORKER_CONCURRENCY=8
```

---

## 7. Database Migration

### 7.1 Alembic Migration Setup

```python
# alembic/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from src.core.config import settings
from src.models.base import Base

# Import all models to ensure they're registered
from src.models.document import Document
from src.models.chunk import Chunk
from src.models.user import User

config = context.config

# Override sqlalchemy.url with our DATABASE_URL
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### 7.2 Migration Scripts

```bash
#!/bin/bash
# scripts/migrate.sh

set -e

echo "Running database migrations..."

# Check if database is accessible
echo "Checking database connectivity..."
poetry run python -c "from src.core.database import engine; engine.connect()"

# Run migrations
echo "Applying migrations..."
poetry run alembic upgrade head

# Verify migration
echo "Verifying migration..."
poetry run alembic current

echo "Migration completed successfully!"
```

### 7.3 Data Migration Scripts

```python
# scripts/data_migration.py
"""Data migration script for production deployments."""

import asyncio
from sqlalchemy.orm import Session
from src.core.database import SessionLocal, engine
from src.models.document import Document
from src.models.chunk import Chunk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def migrate_document_metadata():
    """Migrate document metadata to new schema."""
    db = SessionLocal()
    try:
        # Example: Update document metadata format
        documents = db.query(Document).filter(Document.metadata.is_(None)).all()
        
        for doc in documents:
            # Migrate old metadata format to new format
            doc.metadata = {
                "version": "2.0",
                "migrated_at": "2024-01-01T00:00:00Z",
                "legacy_data": doc.legacy_metadata or {}
            }
            
        db.commit()
        logger.info(f"Migrated {len(documents)} documents")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()

async def reindex_chunks():
    """Reindex chunks with new embedding model."""
    db = SessionLocal()
    try:
        chunks = db.query(Chunk).filter(Chunk.embedding.is_(None)).all()
        
        for chunk in chunks:
            # Regenerate embeddings with new model
            # This would call your embedding service
            pass
            
        db.commit()
        logger.info(f"Reindexed {len(chunks)} chunks")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Reindexing failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(migrate_document_metadata())
    asyncio.run(reindex_chunks())
```

---

## 8. Monitoring Setup

### 8.1 Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'chunking-system-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'chunking-system-workers'
    static_configs:
      - targets: ['worker:9091']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
      
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 8.2 Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "Chunking System Overview",
    "tags": ["chunking-system"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Document Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(documents_processed_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Active WebSocket Connections",
        "type": "singlestat",
        "targets": [
          {
            "expr": "websocket_connections_active"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### 8.3 Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
- name: chunking-system-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status_code=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
      
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }} seconds"
      
  - alert: DatabaseConnectionFailure
    expr: up{job="postgres-exporter"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"
      description: "Cannot connect to PostgreSQL database"
      
  - alert: RedisConnectionFailure
    expr: up{job="redis-exporter"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Redis connection failure"
      description: "Cannot connect to Redis cache"
      
  - alert: WorkerQueueBacklog
    expr: celery_queue_length > 100
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Worker queue backlog"
      description: "Celery queue has {{ $value }} pending tasks"
```

---

## 9. Backup & Recovery

### 9.1 Database Backup Strategy

```bash
#!/bin/bash
# scripts/backup-database.sh

set -e

# Configuration
BACKUP_DIR="/backups/database"
S3_BUCKET="chunking-system-backups"
RETENTION_DAYS=30
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="chunking_system_backup_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

echo "Starting database backup..."

# Create database dump
pg_dump $DATABASE_URL > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"
BACKUP_FILE="${BACKUP_FILE}.gz"

echo "Database backup created: $BACKUP_FILE"

# Upload to S3 (if configured)
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading backup to S3..."
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" "s3://$S3_BUCKET/database/"
    echo "Backup uploaded to S3"
fi

# Clean up old backups
echo "Cleaning up old backups..."
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete

# Clean up old S3 backups
if [ ! -z "$S3_BUCKET" ]; then
    aws s3 ls "s3://$S3_BUCKET/database/" | while read -r line; do
        createDate=$(echo $line | awk '{print $1" "$2}')
        createDate=$(date -d "$createDate" +%s)
        olderThan=$(date -d "$RETENTION_DAYS days ago" +%s)
        if [[ $createDate -lt $olderThan ]]; then
            fileName=$(echo $line | awk '{print $4}')
            if [[ $fileName != "" ]]; then
                aws s3 rm "s3://$S3_BUCKET/database/$fileName"
            fi
        fi
    done
fi

echo "Database backup completed successfully!"
```

### 9.2 File Storage Backup

```bash
#!/bin/bash
# scripts/backup-files.sh

set -e

# Configuration
UPLOAD_DIR="./uploads"
BACKUP_DIR="/backups/files"
S3_BUCKET="chunking-system-backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="files_backup_${TIMESTAMP}.tar.gz"

echo "Starting file backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Create compressed archive
tar -czf "$BACKUP_DIR/$BACKUP_FILE" -C "$(dirname $UPLOAD_DIR)" "$(basename $UPLOAD_DIR)"

echo "File backup created: $BACKUP_FILE"

# Upload to S3
if [ ! -z "$S3_BUCKET" ]; then
    echo "Uploading file backup to S3..."
    aws s3 cp "$BACKUP_DIR/$BACKUP_FILE" "s3://$S3_BUCKET/files/"
    echo "File backup uploaded to S3"
fi

echo "File backup completed successfully!"
```

### 9.3 Recovery Procedures

```bash
#!/bin/bash
# scripts/restore-database.sh

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Example: $0 chunking_system_backup_20240101_120000.sql.gz"
    exit 1
fi

BACKUP_FILE=$1
RESTORE_DIR="/tmp/restore"

echo "Starting database restore from $BACKUP_FILE..."

# Create restore directory
mkdir -p $RESTORE_DIR

# Download from S3 if needed
if [[ $BACKUP_FILE == s3://* ]]; then
    echo "Downloading backup from S3..."
    aws s3 cp "$BACKUP_FILE" "$RESTORE_DIR/"
    BACKUP_FILE="$RESTORE_DIR/$(basename $BACKUP_FILE)"
fi

# Decompress if needed
if [[ $BACKUP_FILE == *.gz ]]; then
    echo "Decompressing backup..."
    gunzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE%.gz}"
fi

# Stop application services
echo "Stopping application services..."
docker-compose stop backend worker

# Create new database
echo "Creating new database..."
psql $DATABASE_URL -c "DROP DATABASE IF EXISTS chunking_system_restore;"
psql $DATABASE_URL -c "CREATE DATABASE chunking_system_restore;"

# Restore database
echo "Restoring database..."
psql "${DATABASE_URL/chunking_system/chunking_system_restore}" < "$BACKUP_FILE"

# Switch databases (manual step)
echo "Database restored to chunking_system_restore"
echo "Manual step required: Update DATABASE_URL to point to restored database"
echo "Then restart services with: docker-compose start backend worker"

# Cleanup
rm -rf $RESTORE_DIR

echo "Database restore completed!"
```

---

## 10. Troubleshooting

### 10.1 Common Issues and Solutions

#### Database Connection Issues

```bash
# Check database connectivity
psql $DATABASE_URL -c "SELECT 1;"

# Check database logs
docker-compose logs postgres

# Check connection pool status
docker-compose exec backend python -c "
from src.core.database import engine
print(f'Pool size: {engine.pool.size()}')
print(f'Checked out: {engine.pool.checkedout()}')
print(f'Overflow: {engine.pool.overflow()}')
"
```

#### Redis Connection Issues

```bash
# Test Redis connectivity
redis-cli -u $REDIS_URL ping

# Check Redis memory usage
redis-cli -u $REDIS_URL info memory

# Monitor Redis commands
redis-cli -u $REDIS_URL monitor
```

#### High Memory Usage

```bash
# Check container memory usage
docker stats

# Check Python memory usage
docker-compose exec backend python -c "
import psutil
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Profile memory usage
docker-compose exec backend python -m memory_profiler src/main.py
```

#### Performance Issues

```bash
# Check slow queries
docker-compose exec postgres psql -U $POSTGRES_USER -d $POSTGRES_DB -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/health

# Monitor worker performance
docker-compose exec worker celery -A src.worker.celery_app inspect stats
```

### 10.2 Log Analysis

```bash
# View application logs
docker-compose logs -f backend

# Search for errors
docker-compose logs backend | grep ERROR

# Monitor real-time logs
tail -f /var/log/chunking-system/app.log | jq .

# Analyze log patterns
grep "HTTP" /var/log/chunking-system/app.log | awk '{print $9}' | sort | uniq -c | sort -nr
```

### 10.3 Health Check Scripts

```bash
#!/bin/bash
# scripts/health-check.sh

set -e

echo "Performing comprehensive health check..."

# Check API health
echo "Checking API health..."
if curl -f http://localhost:8000/health; then
    echo "✓ API is healthy"
else
    echo "✗ API health check failed"
    exit 1
fi

# Check database
echo "Checking database..."
if psql $DATABASE_URL -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✓ Database is accessible"
else
    echo "✗ Database connection failed"
    exit 1
fi

# Check Redis
echo "Checking Redis..."
if redis-cli -u $REDIS_URL ping > /dev/null 2>&1; then
    echo "✓ Redis is accessible"
else
    echo "✗ Redis connection failed"
    exit 1
fi

# Check worker status
echo "Checking workers..."
if docker-compose exec -T worker celery -A src.worker.celery_app inspect ping > /dev/null 2>&1; then
    echo "✓ Workers are responding"
else
    echo "✗ Workers are not responding"
    exit 1
fi

# Check disk space
echo "Checking disk space..."
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 90 ]; then
    echo "✓ Disk usage is acceptable ($DISK_USAGE%)"
else
    echo "⚠ Disk usage is high ($DISK_USAGE%)"
fi

# Check memory usage
echo "Checking memory usage..."
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ $MEM_USAGE -lt 90 ]; then
    echo "✓ Memory usage is acceptable ($MEM_USAGE%)"
else
    echo "⚠ Memory usage is high ($MEM_USAGE%)"
fi

echo "Health check completed successfully!"
```

---

## Conclusion

This deployment guide provides comprehensive instructions for deploying the NextJS-FastAPI integrated chunking system across various environments and platforms. Key highlights include:

- **Multi-environment support**: Development, staging, and production configurations
- **Cloud platform compatibility**: AWS, GCP, and Azure deployment options
- **Robust monitoring**: Prometheus, Grafana, and comprehensive alerting
- **Backup & recovery**: Automated backup strategies and recovery procedures
- **Troubleshooting**: Common issues and diagnostic tools

For successful deployment:

1. **Start with local development** to validate the setup
2. **Use staging environment** to test production configurations
3. **Implement monitoring** before going to production
4. **Set up automated backups** immediately after deployment
5. **Document any customizations** specific to your environment

Regular maintenance, monitoring, and updates are essential for maintaining a healthy production system.