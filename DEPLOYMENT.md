# Cloud Run Deployment Guide

This guide provides the exact commands to deploy the Foundation Color Matcher application to Google Cloud Run from source.

## Prerequisites

1. **Google Cloud CLI installed and authenticated**
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable required APIs**
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   ```

## Deployment Commands

### Basic Deployment
```bash
gcloud run deploy foundation-matcher \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars SECRET_KEY=$(openssl rand -base64 32)
```

### Advanced Deployment with Custom Configuration
```bash
gcloud run deploy foundation-matcher \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0 \
  --concurrency 80 \
  --set-env-vars SECRET_KEY=$(openssl rand -base64 32),FLASK_ENV=production \
  --tag latest
```

### Deployment to Specific Region
```bash
# For better performance, choose a region close to your users
# Available regions: us-central1, us-east1, us-west1, europe-west1, asia-east1, etc.

gcloud run deploy foundation-matcher \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars SECRET_KEY=$(openssl rand -base64 32)
```

## Configuration Options

### Memory and CPU Settings
- **Memory**: 2Gi (recommended for OpenCV processing)
- **CPU**: 2 (recommended for image analysis)
- **Timeout**: 300 seconds (5 minutes for complex image processing)

### Scaling Settings
- **Max Instances**: 10 (adjust based on expected traffic)
- **Min Instances**: 0 (cost-effective, allows scaling to zero)
- **Concurrency**: 80 (number of concurrent requests per instance)

### Environment Variables
- **SECRET_KEY**: Auto-generated secure key for Flask sessions
- **FLASK_ENV**: Set to 'production' for production deployment

## Post-Deployment

### Get Service URL
```bash
gcloud run services describe foundation-matcher \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

### View Logs
```bash
gcloud run logs tail foundation-matcher \
  --platform managed \
  --region us-central1
```

### Update Deployment
```bash
# To update the application, simply run the deploy command again
gcloud run deploy foundation-matcher \
  --source . \
  --platform managed \
  --region us-central1
```

## Security Considerations

### Custom Domain (Optional)
```bash
# Map custom domain
gcloud run domain-mappings create \
  --service foundation-matcher \
  --domain your-domain.com \
  --region us-central1
```

### IAM Permissions
```bash
# Make service public (already included with --allow-unauthenticated)
gcloud run services add-iam-policy-binding foundation-matcher \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region us-central1
```

## Monitoring and Maintenance

### View Service Details
```bash
gcloud run services describe foundation-matcher \
  --platform managed \
  --region us-central1
```

### Delete Service (if needed)
```bash
gcloud run services delete foundation-matcher \
  --platform managed \
  --region us-central1
```

## Troubleshooting

### Common Issues

1. **Build Timeout**: If the build takes too long, increase the timeout:
   ```bash
   gcloud config set builds/timeout 1200  # 20 minutes
   ```

2. **Memory Issues**: If you encounter memory errors, increase memory:
   ```bash
   --memory 4Gi
   ```

3. **Cold Start Issues**: Set minimum instances to reduce cold starts:
   ```bash
   --min-instances 1
   ```

### View Build Logs
```bash
gcloud builds list --limit 5
gcloud builds log BUILD_ID
```

## Cost Optimization

- Use `--min-instances 0` to scale to zero when not in use
- Set appropriate `--max-instances` based on expected traffic
- Monitor usage with Cloud Monitoring
- Consider using `--cpu-throttling` for cost savings if performance allows

## Files Created for Deployment

The following files have been created to support Cloud Run deployment:

- `Dockerfile` - Container configuration optimized for Cloud Run
- `requirements.txt` - Updated with gunicorn and opencv-python-headless
- `.dockerignore` - Excludes unnecessary files from container
- `.gcloudignore` - Excludes files from source deployment
- `app.py` - Updated with production-ready configuration

## Example Complete Deployment

```bash
# Set your project
gcloud config set project your-project-id

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Deploy
gcloud run deploy foundation-matcher \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --set-env-vars SECRET_KEY=$(openssl rand -base64 32)

# Get the URL
gcloud run services describe foundation-matcher \
  --platform managed \
  --region us-central1 \
  --format 'value(status.url)'
```

Your Foundation Color Matcher application will be deployed and accessible via the provided URL!
