# Natura Foundation Matcher API Documentation

## Overview

The Natura Foundation Matcher API provides programmatic access to skin color analysis and foundation matching using advanced computer vision and color science. The API is designed for easy integration with Google ADK, GenAI, and other services.

**Base URL**: `http://localhost:9090/api/v1`

## Key Features

- **No authentication required** - Simple integration
- **Binary image upload** - Efficient for large images
- **Rate limiting** - 60 requests/hour per IP
- **CORS enabled** - Can be called from web applications
- **Fast processing** - Typically < 2 seconds per image

## Quick Start

### 1. Check API Health

```bash
curl http://localhost:9090/api/v1/health
```

### 2. Analyze an Image

```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -F "image=@selfie.jpg"
```

## API Endpoints

### Health Check

Check if the API is running and healthy.

**Endpoint**: `GET /api/v1/health`

**Example**:
```bash
curl http://localhost:9090/api/v1/health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0",
  "timestamp": "2025-01-17T19:30:00Z"
}
```

### API Status

Get API capabilities and configuration.

**Endpoint**: `GET /api/v1/status`

**Example**:
```bash
curl http://localhost:9090/api/v1/status
```

**Response**:
```json
{
  "api_version": "1.0",
  "capabilities": {
    "brands": ["Natura"],
    "max_file_size_mb": 10,
    "supported_formats": ["jpg", "jpeg", "png", "webp"],
    "rate_limit": "60 requests/hour",
    "min_image_dimension": 200,
    "max_image_dimension": 4000
  },
  "features": {
    "skin_detection": true,
    "undertone_analysis": true,
    "foundation_matching": true,
    "color_calibration": false
  }
}
```

### Analyze Image

Analyze skin color and get foundation recommendations.

**Endpoint**: `POST /api/v1/analyze`

**Parameters**:
- `image` (required): Image file (binary) or base64 string
- `num_matches` (optional): Number of matches to return (1-20, default: 5)
- `include_statistics` (optional): Include detailed statistics (default: false)

#### Method 1: Binary Upload (Recommended)

**Basic Example**:
```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -F "image=@photo.jpg"
```

**With Options**:
```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -F "image=@photo.jpg" \
  -F "num_matches=10" \
  -F "include_statistics=true"
```

**Save Response to File**:
```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -F "image=@photo.jpg" \
  -o result.json
```

**Pretty Print with jq**:
```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -F "image=@photo.jpg" \
  | jq '.'
```

#### Method 2: Base64 Upload

**Prepare Base64**:
```bash
# Convert image to base64
base64 -i photo.jpg -o photo.base64

# Create JSON payload
cat > request.json << EOF
{
  "image": "$(cat photo.base64)",
  "num_matches": 5
}
EOF
```

**Send Request**:
```bash
curl -X POST http://localhost:9090/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d @request.json
```

### Success Response

```json
{
  "success": true,
  "data": {
    "skin_analysis": {
      "color": {
        "lab": {
          "L": 65.4,
          "a": 12.3,
          "b": 18.7
        },
        "rgb": {
          "r": 185,
          "g": 143,
          "b": 112
        },
        "hex": "#B98F70"
      },
      "pixels_analyzed": 125840,
      "undertone": {
        "primary": "Warm",
        "confidence": 85.5
      }
    },
    "matches": [
      {
        "rank": 1,
        "brand": "Natura",
        "product_line": "Una Base Fluida HD FPS 15",
        "shade": "25q",
        "shade_name": "Médio claro com subtom quente",
        "delta_e": 1.45,
        "match_percentage": 92.75,
        "match_quality": "excellent"
      },
      {
        "rank": 2,
        "brand": "Natura",
        "product_line": "Una Base Cremosa FPS 25",
        "shade": "23f",
        "shade_name": "Médio claro com subtom frio",
        "delta_e": 2.12,
        "match_percentage": 89.4,
        "match_quality": "very_good"
      }
    ],
    "processing_time_ms": 1250
  }
}
```

### Error Responses

#### No Skin Detected
```json
{
  "success": false,
  "error": {
    "code": "NO_SKIN_DETECTED",
    "message": "No skin detected in the image",
    "details": "Please ensure good lighting and clear face visibility",
    "suggestions": [
      "Use natural lighting",
      "Ensure face is clearly visible",
      "Avoid shadows on face",
      "Remove glasses if possible"
    ]
  }
}
```

#### Invalid Image
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Image format not supported. Allowed: JPEG, PNG, WebP",
    "details": "Please check image format, size, and dimensions"
  }
}
```

#### Rate Limit Exceeded
```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Maximum 60 requests per hour.",
    "retry_after_seconds": 1800
  }
}
```

## Integration Examples

### Python

```python
import requests
import json

def analyze_skin_color(image_path):
    url = "http://localhost:9090/api/v1/analyze"
    
    with open(image_path, 'rb') as f:
        files = {'image': (image_path, f, 'image/jpeg')}
        data = {'num_matches': 5}
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            return result['data']
        else:
            print(f"Error: {result['error']['message']}")
    else:
        print(f"HTTP Error: {response.status_code}")
    
    return None

# Usage
result = analyze_skin_color('selfie.jpg')
if result:
    print(f"Skin color: {result['skin_analysis']['color']['hex']}")
    print(f"Best match: {result['matches'][0]['shade_name']}")
```

### Python with Google Services

```python
import requests
from google.cloud import storage
import io

def analyze_from_gcs(bucket_name, blob_name):
    # Download from Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    image_bytes = blob.download_as_bytes()
    
    # Send to API
    url = "http://localhost:9090/api/v1/analyze"
    files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
    
    response = requests.post(url, files=files)
    return response.json()
```

### JavaScript/Node.js

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function analyzeSkinColor(imagePath) {
    const form = new FormData();
    form.append('image', fs.createReadStream(imagePath));
    form.append('num_matches', '5');
    
    try {
        const response = await axios.post(
            'http://localhost:9090/api/v1/analyze',
            form,
            {
                headers: form.getHeaders()
            }
        );
        
        if (response.data.success) {
            return response.data.data;
        }
    } catch (error) {
        console.error('Error:', error.response?.data || error.message);
    }
}

// Usage
analyzeSkinColor('selfie.jpg').then(result => {
    if (result) {
        console.log('Skin color:', result.skin_analysis.color.hex);
        console.log('Best match:', result.matches[0].shade_name);
    }
});
```

### cURL with Error Handling

```bash
#!/bin/bash

# Function to analyze image
analyze_image() {
    local image_path=$1
    local output_file="result.json"
    
    # Make API call
    http_code=$(curl -s -o "$output_file" -w "%{http_code}" \
        -X POST http://localhost:9090/api/v1/analyze \
        -F "image=@$image_path")
    
    # Check HTTP status
    if [ "$http_code" -eq 200 ]; then
        # Check if analysis was successful
        if jq -e '.success' "$output_file" > /dev/null; then
            echo "Analysis successful!"
            echo "Skin color: $(jq -r '.data.skin_analysis.color.hex' "$output_file")"
            echo "Best match: $(jq -r '.data.matches[0].shade_name' "$output_file")"
        else
            echo "Analysis failed: $(jq -r '.error.message' "$output_file")"
        fi
    else
        echo "HTTP Error: $http_code"
        cat "$output_file"
    fi
}

# Usage
analyze_image "selfie.jpg"
```

## Rate Limiting

The API implements IP-based rate limiting:
- **Limit**: 60 requests per hour
- **Window**: Rolling 1-hour window
- **Response**: 429 status code when exceeded
- **Retry-After**: Header indicates seconds until next request allowed

Example of handling rate limits:

```python
import time
import requests

def analyze_with_retry(image_path, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(
            'http://localhost:9090/api/v1/analyze',
            files={'image': open(image_path, 'rb')}
        )
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
        else:
            return response.json()
    
    raise Exception("Max retries exceeded")
```

## Image Requirements

### Supported Formats
- JPEG/JPG
- PNG
- WebP

### Size Limits
- **Minimum**: 10KB
- **Maximum**: 10MB
- **Min dimension**: 200x200 pixels
- **Max dimension**: 4000x4000 pixels

### Best Practices
1. Use good lighting (natural light preferred)
2. Ensure face is clearly visible
3. Avoid heavy shadows
4. Remove glasses if possible
5. Use front-facing camera angle

## Error Codes Reference

| Code | Description | Action |
|------|-------------|--------|
| `NO_IMAGE` | No image provided | Include image in request |
| `INVALID_IMAGE` | Image format/size invalid | Check image requirements |
| `NO_SKIN_DETECTED` | No skin found in image | Retry with better photo |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `PROCESSING_ERROR` | Server processing error | Retry request |
| `INTERNAL_ERROR` | Unexpected server error | Contact support |

## Testing

### Basic Test Script

```bash
#!/bin/bash

API_BASE="http://localhost:9090/api/v1"

echo "1. Testing health endpoint..."
curl -s "$API_BASE/health" | jq '.'

echo -e "\n2. Testing status endpoint..."
curl -s "$API_BASE/status" | jq '.'

echo -e "\n3. Testing analyze endpoint..."
# Download test image
curl -s -o test-face.jpg \
  "https://thispersondoesnotexist.com/image"

# Analyze
curl -s -X POST "$API_BASE/analyze" \
  -F "image=@test-face.jpg" \
  | jq '{
      success: .success,
      skin_color: .data.skin_analysis.color.hex,
      undertone: .data.skin_analysis.undertone.primary,
      best_match: .data.matches[0].shade_name,
      match_quality: .data.matches[0].match_quality
    }'

# Cleanup
rm test-face.jpg
```

### Load Testing

```python
import concurrent.futures
import requests
import time

def test_request(i):
    start = time.time()
    response = requests.post(
        'http://localhost:9090/api/v1/analyze',
        files={'image': open('test.jpg', 'rb')}
    )
    duration = time.time() - start
    return {
        'request': i,
        'status': response.status_code,
        'duration': duration
    }

# Run 10 concurrent requests
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(test_request, i) for i in range(10)]
    results = [f.result() for f in futures]
    
    # Print results
    for r in results:
        print(f"Request {r['request']}: {r['status']} in {r['duration']:.2f}s")
```

## Support

For issues or questions:
- Check error messages and suggestions
- Verify image meets requirements
- Test with provided examples
- Review rate limiting if getting 429 errors
