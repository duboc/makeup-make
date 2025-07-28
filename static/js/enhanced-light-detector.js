/**
 * Enhanced Light Detection System
 * Provides sophisticated light analysis for optimal photo capture
 */

class EnhancedLightDetector {
    constructor() {
        this.canvas = document.createElement('canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 128;  // Larger sample for better analysis
        this.canvas.height = 96;
        
        this.faceDetector = null;
        this.initFaceDetection();
        
        // Light analysis state
        this.currentMetrics = {
            brightness: 0,
            contrast: 0,
            uniformity: 0,
            colorTemp: 0,
            quality: 'unknown',
            faceDetected: false
        };
        
        // Thresholds for quality assessment
        this.thresholds = {
            brightness: {
                excellent: { min: 130, max: 180 },
                good: { min: 100, max: 200 },
                fair: { min: 70, max: 220 },
                poor: { min: 0, max: 255 }
            },
            contrast: {
                excellent: { min: 40, max: 80 },
                good: { min: 30, max: 100 },
                fair: { min: 20, max: 120 },
                poor: { min: 0, max: 255 }
            },
            uniformity: {
                excellent: { min: 0.8, max: 1.0 },
                good: { min: 0.6, max: 1.0 },
                fair: { min: 0.4, max: 1.0 },
                poor: { min: 0, max: 1.0 }
            }
        };
    }
    
    async initFaceDetection() {
        // Initialize face detection if available
        if ('FaceDetector' in window) {
            try {
                this.faceDetector = new FaceDetector();
            } catch (e) {
                console.log('Face detection not available:', e);
            }
        }
    }
    
    /**
     * Analyze video frame for light quality
     */
    analyzeFrame(video) {
        if (!video || !video.videoWidth) return this.currentMetrics;
        
        // Draw video frame to canvas
        this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
        const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        const data = imageData.data;
        
        // Calculate metrics
        const brightness = this.calculateBrightness(data);
        const contrast = this.calculateContrast(data);
        const uniformity = this.calculateUniformity(imageData);
        const colorTemp = this.estimateColorTemperature(data);
        
        // Determine overall quality
        const quality = this.assessQuality(brightness, contrast, uniformity);
        
        // Update current metrics
        this.currentMetrics = {
            brightness,
            contrast,
            uniformity,
            colorTemp,
            quality,
            faceDetected: this.currentMetrics.faceDetected
        };
        
        return this.currentMetrics;
    }
    
    /**
     * Calculate average brightness
     */
    calculateBrightness(data) {
        let totalBrightness = 0;
        const pixelCount = data.length / 4;
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            
            // Use perceived brightness formula
            const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
            totalBrightness += brightness;
        }
        
        return Math.round(totalBrightness / pixelCount);
    }
    
    /**
     * Calculate contrast (standard deviation of brightness)
     */
    calculateContrast(data) {
        const brightnesses = [];
        
        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
            brightnesses.push(brightness);
        }
        
        const mean = brightnesses.reduce((a, b) => a + b) / brightnesses.length;
        const variance = brightnesses.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / brightnesses.length;
        
        return Math.round(Math.sqrt(variance));
    }
    
    /**
     * Calculate lighting uniformity across regions
     */
    calculateUniformity(imageData) {
        const regionSize = 16;
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        
        const regionBrightnesses = [];
        
        // Divide image into regions and calculate brightness for each
        for (let y = 0; y < height; y += regionSize) {
            for (let x = 0; x < width; x += regionSize) {
                let regionSum = 0;
                let pixelCount = 0;
                
                for (let ry = 0; ry < regionSize && y + ry < height; ry++) {
                    for (let rx = 0; rx < regionSize && x + rx < width; rx++) {
                        const idx = ((y + ry) * width + (x + rx)) * 4;
                        const brightness = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                        regionSum += brightness;
                        pixelCount++;
                    }
                }
                
                if (pixelCount > 0) {
                    regionBrightnesses.push(regionSum / pixelCount);
                }
            }
        }
        
        // Calculate uniformity as 1 - coefficient of variation
        const mean = regionBrightnesses.reduce((a, b) => a + b) / regionBrightnesses.length;
        const variance = regionBrightnesses.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / regionBrightnesses.length;
        const stdDev = Math.sqrt(variance);
        const coefficientOfVariation = stdDev / mean;
        
        return Math.max(0, Math.min(1, 1 - coefficientOfVariation));
    }
    
    /**
     * Estimate color temperature
     */
    estimateColorTemperature(data) {
        let totalR = 0, totalG = 0, totalB = 0;
        const pixelCount = data.length / 4;
        
        for (let i = 0; i < data.length; i += 4) {
            totalR += data[i];
            totalG += data[i + 1];
            totalB += data[i + 2];
        }
        
        const avgR = totalR / pixelCount;
        const avgG = totalG / pixelCount;
        const avgB = totalB / pixelCount;
        
        // Simple color temperature estimation based on R/B ratio
        const rbRatio = avgR / avgB;
        let colorTemp;
        
        if (rbRatio < 0.7) {
            colorTemp = 'Cool (>6500K)';
        } else if (rbRatio < 1.0) {
            colorTemp = 'Daylight (5000-6500K)';
        } else if (rbRatio < 1.3) {
            colorTemp = 'Neutral (4000-5000K)';
        } else if (rbRatio < 1.6) {
            colorTemp = 'Warm (3000-4000K)';
        } else {
            colorTemp = 'Very Warm (<3000K)';
        }
        
        return colorTemp;
    }
    
    /**
     * Assess overall quality based on metrics
     */
    assessQuality(brightness, contrast, uniformity) {
        let score = 0;
        let factors = 0;
        
        // Brightness score (40% weight)
        if (brightness >= this.thresholds.brightness.excellent.min && 
            brightness <= this.thresholds.brightness.excellent.max) {
            score += 4 * 0.4;
        } else if (brightness >= this.thresholds.brightness.good.min && 
                   brightness <= this.thresholds.brightness.good.max) {
            score += 3 * 0.4;
        } else if (brightness >= this.thresholds.brightness.fair.min && 
                   brightness <= this.thresholds.brightness.fair.max) {
            score += 2 * 0.4;
        } else {
            score += 1 * 0.4;
        }
        
        // Contrast score (30% weight)
        if (contrast >= this.thresholds.contrast.excellent.min && 
            contrast <= this.thresholds.contrast.excellent.max) {
            score += 4 * 0.3;
        } else if (contrast >= this.thresholds.contrast.good.min && 
                   contrast <= this.thresholds.contrast.good.max) {
            score += 3 * 0.3;
        } else if (contrast >= this.thresholds.contrast.fair.min && 
                   contrast <= this.thresholds.contrast.fair.max) {
            score += 2 * 0.3;
        } else {
            score += 1 * 0.3;
        }
        
        // Uniformity score (30% weight)
        if (uniformity >= this.thresholds.uniformity.excellent.min) {
            score += 4 * 0.3;
        } else if (uniformity >= this.thresholds.uniformity.good.min) {
            score += 3 * 0.3;
        } else if (uniformity >= this.thresholds.uniformity.fair.min) {
            score += 2 * 0.3;
        } else {
            score += 1 * 0.3;
        }
        
        // Map score to quality
        if (score >= 3.5) return 'excellent';
        if (score >= 2.5) return 'good';
        if (score >= 1.5) return 'fair';
        return 'poor';
    }
    
    /**
     * Get recommendations based on current metrics
     */
    getRecommendations() {
        const recommendations = [];
        const metrics = this.currentMetrics;
        
        // Brightness recommendations
        if (metrics.brightness < this.thresholds.brightness.good.min) {
            recommendations.push({
                type: 'warning',
                icon: 'fa-lightbulb',
                text: 'Aumente a iluminação - mova-se para perto de uma janela ou ligue mais luzes'
            });
        } else if (metrics.brightness > this.thresholds.brightness.good.max) {
            recommendations.push({
                type: 'warning',
                icon: 'fa-sun',
                text: 'Iluminação muito forte - evite luz direta no rosto'
            });
        }
        
        // Contrast recommendations
        if (metrics.contrast < this.thresholds.contrast.fair.min) {
            recommendations.push({
                type: 'info',
                icon: 'fa-adjust',
                text: 'Iluminação muito uniforme - adicione luz lateral suave'
            });
        } else if (metrics.contrast > this.thresholds.contrast.good.max) {
            recommendations.push({
                type: 'warning',
                icon: 'fa-exclamation-triangle',
                text: 'Sombras muito fortes - use luz mais difusa'
            });
        }
        
        // Uniformity recommendations
        if (metrics.uniformity < this.thresholds.uniformity.fair.min) {
            recommendations.push({
                type: 'warning',
                icon: 'fa-th',
                text: 'Iluminação desigual - evite sombras no rosto'
            });
        }
        
        // Color temperature recommendations
        if (metrics.colorTemp.includes('Very Warm') || metrics.colorTemp.includes('Warm')) {
            recommendations.push({
                type: 'info',
                icon: 'fa-temperature-low',
                text: 'Luz amarelada detectada - prefira luz natural ou branca'
            });
        }
        
        // Perfect conditions
        if (recommendations.length === 0) {
            recommendations.push({
                type: 'success',
                icon: 'fa-check',
                text: 'Condições de iluminação ideais!'
            });
        }
        
        return recommendations;
    }
    
    /**
     * Detect face in frame
     */
    async detectFace(video) {
        if (!this.faceDetector || !video || !video.videoWidth) {
            return false;
        }
        
        try {
            const faces = await this.faceDetector.detect(video);
            this.currentMetrics.faceDetected = faces.length > 0;
            return faces.length > 0;
        } catch (e) {
            // Fallback: assume face is present if we can't detect
            this.currentMetrics.faceDetected = true;
            return true;
        }
    }
    
    /**
     * Update UI with current metrics
     */
    updateUI() {
        const metrics = this.currentMetrics;
        
        // Update light meter
        const percentage = Math.min(100, (metrics.brightness / 200) * 100);
        const lightFill = document.querySelector('.light-meter-fill');
        if (lightFill) {
            lightFill.style.width = `${percentage}%`;
        }
        
        // Update quality badge
        const qualityBadge = document.querySelector('.light-quality-badge');
        if (qualityBadge) {
            qualityBadge.className = `light-quality-badge ${metrics.quality}`;
            const qualityText = {
                'excellent': 'Excelente',
                'good': 'Boa',
                'fair': 'Razoável',
                'poor': 'Fraca'
            };
            qualityBadge.textContent = qualityText[metrics.quality] || 'Analisando...';
        }
        
        // Update metrics
        this.updateMetricValue('brightness-value', metrics.brightness);
        this.updateMetricValue('contrast-value', `${metrics.contrast}`);
        this.updateMetricValue('uniformity-value', `${Math.round(metrics.uniformity * 100)}%`);
        
        // Update recommendations
        const recommendations = this.getRecommendations();
        const recommendationsContainer = document.querySelector('.light-recommendations');
        if (recommendationsContainer) {
            const hasWarnings = recommendations.some(r => r.type === 'warning');
            recommendationsContainer.className = `light-recommendations ${hasWarnings ? 'warning' : ''}`;
            
            recommendationsContainer.innerHTML = recommendations.map(rec => `
                <div class="light-recommendation-item">
                    <i class="fas ${rec.icon}"></i>
                    <span>${rec.text}</span>
                </div>
            `).join('');
        }
        
        // Update face detection status
        const faceStatus = document.querySelector('.face-detection-status');
        if (faceStatus) {
            if (metrics.faceDetected) {
                faceStatus.className = 'face-detection-status detected';
                faceStatus.innerHTML = '<div class="face-detection-icon"></div> Rosto detectado';
            } else {
                faceStatus.className = 'face-detection-status not-detected';
                faceStatus.innerHTML = '<div class="face-detection-icon"></div> Posicione o rosto';
            }
        }
        
        // Update capture button state
        const captureBtn = document.querySelector('.capture-btn');
        if (captureBtn) {
            if (metrics.quality === 'poor') {
                captureBtn.classList.add('warning');
            } else {
                captureBtn.classList.remove('warning');
            }
        }
    }
    
    updateMetricValue(id, value) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    }
}

// Export for use in main application
window.EnhancedLightDetector = EnhancedLightDetector;
