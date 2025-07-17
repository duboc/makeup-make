# Makeup Make - Technical Architecture

## System Flow Diagram

```mermaid
graph TB
    A[User] -->|Upload Photo| B[Web Interface]
    B --> C{File Validation}
    C -->|Invalid| D[Error Message]
    C -->|Valid| E[Image Storage]
    
    E --> F[Skin Detection Pipeline]
    F --> F1[RGB-H-CbCr Detection]
    F --> F2[YCrCb Range Detection]
    F --> F3[HSV Range Detection]
    F --> F4[Fair Skin Detection]
    
    F1 & F2 & F3 & F4 --> G[Multi-method Voting]
    G --> H[Post-processing]
    H --> I[Skin Mask Output]
    
    I --> J[Color Analysis]
    J --> K[Average Color Extraction]
    K --> L{Calibration Active?}
    
    L -->|Yes| M[Apply Calibration]
    L -->|No| N[Direct Conversion]
    
    M & N --> O[RGB to LAB Conversion]
    O --> P[Undertone Detection]
    P --> Q[Foundation Matching]
    
    Q --> R[Delta E Calculation]
    R --> S[Undertone Bonus]
    S --> T[Ranking & Scoring]
    T --> U[Results Display]
```

## Calibration System Flow

```mermaid
graph LR
    A[ColorChecker Image] --> B[Chart Detection]
    B --> C{Auto Success?}
    C -->|No| D[Manual Corner Selection]
    C -->|Yes| E[Patch Extraction]
    D --> E
    
    E --> F[24 Color Patches]
    F --> G[Reference Comparison]
    G --> H[Calculate Matrices]
    
    H --> I[White Balance Matrix]
    H --> J[Gamma Correction]
    H --> K[Color Transform Matrix]
    
    I & J & K --> L[Calibration Profile]
    L --> M[Save/Load Profile]
```

## Class Diagram

```mermaid
classDiagram
    class FoundationColorPredictor {
        +SkinDetector skin_detector
        +ColorCalibrator color_calibrator
        +dict foundation_database
        +predict_foundation_match()
        +find_best_foundation_matches()
        +detect_undertone()
        +calculate_delta_e()
    }
    
    class SkinDetector {
        +CascadeClassifier face_cascade
        +detect_skin()
        +get_average_skin_color()
        +get_skin_color_statistics()
        -_detect_skin_rgbhcbcr()
        -_detect_skin_ycrcb()
        -_detect_skin_hsv()
        -_detect_fair_skin()
        -_post_process_mask()
    }
    
    class ColorCalibrator {
        +ColorCheckerDetector detector
        +CIEColorConverter converter
        +bool is_calibrated
        +calibrate_from_image()
        +apply_calibration()
        +rgb_to_lab_calibrated()
        +save_calibration()
        +load_calibration()
    }
    
    class ColorCheckerDetector {
        +detect_chart()
        +manual_selection()
        +validate_detection()
        -_extract_corners()
        -_calculate_transform_matrix()
        -_extract_patches()
    }
    
    class CIEColorConverter {
        +string illuminant
        +srgb_to_linear_rgb()
        +linear_rgb_to_srgb()
        +srgb_to_xyz()
        +xyz_to_lab()
        +lab_to_srgb()
        +delta_e_76()
        +delta_e_94()
        +delta_e_2000()
    }
    
    FoundationColorPredictor --> SkinDetector
    FoundationColorPredictor --> ColorCalibrator
    ColorCalibrator --> ColorCheckerDetector
    ColorCalibrator --> CIEColorConverter
```

## Data Flow Sequence Diagram

```mermaid
sequenceDiagram
    participant User
    participant Flask
    participant Session
    participant SkinDetector
    participant ColorAnalysis
    participant FoundationMatcher
    participant Database
    
    User->>Flask: Upload Image
    Flask->>Session: Store Image
    Flask-->>User: Upload Success
    
    User->>Flask: Analyze Request
    Flask->>Session: Retrieve Image
    Flask->>SkinDetector: Detect Skin
    
    SkinDetector->>SkinDetector: RGB-H-CbCr Analysis
    SkinDetector->>SkinDetector: Multi-method Voting
    SkinDetector->>SkinDetector: Post-processing
    SkinDetector-->>Flask: Skin Mask
    
    Flask->>ColorAnalysis: Extract Color
    ColorAnalysis->>ColorAnalysis: Calculate Average
    ColorAnalysis->>ColorAnalysis: Apply Calibration
    ColorAnalysis->>ColorAnalysis: Convert to LAB
    ColorAnalysis-->>Flask: LAB Values
    
    Flask->>FoundationMatcher: Find Matches
    FoundationMatcher->>Database: Get Foundations
    FoundationMatcher->>FoundationMatcher: Calculate Delta E
    FoundationMatcher->>FoundationMatcher: Apply Undertone Bonus
    FoundationMatcher->>FoundationMatcher: Rank Results
    FoundationMatcher-->>Flask: Sorted Matches
    
    Flask->>Session: Store Results
    Flask-->>User: Display Results
```

## API Request Flow

```mermaid
graph TD
    A[API Request] --> B{Rate Limit Check}
    B -->|Exceeded| C[429 Error]
    B -->|OK| D{Content Type}
    
    D -->|multipart/form-data| E[Binary Upload]
    D -->|application/json| F[Base64 Upload]
    
    E & F --> G[Image Validation]
    G -->|Invalid| H[400 Error]
    G -->|Valid| I[Process Image]
    
    I --> J[Skin Detection]
    J -->|No Skin| K[Error Response]
    J -->|Success| L[Color Analysis]
    
    L --> M[Foundation Matching]
    M --> N[Format Response]
    N --> O[200 Success]
```

## Mathematical Processing Pipeline

```
1. Image Input (BGR)
   ↓
2. Color Space Conversions
   ├─> RGB = cv2.cvtColor(BGR, COLOR_BGR2RGB)
   ├─> YCrCb = cv2.cvtColor(BGR, COLOR_BGR2YCrCb)
   └─> HSV = cv2.cvtColor(BGR, COLOR_BGR2HSV)
   ↓
3. Skin Detection Criteria
   ├─> RGB: (R>95) & (G>40) & (B>20) & ...
   ├─> CrCb: Linear boundary conditions
   └─> Hue: (H<25°) | (H>230°)
   ↓
4. Voting System
   skin_mask = (votes >= threshold)
   ↓
5. Color Extraction
   avg_color = mean(image[skin_mask > 0])
   ↓
6. Calibration (Optional)
   calibrated = WB × Gamma × ColorMatrix × RGB
   ↓
7. LAB Conversion
   RGB → Linear RGB → XYZ → LAB
   ↓
8. Undertone Analysis
   warmth = b* / (|a*| + 1)
   ↓
9. Foundation Matching
   ΔE = √[(L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²]
   ↓
10. Results
    score = max(0, 100 - ΔE × 5)
```

## Technology Stack Architecture

```
┌─────────────────────────────────────────────┐
│                Frontend Layer                │
├─────────────────────────────────────────────┤
│ • HTML5 (Jinja2 Templates)                  │
│ • CSS3 (Bootstrap 5)                        │
│ • JavaScript (Vanilla + AJAX)               │
│ • Drag & Drop API                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              Application Layer               │
├─────────────────────────────────────────────┤
│ • Flask Web Framework                        │
│ • Session Management                         │
│ • File Upload Handling                       │
│ • REST API Endpoints                         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│            Processing Layer                  │
├─────────────────────────────────────────────┤
│ • OpenCV (Computer Vision)                   │
│ • NumPy (Numerical Computing)                │
│ • scikit-learn (ML Algorithms)              │
│ • PIL/Pillow (Image Processing)             │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              Data Layer                      │
├─────────────────────────────────────────────┤
│ • In-Memory Foundation Database              │
│ • Session Storage                            │
│ • Temporary File Storage                     │
│ • Calibration Profiles (JSON)               │
└─────────────────────────────────────────────┘
```

## Error Handling Flow

```mermaid
graph TD
    A[Request] --> B{Validation}
    B -->|File Size| C[413 Error: Too Large]
    B -->|File Type| D[400 Error: Invalid Type]
    B -->|No File| E[400 Error: No File]
    
    B -->|Valid| F[Processing]
    F -->|OpenCV Error| G[500 Error: Processing Failed]
    F -->|No Skin| H[400 Error: No Skin Detected]
    F -->|Calibration Error| I[500 Error: Calibration Failed]
    
    F -->|Success| J[200 Response]
    
    C & D & E & G & H & I --> K[Error Response]
    K --> L{Client Type}
    L -->|Web| M[Flash Message + Redirect]
    L -->|API| N[JSON Error Response]
```

## Performance Optimizations

1. **Vectorized Operations**
   - NumPy array operations for pixel processing
   - Batch color space conversions
   - Efficient boolean indexing

2. **Image Processing Pipeline**
   - Single-pass skin detection
   - Cached calibration matrices
   - Pre-computed foundation database

3. **Memory Management**
   - Temporary file cleanup
   - Session data expiration
   - Limited file size (16MB)

4. **API Rate Limiting**
   - 60 requests/hour per IP
   - Thread-safe request tracking
   - Automatic cleanup of old entries

## Security Measures

1. **File Upload Security**
   - Type validation (MIME + extension)
   - Size limits (16MB max)
   - Secure filename generation (UUID)
   - Isolated upload directory

2. **Session Security**
   - HTTPOnly cookies
   - Secure flag (HTTPS only in production)
   - SameSite protection
   - Secret key rotation

3. **API Security**
   - Rate limiting
   - Input validation
   - Error message sanitization
   - CORS headers for API endpoints

## Deployment Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Load Balancer │────▶│   Cloud Run     │
└─────────────────┘     │   Instance 1    │
                        └─────────────────┘
                               │
                        ┌─────────────────┐
                        │   Cloud Run     │
                        │   Instance 2    │
                        └─────────────────┘
                               │
                        ┌─────────────────┐
                        │   Cloud Run     │
                        │   Instance N    │
                        └─────────────────┘

Features:
• Auto-scaling based on traffic
• Container-based deployment
• Environment variable configuration
• Health check endpoints
• Graceful shutdown handling
