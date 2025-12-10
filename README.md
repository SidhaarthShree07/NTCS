# NTCS Traffic Monitoring & Violation Detection System

## 📋 Project Overview

An intelligent traffic monitoring system that uses YOLOv8 for vehicle detection, **sensorless automated perspective calibration** (no roadside IoT sensors), and automated violation processing with OCR-based license plate recognition using Google Gemini Vision API. The system records evidence, uploads to Azure Blob Storage, and sends violations to a backend API for processing. The calibration pipeline is a first-class feature: it can auto-compute homography from video alone, validate scale, and publish calibration without external hardware.

## 🏗️ Project Structure

```
NTCS-CALIBRATION/
│
├── src/                                    # Backend Python application
│   ├── calib_server.py                     # Flask server for calibration & monitoring
│   ├── track.py                            # YOLOv8 tracking and speed detection
│   ├── violation_recorder.py               # Violation processing & evidence recording
│   ├── perspective_calibration.py          # Manual perspective calibration
│   ├── auto_homography.py                  # Automatic homography calculation
│   ├── gemini_lines.py                     # Gemini-based line detection
│   ├── tracked_vehicle_calibration.py      # Auto calibration using tracked vehicles
│   ├── vehicle_size_verification.py        # Vehicle dimension verification
│   ├── resources.py                        # Resource management utilities
│   │
│   ├── calib/                              # Calibration data storage
│   ├── models/                             # YOLO model files
│   ├── temp/                               # Temporary files and evidence
│   │
│   ├── Dockerfile                          # Production Docker image
│   ├── Dockerfile.gpu                      # GPU-enabled Docker image
│   ├── docker-compose.yml                  # Local development setup
│   └── requirements.txt                    # Python dependencies
│
├── frontend/                               # React frontend (separate repo)
│   └── → https://github.com/SidhaarthShree07/NTCS-CALIBRATION
│
│
└── README.md                               # This file
```

## 🚀 Features

### Core Capabilities
- **Real-time Vehicle Detection**: YOLOv8-based object detection and tracking
- **Speed Measurement**: Perspective-calibrated speed calculation using dual-line detection
- **Violation Detection**: Automated overspeed violation detection with configurable thresholds
- **License Plate OCR**: Google Gemini Vision API for accurate plate recognition (90%+ confidence filter)
- **Evidence Recording**: H.264 video segments with atomic writes and remuxing
- **Cloud Storage**: Azure Blob Storage integration for evidence archival
- **API Integration**: RESTful API for violation reporting to backend system
- **Sensorless Calibration**: Compute camera calibration purely from video (tracked vehicles and AI-detected lines) instead of roadside IoT sensors

### Calibration Methods
1. **Manual Calibration**: Point-and-click perspective calibration
2. **Auto Calibration**: Tracked vehicle-based distance calculation
3. **Gemini-Assisted**: AI-powered line detection and homography

## 🔄 Workflows

### Challan / Violation Workflow

```
┌─────────────────┐
│  Video Stream   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ YOLOv8 Tracker  │  ← Vehicle detection & tracking
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Speed Calc      │  ← Perspective transform + dual-line timing
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Violation?      │  ← Speed > Limit?
└────────┬────────┘
         │ YES
         v
┌─────────────────┐
│ Wait Middle Line│  ← Capture at optimal moment
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Record Segment  │  ← 10-sec rolling buffer (H.264)
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Trim Clip (5s)  │  ← FFmpeg re-encode with remux
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Gemini OCR      │  ← License plate recognition
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Upload Azure    │  ← Video + Images to Blob Storage
└────────┬────────┘
         │
         v
┌─────────────────┐
│ POST Violation  │  ← Send to Backend API
└─────────────────┘
```

### Calibration Workflow (Automated, Sensorless)

```
┌──────────────────┐
│ Sample Video In  │  ← Live stream or cached clip
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ Gemini Lines     │  ← Gemini-assisted detection of start/end lines
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ YOLO Tracks      │  ← Track same vehicle at both lines
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ Gemini Distance  │  ← Gemini infers real-world gap using vehicle dimensions
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ DeepLabV3 Road   │  ← Road segmentation → polygon
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ Homography Solve │  ← Use lines + polygon to compute H
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ Publish Calib    │  ← Lines, polygon, scale, fps
└─────────┬────────┘
          │
          v
┌──────────────────┐
│ Backend / API    │  ← Served to speed/violation engine
└──────────────────┘
```

## 📡 API Endpoints

### Calibration Server (Flask - Port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status |
| `/api/cameras` | GET | List all cameras |
| `/api/set_video_source` | POST | Set video stream source |
| `/api/load_calibration/:cameraId` | GET | Load calibration for camera |
| `/api/start_speed` | POST | Start speed detection |
| `/api/stop_speed` | POST | Stop speed detection |
| `/video_feed` | GET | Raw video stream |
| `/speed_feed` | GET | Annotated speed detection stream |

### Backend API Integration

**Endpoint**: `POST https://nextgen-fv1h.onrender.com/api/violations`

**Payload**:
```json
{
  "eventId": "EVT-XXXXX",
  "cameraId": "CAM-JAL-007",
  "capturedAt": "2025-11-16T18:17:44",
  "evidence": {
    "imageOriginalUrl": "https://...",
    "imageEnhancedUrl": "https://...",
    "videoClipUrl": "https://..."
  },
  "violation": {
    "type": "OVERSPEED",
    "measured": 76.4,
    "limit": 60.0
  },
  "vehicle": {
    "vehicleClass": "CAR",
    "plate": {
      "text": "7AC3391",
      "confidence": 0.99
    }
  }
}
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- FFmpeg (installed in Docker image)
- Azure Storage Account
- Google Gemini API Key
- Backend API endpoint

### Local Development

1. **Clone the repository**
```powershell
git clone https://github.com/SidhaarthShree07/NTCS-CALIBRATION.git
cd NTCS-CALIBRATION
```

2. **Set up Python environment**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r src/requirements.txt
```

3. **Configure environment variables**

Create `.env` file in project root:
```env
# Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=traffic-violations

# Backend API
HMAC_SECRET=your_hmac_secret
```

4. **Download YOLO models**
```powershell
# Place in src/models/
yolov8l.pt
yolov8l.onnx
```

5. **Run the server**
```powershell
cd src
python calib_server.py
```

Server will start at `http://localhost:5001`

### Docker Deployment

**Build Image**
```powershell
cd src
docker build -t ntcs-backend:latest .
```

**Run Container**
```powershell
docker run -d \
  -p 5001:5001 \
  -e GEMINI_API_KEY=your_key \
  -e AZURE_STORAGE_CONNECTION_STRING=your_string \
  -e AZURE_CONTAINER_NAME=traffic-violations \
  -e HMAC_SECRET=your_secret \
  --name ntcs-backend \
  ntcs-backend:latest
```

### Azure Container Apps Deployment

**Login to ACR**
```powershell
az acr login --name ntcscalibration
```

**Build and Push**
```powershell
cd src
docker build -t ntcscalibration.azurecr.io/ntcs-backend:v7.10 .
docker push ntcscalibration.azurecr.io/ntcs-backend:v7.10
```

**Update Container App**
```powershell
az containerapp update \
  --name ntcs-backend \
  --resource-group ntcs-rg \
  --image ntcscalibration.azurecr.io/ntcs-backend:v7.10
```

## 🔧 Configuration

### Camera Configuration (`cfg/app.yaml`)
```yaml
cameras:
  - id: CAM-JAL-007
    location: Tribune Chowk
    stream_url: https://traffic-stream.vercel.app/stream
    speed_limit: 60
```

### Calibration Settings

**Speed Detection**:
- Segment Duration: 10 seconds (250 frames @ 25 FPS)
- Violation Clip: 5 seconds (trimmed around violation)
- Confidence Threshold: 90% (OCR)
- Cooldown: 30 seconds per vehicle

**Video Recording**:
- Codec: H.264 (libx264)
- Container: MP4 with faststart
- FPS: 25
- Resolution: 3840x2160 (4K)

## 📊 System Requirements

### Minimum (CPU Inference)
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- Network: 10 Mbps upload

### Recommended (Production)
- CPU: 8 cores
- RAM: 16 GB
- Storage: 100 GB SSD
- Network: 50 Mbps upload

### GPU Deployment
- NVIDIA GPU with CUDA 11.8+
- 8 GB VRAM minimum
- Use `Dockerfile.gpu` for build

## 🐛 Troubleshooting

### Video Corruption Issues
If trimmed videos are tiny (< 1KB) or corrupt:
```powershell
# Remux all MP4 files in proof directory
Get-ChildItem temp\proof -Filter *.mp4 | ForEach-Object {
  $src = $_.FullName
  $out = "$($src).remux.mp4"
  ffmpeg -y -i $src -c copy -movflags +faststart $out
  Move-Item $out $src -Force
}
```

### OCR Low Confidence
- Ensure vehicle bbox is properly detected
- Check lighting conditions
- Verify Gemini API key is valid
- Review cropped vehicle images in `temp/proof/`

### Azure Upload Failures
- Verify connection string is correct
- Check container exists and has public read access
- Ensure blob name format is valid

## 📝 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini Vision API key | Yes |
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Storage connection string | Yes |
| `AZURE_CONTAINER_NAME` | Blob container name | Yes |
| `HMAC_SECRET` | HMAC signature secret | Yes |

## 📄 License

This project is proprietary software developed for NTCS Traffic Management System.

## 👥 Contributors

- Developer: Sidhaarth Shree
- Backend API: https://nextgen-fv1h.onrender.com
- Frontend Repo: https://github.com/SidhaarthShree07/NTCS-CALIBRATION

## 🔗 Related Links

- [Frontend Repository](https://github.com/SidhaarthShree07/NTCS-CALIBRATION)
- [Backend API Documentation](https://nextgen-fv1h.onrender.com/api/docs)
- [Azure Blob Storage Setup Guide](src/AZURE_BLOB_STORAGE_SETUP.md)
- [Deployment Guide](src/DEPLOYMENT_STEPS.md)

## 📞 Support

For issues or questions, please open an issue in the GitHub repository or contact the development team.

---

**Version**: 7.10  
**Last Updated**: December 2025  
**Status**: Production
