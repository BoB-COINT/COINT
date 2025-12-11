# COINT - Token Scam Detection System

Ethereum 토큰 스캠 탐지 플랫폼

## System Overview

COINT는 3가지 ML 기반 탐지 모듈을 통합한 토큰 스캠 분석 시스템입니다:

1. **Honeypot Detection (Dynamic Analysis)**: Brownie 기반 스마트 컨트랙트 시뮬레이션 테스트
2. **Honeypot Detection (ML)**: XGBoost v8 모델 (67 features, 96% accuracy)
3. **Exit Scam Detection (ML)**: Attention-based MIL 모델 (거래 패턴 분석)

## Setup

### Production (WSL)
```bash
# Install system dependencies
sudo apt update
sudo apt install nginx python3-pip

# Setup virtual environment
cd /mnt/c/Users/i_jon/OneDrive/Desktop/COINT
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
pip install gunicorn

# Run migrations
python manage.py migrate

# Setup systemd service
sudo systemctl start gunicorn
sudo systemctl enable gunicorn

# Configure nginx
sudo service nginx start
```

## Environment Variables

Create `.env` file:
```env
SECRET_KEY=your-django-secret-key
DEBUG=True

# Blockchain data collection
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_API_KEY
ETHERSCAN_API_URL=https://api.etherscan.io/v2/api
MORALIS_API_KEY=YOUR_MORALIS_API_KEY
CHAINBASE_API_KEY=YOUR_CHAINBASE_API_KEY
```

## Project Structure

```
api/                    Django app (models, views, serializers)
├── models.py          Database schema (11 tables)
├── views.py           REST API endpoints
├── serializers.py     DRF serializers
└── migrations/        DB migrations

pipeline/              Analysis pipeline orchestration
├── adapters.py        Module integration adapters
└── orchestrator.py    Pipeline coordinator

modules/               Analysis modules
├── data_collector/    Unified blockchain data collector
├── honeypot_DA/       Dynamic analysis (Brownie-based)
├── honeypot_ML/       ML-based honeypot detection (XGBoost)
├── exit_ML/           Exit scam detection (Attention MIL)
├── preprocessor/      Feature engineering
└── detect_unformed_lp/ Unformed LP detection

config/                Django settings
```

## Database Schema

11 테이블로 구성:

### Raw Data (3 tables)
- `token_info`: 토큰 메타데이터 및 페어 정보
- `pair_evt`: 페어 이벤트 로그 (Mint, Burn, Swap, Sync)
- `holder_info`: 토큰 홀더 정보

### Processed Data (3 tables)
- `honeypot_processed_data`: Honeypot 탐지 피처 (67개)
- `exit_processed_data_instance`: Exit scam 인스턴스 피처 (5초 윈도우)
- `exit_processed_data_static`: Exit scam 정적 피처

### Analysis Results (5 tables)
- `honeypot_da_result`: 동적 분석 결과
- `honeypot_ml_result`: ML 기반 honeypot 탐지 결과
- `exit_ml_result`: Exit scam 탐지 결과
- `exit_ml_detect_transaction`: 거래별 탐지 상세
- `exit_ml_detect_static`: 윈도우별 정적 피처

### Final Output (1 table)
- `result`: 통합 분석 결과 및 리스크 스코어

## Deployment

### Production Server
- **Domain**: bob-coint.site
- **Backend**: WSL + Gunicorn + Nginx
- **Frontend**: React (port 3000)
- **Database**: SQLite (production)

### Access
- Frontend: http://bob-coint.site
- API: http://bob-coint.site/api/

### Services
```bash
# Gunicorn service
sudo systemctl status gunicorn
sudo systemctl restart gunicorn

# Nginx
sudo service nginx status
sudo service nginx reload
```

## Technologies

**Backend:**
- Django 5.2.7 + Django REST Framework
- SQLite (production & development)
- Gunicorn (WSGI server)
- Nginx (reverse proxy)

**Blockchain:**
- Web3.py 6.20.0 (Ethereum interaction)
- Etherscan API v2
- Moralis API
- Chainbase API
- Brownie (smart contract testing)

**Machine Learning:**
- XGBoost 1.7.6 (honeypot detection)
- PyTorch 2.9.1 (exit scam detection)
- Pandas, NumPy, scikit-learn

**Frontend:**
- React 18
- Environment: Development server (port 3000)

**Infrastructure:**
- WSL (Ubuntu)
- Port forwarding (80, 443)
- Domain: bob-coint.site

## API Usage

### REST API Endpoints

**Create Analysis Job:**
```bash
POST /api/jobs/
Content-Type: application/json

{
  "token_addr": "0x...",
  "days": 30  # optional, analysis period in days
}
```

**Response:**
```json
{
  "id": 1,
  "token_addr": "0x...",
  "status": "pending",
  "created_at": "2025-12-10T12:00:00Z"
}
```

**Get Job Status:**
```bash
GET /api/jobs/{id}/
```

### Python API (Internal)

```python
from pipeline.orchestrator import PipelineOrchestrator

# Run complete analysis
orchestrator = PipelineOrchestrator()
success = orchestrator.execute(token_addr="0x...", days=30)

# Check cached result
from api.models import Result
result = Result.objects.get(token_addr__iexact="0x...")
```

## Development

```bash
# Run tests
python manage.py test

# Create new migration
python manage.py makemigrations

# Apply migrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Run development server
python manage.py runserver
```

## License

Proprietary - BoB Project
