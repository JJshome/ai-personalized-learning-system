# 개발자 가이드

AI 기반 개인 맞춤형 학습 경로 추천 시스템의 개발자 가이드에 오신 것을 환영합니다. 이 문서는 시스템의 확장, 수정, 또는 통합을 위한 기술적 정보를 제공합니다.

## 목차

1. [개발 환경 설정](#개발-환경-설정)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [코드 구성](#코드-구성)
4. [핵심 모듈 및 API](#핵심-모듈-및-api)
5. [데이터 모델](#데이터-모델)
6. [확장 및 커스터마이징](#확장-및-커스터마이징)
7. [테스트 및 품질 보증](#테스트-및-품질-보증)
8. [배포 가이드](#배포-가이드)
9. [기여 가이드라인](#기여-가이드라인)
10. [문제 해결 및 FAQ](#문제-해결-및-faq)

## 개발 환경 설정

### 시스템 요구 사항

- **운영 체제**: Linux (Ubuntu 20.04+ 권장), macOS 10.15+, Windows 10+
- **Python**: 3.9+
- **Node.js**: 16.0+
- **데이터베이스**: PostgreSQL 13+, MongoDB 5.0+
- **GPU**: CUDA 호환 GPU (선택 사항이나 모델 학습에 권장)
- **Docker**: 20.10+ (컨테이너 배포용)

### 개발 환경 구성

1. 저장소 복제:
```bash
git clone https://github.com/your-organization/ai-personalized-learning-system.git
cd ai-personalized-learning-system
```

2. Python 가상 환경 설정:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발 전용 패키지
```

3. 프론트엔드 의존성 설치:
```bash
cd frontend
npm install
```

4. 환경 설정 파일 생성:
```bash
cp .env.example .env
# 환경 변수를 적절히 편집하세요
```

5. 데이터베이스 설정:
```bash
# PostgreSQL 설정
createdb aipls_db
python scripts/init_db.py

# MongoDB 설정 (필요시)
python scripts/init_mongodb.py
```

6. 개발 서버 실행:
```bash
# 백엔드 API 서버
python src/main.py --debug

# 프론트엔드 개발 서버 (별도 터미널에서)
cd frontend
npm run dev
```

## 시스템 아키텍처

시스템은 마이크로서비스 아키텍처를 기반으로 설계되었으며, 다음과 같은 주요 구성 요소로 이루어져 있습니다:

### 백엔드 서비스

1. **API 게이트웨이**: 모든 외부 요청을 처리하고 적절한 마이크로서비스로 라우팅합니다.
   - 기술 스택: FastAPI, Nginx
   - 디렉토리: `src/gateway/`

2. **사용자 서비스**: 인증, 권한 관리, 프로필 관리를 담당합니다.
   - 기술 스택: FastAPI, PostgreSQL, JWT
   - 디렉토리: `src/services/user/`

3. **학습 경로 서비스**: 학습 경로 생성, 관리, 추천을 담당합니다.
   - 기술 스택: FastAPI, PyTorch, MongoDB
   - 디렉토리: `src/services/learning_path/`

4. **콘텐츠 서비스**: 학습 콘텐츠 관리 및 제공을 담당합니다.
   - 기술 스택: FastAPI, MongoDB, S3
   - 디렉토리: `src/services/content/`

5. **분석 서비스**: 학습 데이터 분석 및 인사이트 생성을 담당합니다.
   - 기술 스택: FastAPI, PyTorch, Pandas, NumPy
   - 디렉토리: `src/services/analytics/`

6. **생체 데이터 서비스**: 웨어러블 기기에서 수집된 생체 데이터 처리를 담당합니다.
   - 기술 스택: FastAPI, WebSocket, Redis, NumPy, SciPy
   - 디렉토리: `src/services/biometric/`

### 프론트엔드

1. **웹 애플리케이션**: 학습자, 교사, 학부모를 위한 웹 인터페이스
   - 기술 스택: React, TypeScript, Tailwind CSS
   - 디렉토리: `frontend/web/`

2. **모바일 애플리케이션**: iOS 및 Android용 모바일 애플리케이션
   - 기술 스택: React Native, TypeScript
   - 디렉토리: `frontend/mobile/`

### 인프라 및 DevOps

1. **CI/CD 파이프라인**: 자동화된 테스트 및 배포 파이프라인
   - 기술 스택: GitHub Actions, Docker, Kubernetes
   - 디렉토리: `.github/workflows/`, `deployment/`

2. **모니터링 및 로깅**: 시스템 모니터링 및 로깅 인프라
   - 기술 스택: Prometheus, Grafana, ELK Stack
   - 디렉토리: `monitoring/`

### 시스템 통신 흐름도

```
클라이언트 애플리케이션 <--> API 게이트웨이 <--> 마이크로서비스
                                   ^
                                   |
                                   v
                              인증 서비스
```

각 마이크로서비스는 다음과 같은 방식으로 통신합니다:

- **동기식 통신**: REST API 또는 gRPC를 통한 서비스 간 직접 통신
- **비동기식 통신**: Kafka 또는 RabbitMQ를 통한 메시지 기반 통신
- **실시간 통신**: WebSocket 또는 Server-Sent Events를 통한 클라이언트와의 실시간 통신

## 코드 구성

### 디렉토리 구조

```
ai-personalized-learning-system/
├── .github/                    # GitHub 관련 설정
├── docs/                       # 프로젝트 문서
├── frontend/                   # 프론트엔드 코드
│   ├── web/                    # 웹 애플리케이션
│   └── mobile/                 # 모바일 애플리케이션
├── monitoring/                 # 모니터링 및 로깅 설정
├── scripts/                    # 유틸리티 스크립트
├── src/                        # 백엔드 소스 코드
│   ├── gateway/                # API 게이트웨이
│   ├── core/                   # 공통 유틸리티 및 라이브러리
│   ├── models/                 # 공유 데이터 모델
│   └── services/               # 마이크로서비스
│       ├── user/               # 사용자 서비스
│       ├── learning_path/      # 학습 경로 서비스
│       ├── content/            # 콘텐츠 서비스
│       ├── analytics/          # 분석 서비스
│       └── biometric/          # 생체 데이터 서비스
├── tests/                      # 테스트 코드
├── deployment/                 # 배포 설정
├── .env.example                # 환경 변수 예시
├── docker-compose.yml          # Docker Compose 설정
├── README.md                   # 프로젝트 개요
└── requirements.txt            # Python 의존성
```

### 코딩 규칙

이 프로젝트는 다음과 같은 코딩 규칙을 따릅니다:

- **Python**: PEP 8 스타일 가이드, Black 포맷터 사용
- **TypeScript/JavaScript**: Airbnb 스타일 가이드, ESLint 및 Prettier 사용
- **문서화**: 모든 공개 API, 클래스, 함수에 대한 문서 주석 필수
- **테스트**: 핵심 기능에 대한 단위 테스트 및 통합 테스트 필수

코드 스타일 검사 및 테스트를 위한 커맨드:

```bash
# 코드 스타일 검사
make lint

# 코드 포맷팅
make format

# 테스트 실행
make test
```

## 핵심 모듈 및 API

### 학습 경로 생성 모듈

학습 경로 생성 모듈(`src/services/learning_path/generator.py`)은 강화학습 기반 알고리즘을 사용하여 개인화된 학습 경로를 생성합니다.

```python
from pathlib import Path
import torch
from src.models import LearnerProfile, LearningPathConfig, LearningPath

class PathGenerator:
    def __init__(self, model_path: Path = None):
        """
        학습 경로 생성기 초기화
        
        Args:
            model_path: 사전 학습된 모델 경로 (기본값: 최신 모델 사용)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: Path = None):
        """모델 로드 로직"""
        pass
        
    def generate_path(self, 
                      learner_profile: LearnerProfile, 
                      config: LearningPathConfig) -> LearningPath:
        """
        학습자 프로필과 설정을 기반으로 학습 경로 생성
        
        Args:
            learner_profile: 학습자 프로필 정보
            config: 학습 경로 설정 (주제, 목표, 난이도 등)
            
        Returns:
            생성된 학습 경로
        """
        pass
```

이 모듈을 사용하는 예제:

```python
from src.models import LearnerProfile, LearningPathConfig
from src.services.learning_path.generator import PathGenerator

# 학습자 프로필 생성
profile = LearnerProfile(
    learner_id="learner_123",
    knowledge_state={"algebra": 0.7, "geometry": 0.5},
    learning_style="visual",
    preferred_content_types=["video", "interactive"]
)

# 학습 경로 설정
config = LearningPathConfig(
    subject="mathematics",
    topic="calculus",
    difficulty="intermediate",
    target_duration_days=14
)

# 경로 생성
generator = PathGenerator()
path = generator.generate_path(profile, config)

print(f"생성된 경로: {path.path_id}")
print(f"총 학습 단위 수: {len(path.units)}")
```

### 생체 신호 처리 모듈

생체 신호 처리 모듈(`src/services/biometric/processor.py`)은 웨어러블 기기에서 수집된 EEG, 심박수 등의 데이터를 처리하여 집중도, 스트레스 수준 등의 지표를 계산합니다.

```python
import numpy as np
from scipy import signal
from src.models import BiometricData, CognitiveState

class BiometricProcessor:
    def __init__(self):
        """생체 신호 처리기 초기화"""
        pass
        
    def process_eeg(self, raw_data: np.ndarray) -> dict:
        """
        EEG 데이터 처리
        
        Args:
            raw_data: 원시 EEG 데이터 배열
            
        Returns:
            처리된 EEG 특성 (알파파, 베타파 등의 비율)
        """
        pass
        
    def process_heart_rate(self, raw_data: np.ndarray) -> dict:
        """
        심박수 데이터 처리
        
        Args:
            raw_data: 원시 심박수 데이터 배열
            
        Returns:
            처리된 심박 특성 (HRV, 심박수 등)
        """
        pass
        
    def estimate_cognitive_state(self, data: BiometricData) -> CognitiveState:
        """
        생체 데이터를 기반으로 인지 상태 추정
        
        Args:
            data: 다양한 생체 신호를 포함하는 데이터 객체
            
        Returns:
            추정된 인지 상태 (집중도, 인지 부하, 감정 상태 등)
        """
        pass
```

### 콘텐츠 적응 API

콘텐츠 적응 API는 학습자의 특성과 요구에 맞게 학습 콘텐츠를 조정하는 기능을 제공합니다.

```http
POST /api/v1/content/adapt
Content-Type: application/json
Authorization: Bearer <token>

{
  "content_id": "content_123",
  "learner_id": "learner_456",
  "adaptations": {
    "difficulty": "easier",
    "format": "visual",
    "detail_level": "more",
    "additional_examples": true
  }
}
```

응답:

```json
{
  "adapted_content_id": "content_123_adapted_789",
  "original_content_id": "content_123",
  "adaptations_applied": {
    "difficulty": "reduced by 1 level",
    "format": "converted to visual format",
    "detail_level": "expanded with additional details",
    "additional_examples": "added 3 examples"
  },
  "content_url": "https://api.aipls.example.com/content/content_123_adapted_789"
}
```

## 데이터 모델

### 주요 데이터 모델

시스템의 핵심 데이터 모델은 `src/models/` 디렉토리에 정의되어 있습니다:

#### 학습자 프로필 (LearnerProfile)

```python
class LearnerProfile:
    """학습자 프로필 데이터 모델"""
    
    def __init__(self,
                 learner_id: str,
                 knowledge_state: dict,
                 learning_style: str,
                 preferred_content_types: list,
                 cognitive_traits: dict = None,
                 learning_goals: list = None,
                 demographic_info: dict = None):
        """
        Args:
            learner_id: 학습자 고유 ID
            knowledge_state: 주제별 지식 수준 (예: {"algebra": 0.7, "geometry": 0.5})
            learning_style: 선호하는 학습 스타일 (visual, auditory, kinesthetic 등)
            preferred_content_types: 선호하는 콘텐츠 유형 목록
            cognitive_traits: 인지 특성 (작업 기억 용량, 처리 속도 등)
            learning_goals: 학습 목표 목록
            demographic_info: 인구통계학적 정보 (나이, 학년 등)
        """
        self.learner_id = learner_id
        self.knowledge_state = knowledge_state
        self.learning_style = learning_style
        self.preferred_content_types = preferred_content_types
        self.cognitive_traits = cognitive_traits or {}
        self.learning_goals = learning_goals or []
        self.demographic_info = demographic_info or {}
```

#### 학습 경로 (LearningPath)

```python
class LearningUnit:
    """학습 경로의 단일 학습 단위"""
    
    def __init__(self,
                 unit_id: str,
                 title: str,
                 content_type: str,
                 difficulty: float,
                 concepts: list,
                 prerequisites: list,
                 estimated_duration: int,
                 content_url: str = None):
        self.unit_id = unit_id
        self.title = title
        self.content_type = content_type
        self.difficulty = difficulty
        self.concepts = concepts
        self.prerequisites = prerequisites
        self.estimated_duration = estimated_duration
        self.content_url = content_url


class LearningPath:
    """학습 경로 데이터 모델"""
    
    def __init__(self,
                 path_id: str,
                 learner_id: str,
                 subject: str,
                 topic: str,
                 units: list[LearningUnit],
                 difficulty: str,
                 target_duration_days: int,
                 created_at: datetime,
                 modified_at: datetime = None):
        self.path_id = path_id
        self.learner_id = learner_id
        self.subject = subject
        self.topic = topic
        self.units = units
        self.difficulty = difficulty
        self.target_duration_days = target_duration_days
        self.created_at = created_at
        self.modified_at = modified_at or created_at
```

#### 생체 데이터 (BiometricData)

```python
class EEGData:
    """EEG 생체 신호 데이터"""
    
    def __init__(self,
                 raw_data: np.ndarray,
                 channel_names: list,
                 sampling_rate: int,
                 timestamp: datetime):
        self.raw_data = raw_data
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate
        self.timestamp = timestamp


class HeartRateData:
    """심박수 생체 신호 데이터"""
    
    def __init__(self,
                 values: list,
                 sampling_rate: int,
                 timestamp: datetime):
        self.values = values
        self.sampling_rate = sampling_rate
        self.timestamp = timestamp


class BiometricData:
    """통합 생체 데이터"""
    
    def __init__(self,
                 learner_id: str,
                 session_id: str,
                 eeg_data: EEGData = None,
                 heart_rate_data: HeartRateData = None,
                 temperature_data: list = None,
                 gsr_data: list = None,
                 timestamp: datetime = None):
        self.learner_id = learner_id
        self.session_id = session_id
        self.eeg_data = eeg_data
        self.heart_rate_data = heart_rate_data
        self.temperature_data = temperature_data
        self.gsr_data = gsr_data
        self.timestamp = timestamp or datetime.now()
```

### 데이터베이스 스키마

시스템은 다음과 같은 데이터베이스를 사용합니다:

1. **PostgreSQL**: 사용자 계정, 인증, 권한 등 관계형 데이터 저장
2. **MongoDB**: 학습 경로, 콘텐츠, 분석 데이터 등 비구조적 데이터 저장
3. **Redis**: 캐싱, 세션 관리, 실시간 데이터 처리

PostgreSQL 스키마 예시 (`src/services/user/schema.sql`):

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE user_profiles (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    learning_style VARCHAR(50),
    preferred_content_types JSONB,
    demographic_info JSONB,
    settings JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

MongoDB 컬렉션 예시 (`src/services/learning_path/db.py`):

```python
# MongoDB 컬렉션 스키마 (PyMongo 사용)
learning_paths_schema = {
    "validator": {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["path_id", "learner_id", "subject", "topic", "units"],
            "properties": {
                "path_id": {
                    "bsonType": "string",
                    "description": "학습 경로 고유 ID"
                },
                "learner_id": {
                    "bsonType": "string",
                    "description": "학습자 고유 ID"
                },
                "subject": {
                    "bsonType": "string",
                    "description": "학습 주제"
                },
                "topic": {
                    "bsonType": "string",
                    "description": "세부 주제"
                },
                "units": {
                    "bsonType": "array",
                    "description": "학습 단위 목록",
                    "items": {
                        "bsonType": "object",
                        "required": ["unit_id", "title", "content_type"],
                        "properties": {
                            "unit_id": {
                                "bsonType": "string",
                                "description": "학습 단위 고유 ID"
                            },
                            "title": {
                                "bsonType": "string",
                                "description": "학습 단위 제목"
                            },
                            "content_type": {
                                "bsonType": "string",
                                "description": "콘텐츠 유형"
                            }
                        }
                    }
                }
            }
        }
    }
}

# 컬렉션 생성 및 스키마 적용
db.create_collection("learning_paths", **learning_paths_schema)
db.learning_paths.create_index([("learner_id", 1)])
db.learning_paths.create_index([("path_id", 1)], unique=True)
```

## 확장 및 커스터마이징

### 새 학습 경로 알고리즘 추가

시스템은 플러그인 아키텍처를 통해 새로운 학습 경로 생성 알고리즘을 쉽게 추가할 수 있습니다:

1. `src/services/learning_path/algorithms/` 디렉토리에 새 알고리즘 모듈 생성
2. 기본 알고리즘 인터페이스 구현

```python
# src/services/learning_path/algorithms/my_algorithm.py
from src.services.learning_path.algorithms.base import BaseAlgorithm
from src.models import LearnerProfile, LearningPathConfig, LearningPath

class MyPathAlgorithm(BaseAlgorithm):
    """새로운 학습 경로 생성 알고리즘"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 알고리즘 초기화 로직
        
    def generate(self, learner_profile: LearnerProfile, config: LearningPathConfig) -> LearningPath:
        """
        학습 경로 생성 알고리즘 구현
        
        Args:
            learner_profile: 학습자 프로필
            config: 학습 경로 설정
            
        Returns:
            생성된 학습 경로
        """
        # 알고리즘 로직 구현
        pass
```

3. 알고리즘 등록

```python
# src/services/learning_path/algorithm_registry.py
from src.services.learning_path.algorithms.my_algorithm import MyPathAlgorithm

# 알고리즘 등록
register_algorithm("my_algorithm", MyPathAlgorithm)
```

4. 설정에서 알고리즘 활성화

```python
# config.yaml
learning_path:
  default_algorithm: "my_algorithm"
  algorithms:
    my_algorithm:
      enabled: true
      config:
        param1: value1
        param2: value2
```

### 새 웨어러블 기기 통합

새로운 웨어러블 기기를 시스템에 통합하려면:

1. `src/services/biometric/devices/` 디렉토리에 새 기기 드라이버 생성
2. 기본 기기 인터페이스 구현

```python
# src/services/biometric/devices/my_device.py
from src.services.biometric.devices.base import BaseDevice
from src.models import BiometricData

class MyDevice(BaseDevice):
    """새 웨어러블 기기 드라이버 구현"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 기기 초기화 로직
        
    async def connect(self, device_id: str):
        """기기에 연결"""
        pass
        
    async def disconnect(self):
        """기기 연결 해제"""
        pass
        
    async def start_streaming(self, callback):
        """데이터 스트리밍 시작"""
        pass
        
    async def stop_streaming(self):
        """데이터 스트리밍 중지"""
        pass
        
    def process_raw_data(self, raw_data) -> BiometricData:
        """원시 데이터를 BiometricData 객체로 변환"""
        pass
```

3. 기기 등록

```python
# src/services/biometric/device_registry.py
from src.services.biometric.devices.my_device import MyDevice

# 기기 등록
register_device("my_device", MyDevice)
```

4. 설정에서 기기 활성화

```python
# config.yaml
biometric:
  devices:
    my_device:
      enabled: true
      config:
        param1: value1
        param2: value2
```

## 테스트 및 품질 보증

### 테스트 프레임워크

시스템은 다음과 같은 테스트 프레임워크를 사용합니다:

- **단위 테스트**: pytest (Python), Jest (JavaScript)
- **통합 테스트**: pytest-fastapi, supertest
- **E2E 테스트**: Cypress, Selenium
- **로드 테스트**: Locust

### 테스트 실행

```bash
# 단위 테스트 실행
pytest tests/unit

# 통합 테스트 실행
pytest tests/integration

# E2E 테스트 실행
cd frontend
npm run test:e2e

# 커버리지 리포트 생성
pytest --cov=src tests/
```

### CI/CD 통합

GitHub Actions를 사용한 지속적 통합 및 배포 설정:

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Lint with flake8
        run: flake8 src tests
      - name: Type check with mypy
        run: mypy src
      - name: Test with pytest
        run: pytest tests/ --cov=src --cov-report=xml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v1
```

## 배포 가이드

### Docker 배포

Docker를 사용한 배포 방법:

```bash
# 이미지 빌드
docker-compose build

# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### Kubernetes 배포

Kubernetes를 사용한 배포 방법:

1. Helm 차트 설치:

```bash
cd deployment/helm
helm install aipls ./aipls-chart -n aipls --create-namespace
```

2. 배포 상태 확인:

```bash
kubectl get pods -n aipls
kubectl get services -n aipls
```

3. 인그레스 설정:

```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

4. 스케일링:

```bash
# 수평적 스케일링
kubectl scale deployment aipls-api --replicas=3 -n aipls

# 수직적 스케일링
kubectl apply -f deployment/kubernetes/resources/high-load.yaml
```

### 클라우드 배포

AWS, GCP, Azure 등의 클라우드 플랫폼에 배포하는 방법:

1. AWS EKS:

```bash
# EKS 클러스터 생성
eksctl create cluster -f deployment/aws/cluster.yaml

# Kubernetes 설정
kubectl apply -f deployment/aws/k8s/
```

2. GCP GKE:

```bash
# GKE 클러스터 생성
gcloud container clusters create aipls-cluster --num-nodes=3

# Kubernetes 설정
kubectl apply -f deployment/gcp/k8s/
```

3. Azure AKS:

```bash
# AKS 클러스터 생성
az aks create -g myResourceGroup -n aiplsAKSCluster --node-count 3

# Kubernetes 설정
kubectl apply -f deployment/azure/k8s/
```

## 기여 가이드라인

### 개발 워크플로우

1. 이슈 생성 또는 할당
2. 기능 브랜치 생성 (`feature/feature-name` 또는 `bugfix/bug-name`)
3. 코드 작성 및 테스트
4. Pull Request 생성
5. 코드 리뷰 및 수정
6. PR 승인 및 병합

### 커밋 메시지 규칙

커밋 메시지는 다음 형식을 따릅니다:

```
<유형>(<범위>): <제목>

<본문>

<꼬리말>
```

유형:
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 형식 변경 (코드 기능 변경 없음)
- `refactor`: 코드 리팩토링 (기능 변경이나 버그 수정 없음)
- `test`: 테스트 추가 또는 수정
- `chore`: 빌드 프로세스, 도구 변경 등

예시:
```
feat(learning-path): 강화학습 기반 경로 생성 알고리즘 추가

- PPO 알고리즘을 사용하여 학습 경로 최적화
- 지식 그래프와 학습자 선호도를 고려한 보상 함수 구현
- 경로 생성 성능 20% 향상

Closes #123
```

### 코드 리뷰 가이드라인

코드 리뷰 시 다음 사항에 집중합니다:

1. 코드 품질 및 가독성
2. 테스트 커버리지
3. 성능 및 확장성
4. 보안 및 개인정보 보호
5. 문서화 및 주석

## 문제 해결 및 FAQ

### 일반적인 개발 문제

#### 의존성 충돌

문제: 패키지 의존성 충돌로 인한 설치 오류

해결책:
```bash
# 가상 환경 재생성
rm -rf venv
python -m venv venv
source venv/bin/activate

# 의존성 순서대로 설치
pip install -r requirements-base.txt
pip install -r requirements-ext.txt
pip install -r requirements-dev.txt
```

#### 데이터베이스 연결 오류

문제: 데이터베이스 연결 실패

해결책:
1. 데이터베이스 서비스 실행 여부 확인
```bash
systemctl status postgresql
systemctl status mongodb
```

2. 환경 변수 확인
```bash
cat .env | grep DB_
```

3. 데이터베이스 로그 확인
```bash
tail -f /var/log/postgresql/postgresql-13-main.log
```

#### 모델 학습 문제

문제: GPU 메모리 부족으로 인한 모델 학습 실패

해결책:
1. 배치 크기 감소
```python
# config.yaml
training:
  batch_size: 32  # 64에서 32로 감소
```

2. 그래디언트 누적 사용
```python
# src/services/learning_path/trainer.py
accumulation_steps = 4  # 그래디언트를 4번 누적 후 업데이트
```

3. 혼합 정밀도 학습 사용
```python
# src/services/learning_path/trainer.py
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

### API 관련 문제

#### 요청 타임아웃

문제: API 요청 처리 시간 초과

해결책:
1. 요청 처리를 비동기화
```python
# src/services/content/routes.py
@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content(request: ContentGenerationRequest):
    # 요청을 큐에 넣고 작업 ID 반환
    job_id = await content_service.queue_generation_job(request)
    return {"job_id": job_id, "status": "queued"}
    
@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    # 작업 상태 확인
    status = await content_service.get_job_status(job_id)
    return status
```

2. 처리 시간이 긴 작업은 웹훅 사용
```python
# config.yaml
content:
  generation:
    use_webhooks: true
    webhook_url: "https://example.com/api/webhooks/content-ready"
```

#### 인증 문제

문제: 토큰 만료 또는 검증 실패

해결책:
1. 토큰 새로 고침 구현
```python
# src/services/user/auth.py
@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str = Body(..., embed=True)):
    new_tokens = await auth_service.refresh_token(refresh_token)
    return new_tokens
```

2. 토큰 디버깅 도구 사용
```bash
# JWT 토큰 내용 확인
python -c "import jwt, sys; print(jwt.decode(sys.argv[1], verify=False))" <token>
```

### 확장성 관련 FAQ

#### Q: 대규모 사용자를 처리하기 위한 추천 아키텍처는?
A: 대규모 시스템의 경우 다음과 같은 접근 방식을 권장합니다:
- 마이크로서비스 분리: 각 서비스를 독립적으로 스케일링
- 읽기/쓰기 분리: 읽기 전용 복제본 사용
- 캐싱 레이어: Redis, Memcached 등을 사용한 결과 캐싱
- CDN: 정적 콘텐츠 배포용 CDN 사용
- 메시지 큐: Kafka, RabbitMQ 등을 통한 비동기 처리
- 샤딩: 사용자 ID 기반 데이터베이스 샤딩

#### Q: 새로운 AI 모델을 추가하는 방법은?
A: 새로운 AI 모델을 추가하려면:
1. `src/services/learning_path/models/` 디렉토리에 모델 구현
2. 모델 인터페이스 구현
3. 모델 등록 (`src/services/learning_path/model_registry.py`)
4. 설정에서 모델 활성화
5. 모델 평가 및 비교 실행

#### Q: 시스템의 보안 취약점을 테스트하는 방법은?
A: 다음과 같은 보안 테스트를 권장합니다:
1. 정적 코드 분석: Bandit, SonarQube 등을 사용
2. 의존성 취약점 검사: Safety, Dependabot 등 사용
3. 침투 테스트: OWASP ZAP, Burp Suite 등을 사용한 API 취약점 스캔
4. 인증/인가 테스트: 역할 기반 접근 제어 테스트
5. 개인정보 보호 검사: 민감 정보 유출 테스트

## 추가 자료

- [API 문서](https://docs.aipls.example.com/api)
- [데이터 모델 도표](https://docs.aipls.example.com/models)
- [아키텍처 다이어그램](https://docs.aipls.example.com/architecture)
- [성능 최적화 가이드](https://docs.aipls.example.com/performance)
- [보안 체크리스트](https://docs.aipls.example.com/security)