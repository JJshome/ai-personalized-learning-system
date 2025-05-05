# API 참조

AI 기반 개인 맞춤형 학습 경로 추천 시스템은 다양한 API를 제공하여 외부 시스템과의 통합 및 확장을 가능하게 합니다. 이 문서는 시스템의 주요 API에 대한 참조 가이드입니다.

## API 개요

시스템의 API는 다음과 같은 특징을 가지고 있습니다:

- **RESTful 아키텍처**: 표준 HTTP 메서드(GET, POST, PUT, DELETE)를 사용하여 리소스를 관리합니다.
- **GraphQL 지원**: 복잡한 쿼리가 필요한 경우 GraphQL 엔드포인트를 통해 필요한 데이터만 정확히 요청할 수 있습니다.
- **WebSocket 연결**: 실시간 데이터 스트리밍이 필요한 경우 WebSocket 연결을 통해 지속적인 데이터 흐름을 제공합니다.
- **OAuth 2.0 인증**: 안전한
 API 접근을 위해 OAuth 2.0 인증 프로토콜을 사용합니다.
- **JSON 데이터 형식**: 모든 API 요청 및 응답은 JSON 형식을 사용합니다.

## 기본 URL

```
https://api.aipls.example.com/v1
```

## 인증

API를 사용하기 위해서는 OAuth 2.0 인증이 필요합니다. 다음과 같은 방법으로 인증 토큰을 획득할 수 있습니다:

```http
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&client_id=YOUR_CLIENT_ID&client_secret=YOUR_CLIENT_SECRET
```

응답:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

획득한 토큰은 모든 API 요청의 헤더에 포함되어야 합니다:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 주요 API 엔드포인트

### 학습자 관리 API

#### 학습자 생성

```http
POST /learners
Content-Type: application/json

{
  "name": "홍길동",
  "email": "hong@example.com",
  "birth_date": "2010-05-15",
  "grade": 5,
  "preferences": {
    "learning_style": "visual",
    "difficulty_preference": "challenging",
    "session_duration": 25
  }
}
```

응답:

```json
{
  "id": "learner_12345",
  "name": "홍길동",
  "email": "hong@example.com",
  "birth_date": "2010-05-15",
  "grade": 5,
  "preferences": {
    "learning_style": "visual",
    "difficulty_preference": "challenging",
    "session_duration": 25
  },
  "created_at": "2025-05-01T09:00:00Z"
}
```

#### 학습자 조회

```http
GET /learners/{learner_id}
```

응답:

```json
{
  "id": "learner_12345",
  "name": "홍길동",
  "email": "hong@example.com",
  "birth_date": "2010-05-15",
  "grade": 5,
  "preferences": {
    "learning_style": "visual",
    "difficulty_preference": "challenging",
    "session_duration": 25
  },
  "progress": {
    "current_path_id": "path_56789",
    "completed_units": 15,
    "total_units": 30,
    "average_score": 85.5
  },
  "created_at": "2025-05-01T09:00:00Z",
  "updated_at": "2025-05-05T14:30:00Z"
}
```

#### 학습자 업데이트

```http
PUT /learners/{learner_id}
Content-Type: application/json

{
  "preferences": {
    "learning_style": "auditory",
    "difficulty_preference": "balanced",
    "session_duration": 30
  }
}
```

응답:

```json
{
  "id": "learner_12345",
  "name": "홍길동",
  "email": "hong@example.com",
  "preferences": {
    "learning_style": "auditory",
    "difficulty_preference": "balanced",
    "session_duration": 30
  },
  "updated_at": "2025-05-05T15:00:00Z"
}
```

### 학습 경로 API

#### 학습 경로 생성 요청

```http
POST /learning-paths
Content-Type: application/json

{
  "learner_id": "learner_12345",
  "subject": "mathematics",
  "topic": "algebra",
  "learning_objective": "Solve quadratic equations with confidence",
  "target_duration_days": 14,
  "difficulty": "intermediate"
}
```

응답:

```json
{
  "path_id": "path_56789",
  "learner_id": "learner_12345",
  "subject": "mathematics",
  "topic": "algebra",
  "learning_objective": "Solve quadratic equations with confidence",
  "target_duration_days": 14,
  "difficulty": "intermediate",
  "units": [
    {
      "unit_id": "unit_1",
      "title": "Linear Equations Review",
      "estimated_duration_minutes": 30,
      "prerequisites": []
    },
    {
      "unit_id": "unit_2",
      "title": "Introduction to Quadratic Equations",
      "estimated_duration_minutes": 45,
      "prerequisites": ["unit_1"]
    },
    // 추가 학습 단위...
  ],
  "created_at": "2025-05-05T15:05:00Z"
}
```

#### 학습 경로 조회

```http
GET /learning-paths/{path_id}
```

응답:

```json
{
  "path_id": "path_56789",
  "learner_id": "learner_12345",
  "subject": "mathematics",
  "topic": "algebra",
  "learning_objective": "Solve quadratic equations with confidence",
  "target_duration_days": 14,
  "difficulty": "intermediate",
  "units": [
    {
      "unit_id": "unit_1",
      "title": "Linear Equations Review",
      "status": "completed",
      "score": 95,
      "completion_date": "2025-05-06T10:30:00Z"
    },
    {
      "unit_id": "unit_2",
      "title": "Introduction to Quadratic Equations",
      "status": "in_progress",
      "progress_percentage": 65
    },
    // 추가 학습 단위...
  ],
  "overall_progress": 0.35,
  "estimated_completion_date": "2025-05-18T00:00:00Z"
}
```

#### 학습 경로 재생성 요청

```http
POST /learning-paths/{path_id}/regenerate
Content-Type: application/json

{
  "reason": "too_difficult",
  "feedback": "The pace is too fast, need more explanation and practice",
  "preserve_completed_units": true
}
```

응답:

```json
{
  "path_id": "path_56789",
  "learner_id": "learner_12345",
  "units": [
    {
      "unit_id": "unit_1",
      "title": "Linear Equations Review",
      "status": "completed",
      "score": 95
    },
    {
      "unit_id": "unit_2a",
      "title": "Basics of Quadratic Expressions",
      "status": "pending"
    },
    {
      "unit_id": "unit_2b",
      "title": "Graphical Representation of Quadratic Functions",
      "status": "pending"
    },
    // 변경된 학습 단위...
  ],
  "regenerated_at": "2025-05-07T09:15:00Z"
}
```

### 학습 콘텐츠 API

#### 학습 단위 콘텐츠 조회

```http
GET /learning-units/{unit_id}/content
```

응답:

```json
{
  "unit_id": "unit_2",
  "title": "Introduction to Quadratic Equations",
  "content_type": "mixed",
  "content_blocks": [
    {
      "type": "text",
      "content": "A quadratic equation is a second-degree polynomial equation in a single variable...",
      "order": 1
    },
    {
      "type": "image",
      "url": "https://content.aipls.example.com/images/quadratic-formula.png",
      "alt_text": "The quadratic formula: x = (-b ± √(b² - 4ac)) / 2a",
      "order": 2
    },
    {
      "type": "video",
      "url": "https://content.aipls.example.com/videos/solving-quadratics.mp4",
      "duration_seconds": 180,
      "transcript_url": "https://content.aipls.example.com/transcripts/solving-quadratics.txt",
      "order": 3
    },
    {
      "type": "interactive",
      "interactive_type": "quiz",
      "questions": [
        {
          "question_id": "q1",
          "text": "What is the degree of a quadratic equation?",
          "options": ["1", "2", "3", "4"],
          "correct_option_index": 1
        }
        // 추가 문제...
      ],
      "order": 4
    }
  ],
  "estimated_duration_minutes": 45,
  "difficulty_level": "intermediate"
}
```

#### 학습 콘텐츠 생성 요청

```http
POST /learning-content/generate
Content-Type: application/json

{
  "subject": "science",
  "topic": "photosynthesis",
  "learner_id": "learner_12345",
  "difficulty_level": "beginner",
  "preferred_format": "visual",
  "target_age": 10
}
```

응답:

```json
{
  "content_id": "content_67890",
  "subject": "science",
  "topic": "photosynthesis",
  "content_blocks": [
    {
      "type": "text",
      "content": "Photosynthesis is the process used by plants to convert light energy into chemical energy...",
      "order": 1
    },
    {
      "type": "image",
      "url": "https://content.aipls.example.com/images/photosynthesis-diagram.png",
      "alt_text": "Diagram showing the process of photosynthesis",
      "order": 2
    },
    // 추가 콘텐츠 블록...
  ],
  "generation_timestamp": "2025-05-07T10:00:00Z"
}
```

### 학습 분석 API

#### 학습자 진행 상황 조회

```http
GET /analytics/learners/{learner_id}/progress
```

응답:

```json
{
  "learner_id": "learner_12345",
  "overall_progress": {
    "completed_paths": 2,
    "active_paths": 1,
    "total_learning_time_hours": 15.5,
    "average_score": 87.3
  },
  "path_progress": [
    {
      "path_id": "path_56789",
      "subject": "mathematics",
      "topic": "algebra",
      "progress_percentage": 35,
      "estimated_completion_date": "2025-05-18T00:00:00Z",
      "performance_trend": "improving"
    },
    // 추가 경로 정보...
  ],
  "skill_mastery": {
    "algebra": 0.75,
    "geometry": 0.6,
    "statistics": 0.85
    // 추가 기술...
  },
  "learning_patterns": {
    "peak_performance_times": ["09:00-11:00", "15:00-16:00"],
    "average_session_duration_minutes": 27,
    "preferred_content_types": ["video", "interactive"]
  }
}
```

#### 학습 활동 로그 전송

```http
POST /analytics/activity-logs
Content-Type: application/json

{
  "learner_id": "learner_12345",
  "unit_id": "unit_2",
  "activity_type": "content_view",
  "content_block_id": "content_block_123",
  "start_time": "2025-05-07T10:15:00Z",
  "end_time": "2025-05-07T10:20:00Z",
  "metadata": {
    "view_percentage": 100,
    "pause_count": 2,
    "replay_count": 1
  },
  "device_info": {
    "type": "tablet",
    "os": "iPadOS 18.2",
    "browser": "Safari"
  }
}
```

응답:

```json
{
  "log_id": "log_abcdef",
  "status": "recorded",
  "timestamp": "2025-05-07T10:20:05Z"
}
```

### 생체 데이터 API

#### 생체 데이터 스트림 시작 (WebSocket)

```http
WebSocket: wss://api.aipls.example.com/v1/biometric-data/stream?learner_id=learner_12345&session_id=session_6789
```

메시지 형식:

```json
{
  "timestamp": "2025-05-07T10:25:12.345Z",
  "data_type": "eeg",
  "channel_count": 4,
  "values": [0.15, 0.23, -0.05, 0.11],
  "sampling_rate": 256
}
```

#### 생체 데이터 일괄 전송

```http
POST /biometric-data/batch
Content-Type: application/json

{
  "learner_id": "learner_12345",
  "session_id": "session_6789",
  "data_type": "heart_rate",
  "start_time": "2025-05-07T10:15:00Z",
  "end_time": "2025-05-07T10:45:00Z",
  "sampling_rate": 1,
  "data": [72, 75, 78, 80, 82, 85, 87, 90, 92, 95, 96, 95, 93, 90, 87, 85, 82, 80, 78, 75, 73, 72, 70, 68, 67, 68, 70, 72, 75, 77]
}
```

응답:

```json
{
  "batch_id": "batch_12345",
  "records_processed": 30,
  "status": "success"
}
```

### GraphQL API

복잡한 데이터 조회를 위한 GraphQL 엔드포인트를 제공합니다:

```http
POST /graphql
Content-Type: application/json

{
  "query": "query GetLearnerDetails($id: ID!) { learner(id: $id) { name email progress { current_path { title subject topic progress_percentage } recommended_next_units { unit_id title difficulty estimated_duration_minutes } skill_mastery { skill level } } } }",
  "variables": {
    "id": "learner_12345"
  }
}
```

## 오류 처리

API는 표준 HTTP 상태 코드를 사용하여 오류를 나타냅니다:

- 200: 성공
- 400: 잘못된 요청 (요청 형식이 잘못되었거나 필수 파라미터가 누락된 경우)
- 401: 인증 실패
- 403: 권한 없음
- 404: 리소스를 찾을 수 없음
- 429: 요청 한도 초과
- 500: 서버 오류

오류 응답 형식:

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "The parameter 'difficulty' must be one of: 'beginner', 'intermediate', 'advanced'",
    "details": {
      "parameter": "difficulty",
      "provided_value": "expert",
      "allowed_values": ["beginner", "intermediate", "advanced"]
    }
  },
  "request_id": "req_12345"
}
```

## 수정사항 및 변경 이력 로깅

모든 주요 API 작업(생성, 삭제, 수정)은 감사 로그에 기록됩니다. 이 로그는 다음 엔드포인트를 통해 조회할 수 있습니다:

```http
GET /audit-logs?resource_type=learner&resource_id=learner_12345&from=2025-05-01T00:00:00Z&to=2025-05-07T23:59:59Z
```

응답:

```json
{
  "logs": [
    {
      "log_id": "audit_12345",
      "timestamp": "2025-05-05T15:00:00Z",
      "action": "update",
      "resource_type": "learner",
      "resource_id": "learner_12345",
      "changes": {
        "preferences.learning_style": {
          "from": "visual",
          "to": "auditory"
        },
        "preferences.session_duration": {
          "from": 25,
          "to": 30
        }
      },
      "performed_by": "api_key_user_67890",
      "ip_address": "203.0.113.42"
    },
    // 추가 로그...
  ],
  "total_count": 5,
  "page": 1,
  "page_size": 10
}
```

## 요청 제한

API 사용량을 조절하기 위해 요청 제한이 적용됩니다:

- 기본 플랜: 분당 60 요청, 일일 5,000 요청
- 프리미엄 플랜: 분당 300 요청, 일일 30,000 요청

현재 사용량은 응답 헤더에서 확인할 수 있습니다:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 58
X-RateLimit-Reset: 1620396060
```

## API 클라이언트

다양한 언어로 제공되는 API 클라이언트 라이브러리:

- Python: `pip install aipls-client`
- JavaScript: `npm install aipls-client`
- Java: Maven이나 Gradle을 통해 `com.example.aipls:aipls-client:1.0.0`
- PHP: `composer require aipls/aipls-client`

## API 버전 관리

API는 URL에 버전을 포함하여 버전을 관리합니다(예: `/v1/learners`). 주요 변경사항이 있을 경우 새로운 버전이 출시됩니다.

## 지원 및 피드백

API 관련 지원 및 피드백은 다음 채널을 통해 제공됩니다:

- 개발자 포럼: https://developers.aipls.example.com/forum
- 이메일: api-support@aipls.example.com
- 문서 저장소: https://github.com/aipls/api-docs