# 🧠 AI 기반 개인 맞춤형 학습 경로 추천 시스템

<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/banner.svg" alt="AI Learning System Banner" width="800"/>
</p>

## 🚀 시스템 개요

이 시스템은 생체신호 분석과 AI 기술을 활용하여 학습자 개인별 특성, 목표, 학습 스타일, 실시간 인지 상태에 기반한 맞춤형 학습 경험을 제공합니다. 학습 효과를 최대화하기 위해 최적화된 학습 경로를 추천하고, 실시간으로 콘텐츠를 조정하며, 설명 가능한 AI 기능으로 시스템의 결정을 사용자에게 투명하게 제공합니다.

<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/system_overview.svg" alt="System Overview" width="700"/>
</p>

## ✨ 핵심 기능

### 🎯 다중 모달 데이터 수집
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/multimodal_data.svg" alt="Multimodal Data Collection" width="600"/>
</p>

- **웨어러블 생체 센서**: EEG(뇌파), 심박수, 안구 추적 등 다양한 생체 신호를 실시간 수집
- **학습 활동 데이터**: 상호작용 패턴, 선호도, 반응 시간 분석
- **귀 삽입물 생체 센서**: 초소형 센서로 편리하게 실시간 생체 신호 측정

### 🤖 AI 기반 개인화
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/ai_personalization.svg" alt="AI-Driven Personalization" width="600"/>
</p>

- **강화학습 기반 경로 최적화**: 학습자 상태와 학습 목표에 따른 최적 학습 경로 생성
- **지식 그래프 모델링**: 학습 개념 간 관계를 그래프로 모델링하여 효과적인 학습 순서 구성
- **개인화된 예측 모델**: 학습자별 인지 상태와 성과를 예측하는 맞춤형 AI 모델

### 🔄 동적 학습 경로 적응
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/adaptive_path.svg" alt="Dynamic Learning Path" width="600"/>
</p>

- **실시간 경로 조정**: 학습 진행 상황과 인지 상태에 따라 학습 경로를 동적으로 조정
- **마이크로 러닝 경로**: 작은 학습 단위로 구성된 적응형 경로 생성
- **대안 경로 제안**: 다양한 학습 방식과 콘텐츠 유형을 포함한 대안 경로 제시

### 📚 콘텐츠 개인화
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/content_personalization.svg" alt="Content Personalization" width="600"/>
</p>

- **학습 스타일 기반 조정**: 시각/언어적, 능동/반성적 등 학습 스타일에 맞게 콘텐츠 조정
- **AI 생성 콘텐츠**: 개인 수준에 맞는 맞춤형 콘텐츠 실시간 생성
- **난이도 동적 조절**: 학습자의 이해도와 인지 부하에 맞춰 콘텐츠 난이도 조절

### 📊 설명 가능한 AI (XAI)
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/explainable_ai.svg" alt="Explainable AI" width="600"/>
</p>

- **추천 근거 시각화**: AI의 추천 결정 과정을 시각적으로 표현
- **자연어 설명 생성**: 학습 경로와 적응 결정에 대한 이해하기 쉬운 설명 제공
- **투명한 모델 결정**: SHAP 값과 주의력 시각화를 통한 모델 결정 투명성 확보

### 🔒 보안 및 개인정보 보호
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/security_privacy.svg" alt="Security and Privacy" width="600"/>
</p>

- **동형 암호화**: 암호화된 상태로 데이터 처리 가능
- **차등 프라이버시**: 개인정보 보호와 데이터 유용성의 균형 유지
- **블록체인 기반 무결성**: 학습 기록의 무결성을 블록체인으로 보장
- **엣지 컴퓨팅 보호**: 민감한 생체 데이터를 엣지 장치에서 직접 처리

### 🧠 인지 상태 모니터링
<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/cognitive_monitoring.svg" alt="Cognitive State Monitoring" width="600"/>
</p>

- **실시간 집중도 추적**: 생체 신호를 통한 집중도와 참여도 실시간 모니터링
- **인지 부하 측정**: 작업 기억 용량과 정보 처리 부하 측정
- **최적 학습 구간 예측**: 개인별 최적 학습 시간대와 기간 예측

## 🏛️ 시스템 아키텍처

<p align="center">
  <img src="https://raw.githubusercontent.com/JJshome/ai-personalized-learning-system/main/assets/architecture.svg" alt="System Architecture" width="800"/>
</p>

시스템은 다음과 같은 통합된 구성 요소로 이루어져 있습니다:

1. **데이터 수집 서브시스템**
   - 웨어러블 기기에서 생체 센서 데이터 수집
   - 학습 활동 추적 및 처리
   - 엣지 AI를 통한 개인정보 보호 데이터 분석

2. **AI 분석 및 경로 추천**
   - 지식 상태, 학습 스타일, 인지 프로필을 포착하는 학습 모델
   - 개인화된 학습 시퀀스를 생성하는 경로 생성기
   - 경로 최적화를 위한 강화 학습 알고리즘

3. **콘텐츠 관리 및 적응**
   - 콘텐츠 저장소 통합
   - 학습자 요구에 기반한 적응형 콘텐츠 제공
   - 실시간 콘텐츠 수정

4. **설명 가능한 AI (XAI) 모듈**
   - 시스템 결정에 대한 설명 생성
   - 인사이트를 위한 시각화 생성
   - 모델 투명성 및 신뢰 구축 기능

5. **보안 및 개인정보 보호 모듈**
   - 동형 암호화 및 차등 프라이버시 기술
   - 블록체인 기반 데이터 무결성
   - 엣지 컴퓨팅 프라이버시 보호

## 🛠️ 기술 상세

### 첨단 기술 활용

- **귀 삽입물 생체 센서**: EEG, 심박수 등 생체 신호를 수집하는 초소형 센서
- **엣지 AI 칩**: 개인정보 보호와 저지연을 위한 로컬 데이터 처리
- **강화 학습**: 개인 요구에 기반한 학습 경로 최적화
- **연합 학습**: 개인정보를 보호하면서 모델 개선 가능
- **트랜스포머 모델**: 시계열 학습 데이터 시퀀스 처리
- **지식 그래프**: 학습 개념 간 연결 표현
- **설명 가능한 AI 알고리즘**: SHAP 값, 주의력 시각화, 자연어 설명

### 구현 참고사항

시스템은 모듈식 아키텍처로 유연한 배포와 사용자 지정이 가능합니다:

- Python 기반 구현으로 깔끔하고 잘 문서화된 코드
- 최첨단 머신러닝 라이브러리 활용
- 설계부터 보안 및 개인정보 보호 고려
- 포괄적인 API 문서 제공

## 📊 활용 분야

- **K-12 교육**: 모든 연령대 학생을 위한 맞춤형 학습 경로
- **고등 교육**: 대학 과정 적응 및 학습 경로 최적화
- **기업 교육**: 기술 개발 및 전문 교육
- **평생 학습**: 성인을 위한 자기 주도적 교육
- **특수 교육**: 특별한 요구를 가진 학습자를 위한 맞춤형 접근
- **원격 학습**: 원격 교육을 위한 향상된 참여도

## 🚀 시작하기

### 필요 조건

- Python 3.9+
- 필수 패키지 (requirements.txt 참조)
- 호환 가능한 생체 센서 기기 (선택 사항)

### 설치

1. 저장소 복제:
```bash
git clone https://github.com/JJshome/ai-personalized-learning-system.git
cd ai-personalized-learning-system
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 시스템 구성:
```bash
cp config.example.json config.json
# config.json 수정
```

4. 시스템 실행:
```bash
python src/main.py
```

## 📚 문서

더 자세한 정보는 다음 문서를 참조하세요:

- [시스템 아키텍처](docs/architecture.md)
- [API 참조](docs/api.md)
- [사용자 가이드](docs/user_guide.md)
- [개발자 가이드](docs/developer_guide.md)
- [생체 센서 통합](docs/biosensors.md)

## 🔬 연구 배경

이 시스템은 교육 공학, 인지 과학, 인공지능 분야의 최첨단 연구를 기반으로 합니다. 주요 연구 영역은 다음과 같습니다:

- 개인화 학습 및 적응형 교육 시스템
- 인지 부하 이론 및 주의력 관리
- 교육에서의 기계 학습 응용
- 교육 기술을 위한 설명 가능한 AI
- 인지 상태 평가를 위한 생체 데이터 분석

## 🤝 기여하기

커뮤니티의 기여를 환영합니다! 자세한 내용은 [기여 가이드라인](CONTRIBUTING.md)을 참조하세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- 교육 기술 연구 커뮤니티
- 오픈 소스 AI 및 ML 라이브러리
- 기여자 및 테스터
