# Week 1 학습 요약 (Java 개발자 관점)

> 경력 Java 개발자로서 1주차에 파이썬/ML/DL 기초를 빠르게 정리한 기록

---

## 1. 핵심 개념 정리

### 1) Pandas vs NumPy

- **NumPy**: 배열 기반의 고속 수치 연산 라이브러리
  - 벡터 연산으로 Python 루프보다 훨씬 빠름
  - 수치 계산(통계, 선형대수)에 최적화
- **Pandas**: 테이블 형태(DataFrame) 데이터 처리에 특화
  - 실무에서 다루는 CSV/엑셀/DB 결과를 손쉽게 다룸
  - DataFrame은 “행/열 + 인덱스 + 컬럼명” 구조로 관리

### 2) Scikit-learn Pipeline

- 전처리 + 모델 학습을 **하나의 흐름**으로 묶어 관리
- 데이터 유출 방지, 재사용성, 실험 관리에 유리
- 실무에서는 모델 배포 시 전처리 누락을 줄이는 핵심 장치

### 3) PyTorch Tensor vs NumPy array

- **NumPy array**: CPU 기반 수치 연산
- **PyTorch Tensor**: NumPy + GPU 가속 + 자동미분(Autograd)
- 딥러닝 학습의 핵심은 Tensor + Autograd + Optimizer 조합

### 4) CNN 작동 원리

- **Conv2D**로 지역 특징 추출 → **ReLU** → **Pooling**으로 축약
- 반복적으로 특징을 추출한 뒤 **Fully Connected**로 분류
- 이미지 데이터에서 “패턴을 찾아내는” 구조적 장점이 있음

---

## 2. Java와의 비교

### 1) DataFrame vs List<Map>

- DataFrame ≈ `List<Map<String, Object>>`에 인덱스/컬럼 메타데이터가 붙은 형태
- SQL 결과를 자바로 받는 것보다 훨씬 직관적이고 간결함

### 2) Pipeline vs Builder Pattern

- Pipeline은 여러 단계를 **체이닝**해 하나의 객체로 구성
- Builder Pattern처럼 “설정 → 조립 → 실행” 흐름이 동일

### 3) 함수형 프로그래밍 관점

- Pandas/NumPy의 벡터 연산은 Java Stream의 map/reduce와 유사
- 차이는 “루프를 직접 돌리지 않고 내부 최적화된 연산”을 사용한다는 점

---

## 3. 실무 적용 방안

### 1) 금융 데이터 분석

- 주가/금리/거래량 데이터를 Pandas로 전처리 → 통계/지표 계산
- 리스크/수익률/변동성 지표로 투자 보고서 작성 가능

### 2) 신용평가 자동화

- Scikit-learn Pipeline으로 전처리부터 모델 학습까지 일관 관리
- 규제 대응을 고려해 Logistic Regression + 트리 모델 병행 검증

### 3) 이미지 분류 활용 사례

- OCR, 신분증/문서 인식, 이상 탐지 등 금융권 문서 자동화에 적용 가능
- CNN 구조는 이미지 기반 업무 자동화의 핵심

---

## 4. 어려웠던 점과 해결

- **Python 문법 익숙하지 않음**
  - Java 습관으로 구조화하려 했고, 함수 분리로 해결
- **벡터 연산 사고방식 전환**
  - “for 루프 → 벡터 연산”으로 전환하는 데 시간이 걸림
  - 실제 데이터 처리 속도 개선을 확인하며 감각을 익힘
- **딥러닝 학습 루프 이해**
  - `forward → loss → backward → update` 흐름을 반복하며 익힘

---

## 5. 다음 주 학습 계획 (2주차 미리보기)

- PyTorch 심화: CNN 구조 개선, 정규화/스케줄러 적용
- 불균형 데이터 대응: SMOTE, class weight 적용 실습
- 모델 해석 고도화: SHAP/캘리브레이션 기반 설명력 강화
- 금융 도메인 데이터로 실전 미니 프로젝트 진행

---

**정리 한 줄:**  
> “전통적인 Java 개발에서 데이터 기반 의사결정으로 확장하는 1주차였다.”
