# Week 3: BERT + NLP 심화 (56시간)

> **목표:** Week 2에서 배운 BERT를 실전 금융 프로젝트에 심화 적용

## 📅 주차 일정

### Day 15-17 (월-수): BERT 고급 기법
**학습 시간:** 24시간
- Multi-task Learning
- Domain Adaptation
- Model Optimization

### Day 18-21 (목-일): 실전 금융 NLP 프로젝트
**학습 시간:** 32시간
- 금융 문서 분류 시스템
- 리스크 평가 자동화
- 규제 문서 분석

## 🎯 학습 목표

### 핵심 역량
- ✅ BERT 고급 기법 (Multi-task, Domain Adaptation)
- ✅ 금융 도메인 NLP 파이프라인
- ✅ 모델 최적화 (Quantization, Pruning)
- ✅ 프로덕션 레벨 코드

### 완성 프로젝트
- **금융 문서 통합 분류 시스템**
- **계약서 리스크 자동 평가**
- **규제 준수 분석 도구**

## 💡 Week 2 복습 + 심화

Week 2에서 배운 내용을 더욱 심화하고 실전에 적용하는 주차입니다.

### Week 2 핵심 복습
1. BERT Fine-tuning 과정
2. Hugging Face Trainer API
3. Extractive QA 원리
4. 모델 평가 메트릭

### Week 3 심화 내용
1. **Multi-task Learning:** 하나의 모델로 여러 태스크 수행
2. **Domain Adaptation:** 금융 도메인에 특화된 BERT
3. **Model Optimization:** 배포를 위한 경량화
4. **End-to-End Pipeline:** 실전 시스템 구축

## 💻 심화 프로젝트

### Project 8: 금융 문서 통합 분류 시스템

**목표:** 하나의 BERT 모델로 여러 분류 작업 동시 수행

```python
"""
Multi-task Learning BERT
- Task 1: 문서 타입 분류 (대출/보험/투자)
- Task 2: 긍정/부정 분류
- Task 3: 우선순위 분류 (긴급/일반/낮음)
"""

import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, num_doc_types=5, num_sentiments=3, num_priorities=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("klue/bert-base")
        
        # 공유 레이어
        self.dropout = nn.Dropout(0.3)
        
        # Task별 헤드
        self.doc_classifier = nn.Linear(768, num_doc_types)
        self.sentiment_classifier = nn.Linear(768, num_sentiments)
        self.priority_classifier = nn.Linear(768, num_priorities)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        
        doc_logits = self.doc_classifier(pooled)
        sentiment_logits = self.sentiment_classifier(pooled)
        priority_logits = self.priority_classifier(pooled)
        
        return {
            'doc_type': doc_logits,
            'sentiment': sentiment_logits,
            'priority': priority_logits
        }

# Multi-task Loss
def multi_task_loss(outputs, labels, weights={'doc': 1.0, 'sent': 0.5, 'pri': 0.5}):
    loss_fn = nn.CrossEntropyLoss()
    
    doc_loss = loss_fn(outputs['doc_type'], labels['doc_type'])
    sent_loss = loss_fn(outputs['sentiment'], labels['sentiment'])
    pri_loss = loss_fn(outputs['priority'], labels['priority'])
    
    total_loss = (
        weights['doc'] * doc_loss +
        weights['sent'] * sent_loss +
        weights['pri'] * pri_loss
    )
    
    return total_loss, {
        'doc_loss': doc_loss.item(),
        'sentiment_loss': sent_loss.item(),
        'priority_loss': pri_loss.item(),
        'total_loss': total_loss.item()
    }
```

---

### Project 9: 계약서 리스크 자동 평가

```python
"""
금융 계약서 리스크 평가 시스템
- 불공정 조항 탐지
- 위험도 점수 산출
- 개선 권장사항 생성
"""

from transformers import pipeline
import re

class ContractRiskAnalyzer:
    def __init__(self):
        # BERT QA 파이프라인
        self.qa_pipeline = pipeline(
            "question-answering",
            model="klue/roberta-large"
        )
        
        # 감성 분석 (위험도 판단)
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="./financial_sentiment_model"
        )
        
        # 위험 키워드
        self.risk_keywords = [
            "위약금", "연체", "강제집행", "담보", 
            "면책", "제한", "금지", "의무"
        ]
    
    def analyze_contract(self, contract_text):
        """계약서 종합 분석"""
        
        # 1. 주요 조항 추출
        clauses = self._extract_clauses(contract_text)
        
        # 2. 각 조항 위험도 분석
        risks = []
        for clause in clauses:
            risk_score = self._calculate_risk(clause)
            if risk_score > 0.6:
                risks.append({
                    'clause': clause,
                    'risk_score': risk_score,
                    'keywords': self._find_risk_keywords(clause)
                })
        
        # 3. 종합 리포트
        report = self._generate_report(risks)
        
        return report
    
    def _calculate_risk(self, text):
        """텍스트 위험도 계산"""
        # 위험 키워드 개수
        keyword_score = sum(1 for kw in self.risk_keywords if kw in text)
        
        # 감성 분석 (부정적일수록 위험)
        sentiment = self.sentiment(text)[0]
        sentiment_score = 1 - sentiment['score'] if sentiment['label'] == 'NEGATIVE' else 0
        
        # 문장 길이 (복잡할수록 위험)
        length_score = min(len(text) / 500, 1.0)
        
        # 종합 점수
        total_score = (
            keyword_score * 0.4 +
            sentiment_score * 0.4 +
            length_score * 0.2
        )
        
        return min(total_score, 1.0)
    
    def _find_risk_keywords(self, text):
        return [kw for kw in self.risk_keywords if kw in text]
    
    def _extract_clauses(self, text):
        """계약서에서 조항 추출"""
        # 번호가 있는 조항 분리
        clauses = re.split(r'\n\s*\d+\.', text)
        return [c.strip() for c in clauses if len(c.strip()) > 20]
    
    def _generate_report(self, risks):
        """리스크 리포트 생성"""
        if not risks:
            return "위험 요소가 발견되지 않았습니다."
        
        report = f"총 {len(risks)}개의 위험 조항이 발견되었습니다.\n\n"
        
        for i, risk in enumerate(risks, 1):
            report += f"{i}. 위험도: {risk['risk_score']:.2f}\n"
            report += f"   조항: {risk['clause'][:100]}...\n"
            report += f"   위험 키워드: {', '.join(risk['keywords'])}\n\n"
        
        return report

# 사용 예시
analyzer = ContractRiskAnalyzer()
contract = """
제1조 (대출 조건)
본 계약의 이자율은 연 15%이며, 연체 시 연 25%의 
지연 이자가 부과됩니다. 3개월 이상 연체 시 
담보물에 대한 강제집행이 가능합니다.
"""

report = analyzer.analyze_contract(contract)
print(report)
```

---

### Project 10: 규제 준수 분석 도구

```python
"""
금융 규제 문서 자동 분석
- 규제 조항 vs 내부 정책 비교
- 위반 가능성 탐지
- 개선 권장사항
"""

class ComplianceAnalyzer:
    def __init__(self):
        self.regulation_db = self._load_regulations()
        self.qa_model = pipeline("question-answering", model="klue/roberta-large")
    
    def check_compliance(self, policy_text, regulation_type="금융소비자보호법"):
        """정책 문서가 규제를 준수하는지 확인"""
        
        # 1. 관련 규제 조항 검색
        regulations = self.regulation_db[regulation_type]
        
        # 2. 각 규제 조항 체크
        violations = []
        for reg in regulations:
            question = f"{reg['requirement']}를 명시하고 있습니까?"
            
            result = self.qa_model(
                question=question,
                context=policy_text
            )
            
            if result['score'] < 0.5:  # 낮은 신뢰도 = 미준수 가능성
                violations.append({
                    'regulation': reg['title'],
                    'requirement': reg['requirement'],
                    'confidence': result['score'],
                    'recommendation': reg['recommendation']
                })
        
        # 3. 리포트 생성
        return self._generate_compliance_report(violations)
    
    def _load_regulations(self):
        """규제 DB 로드 (예시)"""
        return {
            "금융소비자보호법": [
                {
                    'title': '제1조 소비자 정보 제공',
                    'requirement': '금융상품의 주요 내용과 위험사항을 명확히 설명',
                    'recommendation': '상품 설명서에 위험 등급과 주요 내용을 추가하세요'
                },
                {
                    'title': '제2조 불공정 영업행위 금지',
                    'requirement': '허위·과장 정보 제공 금지',
                    'recommendation': '마케팅 문구에서 과장 표현을 제거하세요'
                }
            ]
        }
    
    def _generate_compliance_report(self, violations):
        if not violations:
            return "✅ 모든 규제 조항을 준수하고 있습니다."
        
        report = f"⚠️  {len(violations)}개의 위반 가능성이 발견되었습니다.\n\n"
        
        for i, v in enumerate(violations, 1):
            report += f"{i}. {v['regulation']}\n"
            report += f"   요구사항: {v['requirement']}\n"
            report += f"   신뢰도: {v['confidence']:.2f}\n"
            report += f"   권장사항: {v['recommendation']}\n\n"
        
        return report
```

---

## ✅ Week 3 완료 체크리스트

### 프로젝트 완성도
- [ ] Multi-task BERT 구현 ✅
- [ ] 계약서 리스크 분석 ✅
- [ ] 규제 준수 분석 도구 ✅

### 기술 습득
- [ ] Multi-task Learning
- [ ] Domain Adaptation
- [ ] 실전 NLP 파이프라인
- [ ] 프로덕션 코드 작성

### GitHub
- [ ] 3개 심화 프로젝트 커밋
- [ ] 상세 README 작성
- [ ] 코드 리팩토링 완료

### 다음 주 준비
- [ ] OpenAI API 키 발급
- [ ] LangChain 개념 학습
- [ ] ChromaDB 설치

## 📊 학습 시간 기록

| 일자 | 활동 | 시간 | 완료 |
|------|------|------|------|
| Day 15 | Multi-task Learning | 8h | [ ] |
| Day 16 | 문서 통합 분류 시스템 | 8h | [ ] |
| Day 17 | 모델 최적화 | 8h | [ ] |
| Day 18 | 계약서 리스크 분석 | 8h | [ ] |
| Day 19 | 규제 준수 도구 (1) | 8h | [ ] |
| Day 20 | 규제 준수 도구 (2) | 8h | [ ] |
| Day 21 | 통합 테스트 & 정리 | 8h | [ ] |

---

**Week 3 완료 후 → Week 4 (LangChain + RAG)로 진행**
