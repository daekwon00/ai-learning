"""
Week 1 실습: Python ML 기초 - 환경 테스트
작성일: 2026.01.17
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def test_environment():
    """개발 환경 테스트"""
    print("=== 개발 환경 테스트 ===")
    print(f"NumPy 버전: {np.__version__}")
    print(f"Pandas 버전: {pd.__version__}")
    print("\n환경 설정 완료! ✅\n")

def iris_classification_example():
    """
    붓꽃 분류 예제
    - 데이터 로드
    - Train/Test 분할
    - 모델 학습
    - 평가
    """
    print("=== 붓꽃 분류 예제 ===")
    
    # 1. 데이터 로드
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"데이터셋 크기: {X.shape}")
    print(f"클래스 개수: {len(iris.target_names)}")
    print(f"특성 이름: {iris.feature_names}\n")
    
    # 2. Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"훈련 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}\n")
    
    # 3. 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. 예측 및 평가
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"정확도: {accuracy:.2%}\n")
    print("분류 리포트:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 5. 특성 중요도 시각화
    feature_importance = pd.DataFrame({
        'feature': iris.feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n특성 중요도:")
    print(feature_importance)
    
    # 6. 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('중요도')
    plt.title('특성 중요도')
    plt.tight_layout()
    plt.savefig('../notebooks/feature_importance.png')
    print("\n차트 저장: notebooks/feature_importance.png")

def main():
    """메인 실행 함수"""
    test_environment()
    iris_classification_example()
    print("\n✅ Week 1 기본 실습 완료!")
    print("다음 단계: Jupyter Notebook에서 더 자세한 분석 진행")

if __name__ == "__main__":
    main()
