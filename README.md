# Naver News Comment Sentiment & EDA

댓글 데이터를 대상으로 감정 분석 및 EDA(Exploratory Data Analysis)를 수행하는 파이썬 프로젝트입니다.  
감정 분석 모델 기반 확률값, PCA / t-SNE 시각화, 워드클라우드, 댓글 길이 분석, 상관관계 분석 등 다양한 기능을 제공합니다.

---

## 🔹 주요 기능

1. **감정 분석 (6-class)**
   - 모델: `Jinuuuu/KoELECTRA_fine_tunning_emotion`
   - 감정 라벨: `angry`, `anxious`, `embarrassed`, `happy`, `heartache`, `sad`
   - 기존 CSV 파일이 있으면 재분석 없이 로드 가능
   - 확률 기반 PCA / t-SNE 시각화 지원

2. **원본 텍스트 임베딩 기반 PCA / t-SNE**
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 사용
   - 다른 데이터셋 간 표현 차이를 시각화 가능

3. **EDA**
   - 댓글 길이 분석
   - 상관계수 히트맵
   - 확률 분포 히스토그램
   - 워드클라우드 생성

4. **데이터 필터링**
   - 시작일(`START_DATE`)과 기간(`WEEKS`) 설정 가능
   - 지정 기간만 시각화 및 분석

---

## 🔹 파일 구조


---

## 🔹 설치 및 환경

```
# Python 3.9 이상 권장
pip install -r requirements.txt
python main.py --config /path/to/config.json
```

🔹 시각화

상관계수 히트맵 : 댓글 길이와 감정 확률 간 상관 관계

분포 히스토그램 : 각 감정 확률 및 댓글 길이 분포

PCA / t-SNE : 감정 확률 기반 또는 원본 텍스트 임베딩 기반

워드클라우드 : 자주 등장하는 단어 시각화

🔹 참고

한국어 텍스트 분석을 위해 KoELECTRA 및 konlpy 사용

GPU 사용 시 PyTorch 기반 감정 분석 속도 향상 가능