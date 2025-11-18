# Naver News Comment Sentiment & EDA

댓글 데이터를 대상으로 감정 분석과 EDA(Exploratory Data Analysis)를 수행하는 파이썬 프로젝트입니다.  
KoELECTRA 기반 감정 분석 모델을 사용하여 6가지 감정 확률을 계산하고, PCA / t-SNE 시각화, 워드클라우드, 댓글 길이 분석, 상관관계 분석 등 다양한 기능을 제공합니다.

---

## 🔹 주요 기능

1. **감정 분석 (6-class)**
   - 모델: `Jinuuuu/KoELECTRA_fine_tunning_emotion`
   - 감정 라벨: `angry`, `anxious`, `embarrassed`, `happy`, `heartache`, `sad`
   - 기존 CSV 파일이 있으면 재분석 없이 로드 가능
   - 감정 확률 기반 PCA / t-SNE 시각화 지원

2. **원본 텍스트 임베딩 기반 PCA / t-SNE**
   - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 사용 가능
   - 다른 데이터셋 간 표현 차이를 시각화 가능

3. **EDA**
   - 댓글 길이 분석
   - 상관계수 히트맵
   - 감정 확률 분포 히스토그램
   - 워드클라우드 생성

4. **데이터 필터링**
   - 시작일(`START_DATE`)과 기간(`WEEKS`) 설정 가능
   - 지정 기간만 분석 및 시각화 가능

5. **댓글 수집**
   - 네이버 뉴스 댓글 엑셀 파일 → CSV로 자동 수집
   - Playwright 기반 비동기 수집으로 속도 향상

## 🔹 파일 구조
```
project_root/
├─ main.py # 전체 파이프라인 실행 스크립트            
├─ comment_scraper.py # 네이버 뉴스 댓글 수집 모듈           
├─ emotion_analysis.py # 감정 분석 모듈           
├─ eda.py # 댓글 길이, 분포, 상관관계, PCA/TSNE 시각화 모듈           
├─ wordcloud_analysis.py # 워드클라우드 생성 모듈           
├─ config.json # 설정값 (입력 파일, 기간, 분석 옵션 등)           
├─ requirements.txt # 필요한 패키지           
└─ plots/ # 생성된 시각화 이미지 저장 폴더           
```


## 🔹 설치 및 환경
🔹 실행 방법
```bash
# Python 3.9 이상 권장
pip install -r requirements.txt
playwright install
python main.py --config /path/to/config.json
```

🔹 config.json 예시:
```
{
  "EXCEL_PATH": "data/박근혜탄핵_네이버뉴스.xlsx",
  "COMMENT_CSV": "data/박근혜탄핵_네이버뉴스_댓글.csv",
  "CSV_OUTPUT": "data/박근혜탄핵_네이버뉴스_댓글_감정분석.csv",
  "COMMENT_COL": "comment",
  "MODEL_NAME": "Jinuuuu/KoELECTRA_fine_tunning_emotion",
  "START_DATE": "2017-03-10",
  "WEEKS": 4,
  "DO_EMOTION_ANALYSIS": false,
  "DO_WORDCLOUD": true,
  "PLOT_DIR": "plots/박근혜"
}
```
---
🔹 시각화 결과

```
상관계수 히트맵
- 댓글 길이와 감정 확률 간 상관관계 확인

분포 히스토그램
- 각 감정 확률 및 댓글 길이 분포

PCA / t-SNE 시각화
- 감정 확률 기반 또는 텍스트 임베딩 기반 군집 확인

워드클라우드
- 자주 등장하는 단어 시각화
```

---
🔹 참고 사항
```
한국어 텍스트 분석을 위해 KoELECTRA와 konlpy 사용

GPU 사용 시 PyTorch 기반 감정 분석 속도 향상 가능

데이터셋 별로 CSV 단위로 분석, 여러 CSV를 병합하지 않고 개별 분석 권장

```
