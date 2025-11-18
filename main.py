import json
import pandas as pd
from datetime import datetime, timedelta
import torch
import matplotlib.pyplot as plt
import os
import argparse

from emotion_analysis import run_emotion_analysis, emotion_labels
from eda import add_comment_length, correlation_plot, distribution_plot, pca_tsne_plot, text_embedding_pca_tsne
from wordcloud_analysis import generate_wordcloud

# ============================================================
# 1️⃣ argparse로 config 경로 받기
# ============================================================
parser = argparse.ArgumentParser(description="박근혜 탄핵 댓글 감정 분석 및 EDA")
parser.add_argument(
    "--config", type=str, required=True,
    help="설정 파일 경로 (config.json)"
)
args = parser.parse_args()
config_path = args.config

# ============================================================
# 2️⃣ 설정값 불러오기
# ============================================================
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

CSV_INPUT = config["CSV_INPUT"]
CSV_OUTPUT = config["CSV_OUTPUT"]
COMMENT_COL = config["COMMENT_COL"]
MODEL_NAME = config["MODEL_NAME"]
START_DATE = config["START_DATE"]
WEEKS = config["WEEKS"]
DO_EMOTION_ANALYSIS = config["DO_EMOTION_ANALYSIS"]
DO_WORDCLOUD = config["DO_WORDCLOUD"]
PLOT_DIR = config["PLOT_DIR"]

os.makedirs(PLOT_DIR, exist_ok=True)

# ============================================================
# 2️⃣ 데이터 로드 (JSON)
# ============================================================
df = pd.read_csv(CSV_INPUT)
print("로드된 데이터 수:", len(df))

# ============================================================
# 3️⃣ 감정 분석
# ============================================================
if DO_EMOTION_ANALYSIS:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = run_emotion_analysis(df, COMMENT_COL, MODEL_NAME, device)
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
    print("감정 분석 완료 및 저장:", CSV_OUTPUT)
else:
    df = pd.read_csv(CSV_OUTPUT)
    print("기존 감정 분석 CSV 사용")

# ============================================================
# 4️⃣ 댓글 길이 추가
# ============================================================
df = add_comment_length(df, COMMENT_COL)

# ============================================================
# 5️⃣ 날짜 처리 및 기간 필터링
# ============================================================
df['date'] = pd.to_datetime(df['datetime'].str[:10], format='%Y.%m.%d', errors='coerce')
start_date = pd.to_datetime(START_DATE)
end_date = start_date + pd.Timedelta(weeks=WEEKS)
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
print(f"분석 기간 ({START_DATE} ~ {WEEKS}주) 데이터 수:", len(df_filtered))

# ============================================================
# 6️⃣ 상관관계 및 분포 시각화
# ============================================================
# 원본 텍스트 임베딩 PCA/t-SNE
cols = ['comment_length'] + emotion_labels
correlation_plot(df_filtered, cols, save_path=f"{PLOT_DIR}/correlation_matrix.png")
distribution_plot(df_filtered, cols, save_dir=PLOT_DIR)
text_embedding_pca_tsne(df_filtered, COMMENT_COL, save_dir=PLOT_DIR)
pca_tsne_plot(df_filtered, emotion_labels, save_dir=PLOT_DIR)

# ============================================================
# 7️⃣ 워드클라우드
# ============================================================
if DO_WORDCLOUD:
    generate_wordcloud(df_filtered, COMMENT_COL, save_path=f"{PLOT_DIR}/wordcloud.png")
