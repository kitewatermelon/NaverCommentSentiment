import asyncio
import pandas as pd
import os
import torch
from pathlib import Path
import json
import argparse

from comment_scraper import scrape_comments_from_excel_parallel_to_csv
from emotion_analysis import run_emotion_analysis, emotion_labels
from eda import add_comment_length, correlation_plot, distribution_plot, pca_tsne_plot
from wordcloud_analysis import generate_wordcloud

def main(config_path):
    # ============================================================
    # 설정값 불러오기
    # ============================================================
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    EXCEL_PATH = config["EXCEL_PATH"]
    COMMENT_CSV = config["COMMENT_CSV"]
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
    # 댓글 수집 (엑셀 → CSV)
    # ============================================================
    async def scrape_comments():
        await scrape_comments_from_excel_parallel_to_csv(EXCEL_PATH, output_csv=COMMENT_CSV, max_concurrency=10)

    if not Path(COMMENT_CSV).exists():
        print(f"❗ 댓글 CSV가 없어 엑셀에서 수집합니다: {EXCEL_PATH}")
        asyncio.run(scrape_comments())
    else:
        print(f"✅ 댓글 CSV 이미 존재: {COMMENT_CSV}")

    # ============================================================
    # CSV 불러오기
    # ============================================================
    df = pd.read_csv(COMMENT_CSV)
    print("댓글 수:", len(df))

    # ============================================================
    # 감정 분석
    # ============================================================
    if DO_EMOTION_ANALYSIS:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df = run_emotion_analysis(df, COMMENT_COL, MODEL_NAME, device)
        df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")
        print("감정 분석 완료 및 저장:", CSV_OUTPUT)
    else:
        df = pd.read_csv(CSV_OUTPUT)
        print("기존 감정 분석 CSV 사용:", CSV_OUTPUT)

    # ============================================================
    # 댓글 길이 추가
    # ============================================================
    df = add_comment_length(df, COMMENT_COL)

    # ============================================================
    # 날짜 처리 및 기간 필터링
    # ============================================================
    df['date'] = pd.to_datetime(df['datetime'].str[:10], format='%Y.%m.%d', errors='coerce')
    start_date = pd.to_datetime(START_DATE)
    end_date = start_date + pd.Timedelta(weeks=WEEKS)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    print(f"분석 기간 ({START_DATE} ~ {WEEKS}주) 데이터 수:", len(df_filtered))

    # ============================================================
    # 상관관계 / 분포 / PCA / t-SNE 시각화
    # ============================================================
    cols = ['comment_length'] + emotion_labels
    correlation_plot(df_filtered, cols, save_path=f"{PLOT_DIR}/correlation_matrix.png")
    distribution_plot(df_filtered, cols, save_dir=PLOT_DIR)
    pca_tsne_plot(df_filtered, emotion_labels, save_dir=PLOT_DIR)

    # ============================================================
    # 워드클라우드
    # ============================================================
    if DO_WORDCLOUD:
        generate_wordcloud(df_filtered, COMMENT_COL, save_dir=PLOT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config.json 경로")
    args = parser.parse_args()
    main(args.config)
