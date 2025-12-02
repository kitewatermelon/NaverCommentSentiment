import asyncio
import pandas as pd
import re
from pathlib import Path
from playwright.async_api import async_playwright
from tqdm.asyncio import tqdm_asyncio

# -------------------------
# URL에서 news_id 추출
# -------------------------
def extract_news_id(url):
    match = re.search(r"/article/comment/(\d+)/(\d+)", url)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return "unknown"

# -------------------------
# URL별 댓글 + 작성일 수집 (재시도 포함)
# -------------------------
async def scrape_comments_from_url(context, url, retries=2):
    # entertain.naver.com URL은 건너뛰기
    if "entertain.naver.com" in url:
        print(f"⚡ 스킵: {url}")
        return None, None

    for attempt in range(retries):
        page = None
        try:
            page = await context.new_page()
            await page.goto(url, timeout=20000)
            await page.wait_for_selector("ul.u_cbox_list, span.u_cbox_contents_none", timeout=10000)

            # 기사 제목 추출
            title_el = await page.query_selector(".media_end_head_info_original_article_text")
            title = (await title_el.inner_text()).strip() if title_el else "unknown"

            # 언론사 추출
            press_el = await page.query_selector(".feed_title_point")
            press = (await press_el.inner_text()).strip() if press_el else "unknown"

            news_id = extract_news_id(url)
            article = {"news_id": news_id, "title": title, "press": press, "url": url}

            # 댓글 없는 경우 처리
            no_comment = await page.query_selector("span.u_cbox_contents_none")
            if no_comment and "등록된 댓글이 없습니다." in (await no_comment.inner_text()):
                await page.close()
                return article, []

            # "더보기" 클릭 반복
            while True:
                paginate_div = await page.query_selector("div.u_cbox_paginate")
                if not paginate_div:
                    break
                style = await paginate_div.get_attribute("style") or ""
                if "display: none" in style:
                    break
                more_btn = await paginate_div.query_selector("span.u_cbox_page_more")
                if not more_btn:
                    break
                await more_btn.click()
                await page.wait_for_timeout(500)

            # 댓글 수집
            comments = []
            comment_elements = await page.query_selector_all("ul.u_cbox_list li")
            for el in comment_elements:
                try:
                    text_el = await el.query_selector(".u_cbox_contents")
                    date_el = await el.query_selector(".u_cbox_date")
                    nick_el = await el.query_selector(".u_cbox_nick")

                    text = (await text_el.inner_text()).strip() if text_el else ""
                    date = (await date_el.inner_text()).strip() if date_el else ""
                    username = (await nick_el.inner_text()).strip() if nick_el else ""

                    if text:
                        comments.append({
                            "news_id": news_id,
                            "username": username,
                            "comment": text,
                            "datetime": date
                        })
                except:
                    continue

            await page.close()
            return article, comments

        except Exception as e:
            if page:
                await page.close()
            print(f"❌ {url} 시도 {attempt+1}/{retries} 실패: {e}")
            await asyncio.sleep(2)

    # 모든 재시도 실패 시 빈 데이터
    news_id = extract_news_id(url)
    return {"news_id": news_id, "title": "unknown", "press": "unknown", "url": url}, []

# -------------------------
# 엑셀 기반 병렬 처리 + CSV 저장
# -------------------------
async def scrape_comments_from_excel_parallel_to_csv(excel_path, articles_csv, comments_csv, max_concurrency=3, context=None):
    df = pd.read_excel(excel_path)
    if "link" not in df.columns:
        print(f"❌ 엑셀에 'link' 열이 없습니다: {excel_path}")
        return

    urls = df["link"].dropna().tolist()
    print(f"총 {len(urls)}개 URL에서 댓글 수집 시작 → {excel_path}")

    articles_csv = Path(articles_csv)
    comments_csv = Path(comments_csv)
    articles_csv.parent.mkdir(parents=True, exist_ok=True)
    comments_csv.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def sem_task(url):
        async with semaphore:
            return await scrape_comments_from_url(context, url)

    tasks = [sem_task(url) for url in urls]

    for coro in tqdm_asyncio.as_completed(tasks):
        article, comments = await coro
        if article is None:
            continue  # entertain.naver.com URL은 CSV에 저장 안함

        # article CSV 저장 (중복 방지)
        df_article = pd.DataFrame([article])
        df_article.to_csv(
            articles_csv, mode='a', index=False,
            header=not articles_csv.exists(), encoding="utf-8-sig"
        )

        # comments CSV 저장
        if comments:
            df_comments = pd.DataFrame(comments)
            df_comments.to_csv(
                comments_csv, mode='a', index=False,
                header=not comments_csv.exists(), encoding="utf-8-sig"
            )

        print(f"✅ {article['news_id']} 처리 완료: 댓글 {len(comments)}개")

# -------------------------
# 메인 실행
# -------------------------
async def main():
    excel_files = [
        ("data/박근혜탄핵_네이버뉴스.xlsx", "output/park_articles.csv", "output/park_comments.csv"),
        ("data/윤석열탄핵_네이버뉴스.xlsx", "output/yoon_articles.csv", "output/yoon_comments.csv"),
    ]

    Path("output").mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        for excel_path, articles_csv, comments_csv in excel_files:
            await scrape_comments_from_excel_parallel_to_csv(
                excel_path, articles_csv, comments_csv, max_concurrency=1, context=context
            )

        await browser.close()
    print("🎉 전체 완료")

if __name__ == "__main__":
    asyncio.run(main())
