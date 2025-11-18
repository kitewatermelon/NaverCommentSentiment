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
# URL별 댓글 + 작성일 수집
# -------------------------
async def scrape_comments_from_url(context, url):
    page = await context.new_page()
    comments = []

    try:
        await page.goto(url, timeout=20000)
        await page.wait_for_selector("ul.u_cbox_list, span.u_cbox_contents_none", timeout=10000)

        # 댓글 없는 경우
        no_comment = await page.query_selector("span.u_cbox_contents_none")
        if no_comment:
            text = (await no_comment.inner_text()).strip()
            if "등록된 댓글이 없습니다." in text:
                await page.close()
                return extract_news_id(url), []

        # "더보기" 반복 클릭
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

        # 댓글 + 작성일 수집
        comment_elements = await page.query_selector_all("ul.u_cbox_list li")  # li 단위로 반복
        for el in comment_elements:
            try:
                text_el = await el.query_selector(".u_cbox_contents")
                date_el = await el.query_selector(".u_cbox_date")

                text = (await text_el.inner_text()).strip() if text_el else ""
                date = (await date_el.inner_text()).strip() if date_el else ""

                if text:
                    comments.append({"comment": text, "datetime": date})
            except:
                continue

    except Exception as e:
        print(f"❌ 오류 발생 ({url}): {e}")

    await page.close()
    return extract_news_id(url), comments

# -------------------------
# 엑셀 기반 병렬 처리 + CSV 즉시 저장
# -------------------------
async def scrape_comments_from_excel_parallel_to_csv(excel_path, output_csv="comments.csv", max_concurrency=5):
    df = pd.read_excel(excel_path)
    if "link" not in df.columns:
        print("❌ 엑셀에 'link' 열이 없습니다.")
        return

    urls = df["link"].dropna().tolist()
    print(f"총 {len(urls)}개 URL에서 댓글 수집 시작")

    # CSV 파일 존재 여부 체크
    output_path = Path(output_csv)
    first_write = not output_path.exists()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        semaphore = asyncio.Semaphore(max_concurrency)

        async def sem_task(url):
            async with semaphore:
                return await scrape_comments_from_url(context, url)

        tasks = [sem_task(url) for url in urls]

        for coro in tqdm_asyncio.as_completed(tasks):
            news_id, comments = await coro
            if comments:
                df_save = pd.DataFrame([{
                    "news_id": news_id,
                    "comment": c["comment"],
                    "datetime": c["datetime"]
                } for c in comments])
                df_save.to_csv(output_path, mode='a', index=False,
                               header=first_write, encoding="utf-8-sig")
                first_write = False  # 이후부터는 헤더 없이 append
            print(f"✅ {news_id} 댓글 수집 완료: {len(comments)}개")

        await browser.close()

    print(f"🎉 댓글 저장 완료 → {output_csv}")

# -------------------------
# 여러 엑셀 파일 동시 실행
# -------------------------
async def main():
    tasks = [
        scrape_comments_from_excel_parallel_to_csv(
            r"output\윤석열탄핵_네이버뉴스.xlsx",
            output_csv="output/윤석열탄핵_네이버뉴스_댓글.csv",
            max_concurrency=5
        ),
        scrape_comments_from_excel_parallel_to_csv(
            r"output\박근혜탄핵_네이버뉴스.xlsx",
            output_csv="output/박근혜탄핵_네이버뉴스_댓글.csv",
            max_concurrency=5
        )
    ]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
