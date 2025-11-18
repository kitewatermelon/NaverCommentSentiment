from collections import Counter
from konlpy.tag import Okt
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")  # 화면에 띄우지 않고 파일로 저장만
import matplotlib.pyplot as plt


def generate_wordcloud(df, text_col, save_dir=None):

    okt = Okt()
    # 모든 댓글 명사 추출
    noun_list = []
    for text in df[text_col]:
        noun_list.extend(okt.nouns(str(text)))

    counter = Counter(noun_list)
    wc = WordCloud(
        font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        width=800, height=400, background_color='white'
    ).generate_from_frequencies(counter)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_dir+"/wordcloud.png", bbox_inches='tight')