import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

def add_comment_length(df, comment_col):
    df['comment_length'] = df[comment_col].astype(str).apply(len)
    return df

def correlation_plot(df, cols, save_path=None):
    corr_matrix = df[cols].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink":0.8})
    plt.title("Correlation matrix")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print("저장 완료:", save_path)
    plt.close()

def distribution_plot(df, cols, save_dir=None):
    plt.figure(figsize=(16,10))
    for i, col in enumerate(cols):
        plt.subplot(2, 4, i+1)
        sns.histplot(df[col], kde=True, bins=30, color='skyblue')
        plt.title(col)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    if save_dir:
        plt.savefig(f"{save_dir}/distribution_plot.png", bbox_inches='tight', dpi=300)
        print("저장 완료:", f"{save_dir}/distribution_plot.png")
    plt.close()

def pca_tsne_plot(df, emotion_cols, save_dir=None):
    X = df[emotion_cols].values
    X_scaled = StandardScaler().fit_transform(X)
    df['predicted'] = df[emotion_cols].idxmax(axis=1)

    # 레이블 고정 색상
    palette_fixed = {
        'angry': '#E74C3C',
        'anxious': '#F39C12',
        'embarrassed': '#9B59B6',
        'happy': '#2ECC71',
        'heartache': '#3498DB',
        'sad': '#95A5A6'
    }

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['predicted'], palette=palette_fixed, alpha=0.7)
    plt.title("PCA of Emotion Probabilities")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Predicted Emotion')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/pca_plot.png", bbox_inches='tight', dpi=300)
        print("저장 완료:", f"{save_dir}/pca_plot.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df['predicted'], palette=palette_fixed, alpha=0.7)
    plt.title("t-SNE of Emotion Probabilities")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Predicted Emotion')
    if save_dir:
        plt.savefig(f"{save_dir}/tsne_plot.png", bbox_inches='tight', dpi=300)
        print("저장 완료:", f"{save_dir}/tsne_plot.png")
    plt.close()

def text_embedding_pca_tsne(df, text_col, save_dir=None, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """
    원본 텍스트 임베딩 기반 PCA/t-SNE 시각화
    """
    # 1. 임베딩 모델 로드
    model = SentenceTransformer(model_name)

    # 2. 임베딩 계산
    texts = df[text_col].astype(str).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # 3. 표준화
    X_scaled = StandardScaler().fit_transform(embeddings)

    # 4. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], alpha=0.6)
    plt.title("PCA of Text Embeddings")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/text_embedding_pca.png", bbox_inches='tight', dpi=300)
        print("저장 완료:", f"{save_dir}/text_embedding_pca.png")
    plt.close()

    # 5. t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], alpha=0.6)
    plt.title("t-SNE of Text Embeddings")
    if save_dir:
        plt.savefig(f"{save_dir}/text_embedding_tsne.png", bbox_inches='tight', dpi=300)
        print("저장 완료:", f"{save_dir}/text_embedding_tsne.png")
    plt.close()
