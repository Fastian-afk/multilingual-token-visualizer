import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def visualize_embeddings(words, vectors, method="pca"):
    if len(words) < 2:
        raise ValueError("Need at least 2 words to visualize in 2D.")

    vectors = vectors.detach().cpu().numpy()

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=5, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'.")

    reduced = reducer.fit_transform(vectors)

    plt.figure(figsize=(6, 6))
    for i, word in enumerate(words):
        x, y = reduced[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.01, y + 0.01, word, fontsize=12)
    plt.title(f"2D Embedding Projection ({method.upper()})")
    plt.axis("off")
    plt.show()