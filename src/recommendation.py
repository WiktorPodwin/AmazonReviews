import torch
import faiss

import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class RecommendationSystem:
    """
    A class for generating product recommendations based on sentiment and cosine distance of reviews.
    """

    def __init__(self, model_name: str, device: torch.device, df: pd.DataFrame) -> None:
        """
        Args:
            model_name (str): Pretrained model name for generating embeddings.
            device (torch.device): Device to run the model.
            df (pd.DataFrame): DataFrame containing review data.
        """
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.index = None

        self.df = df.copy()
        df["sentiment_score"] = df["sentiment"].map({"POSITIVE": 1, "NEGATIVE": -1})
        self.df = df
        self.text = df["text"].to_list()

    def encode(self, batch_size: int = 16):
        """
        Encodes review text into embeddings and creates a FAISS index for similarity search.

        Args:
            batch_size (int): Number of samples to process at a time.
        """
        embeddings = []

        for i in tqdm(
            range(0, len(self.text), batch_size),
            desc="Processing batches",
            unit="batch",
        ):
            batch = self.text[i : i + batch_size]

            batch_embeddings = self.model.encode(
                batch,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device,
            )

            embeddings.extend(batch_embeddings.cpu().numpy())

        self.df["embeddings"] = embeddings
        embeddings = np.array(embeddings)

        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def recommend(self, k: int = 200) -> pd.DataFrame:
        """
        Recommends products to users based on the cosine distance between their reviews and other reviews.

        Args:
            k (int): The number of candidate reviews to retrieve per user.

        Returns:
            pd.DataFrame: DataFrame with user IDs, recommended products, and similar reviews.
        """
        users = self.df["user_id"].unique()
        recommondations = {}

        for user in users:
            user_reviews = self.df[self.df["user_id"] == user]

            k_results = max(min(len(user_reviews) + k, len(self.df)), 3)

            purchased_products = user_reviews["product_id"].unique()
            user_embedding = np.mean(
                np.stack(user_reviews["embeddings"]), axis=0, keepdims=True
            )
            faiss.normalize_L2(user_embedding)

            coefficients, indices = self.index.search(user_embedding, k=k_results)

            top_reviews = self.df.iloc[indices[0]]
            top_reviews = top_reviews.copy()

            top_reviews["cosine_distance"] = 1 - coefficients.flatten()

            top_reviews = top_reviews[
                ~top_reviews["product_id"].isin(purchased_products)
            ]
            top_reviews["ranking_score"] = (
                top_reviews["sentiment_score"] * top_reviews["cosine_distance"]
            )

            product_scores = top_reviews.groupby(by="product_id")["ranking_score"].sum()

            best_product = product_scores.idxmin() if not product_scores.empty else None

            sel_product_revs = top_reviews[top_reviews["product_id"] == best_product]

            positive_revs = sel_product_revs[
                sel_product_revs["sentiment"] == "POSITIVE"
            ]
            positive_revs = positive_revs.sort_values(
                by="cosine_distance", ascending=True
            )

            negative_revs = sel_product_revs[
                sel_product_revs["sentiment"] == "NEGATIVE"
            ]
            negative_revs = negative_revs.sort_values(
                by="cosine_distance", ascending=True
            )

            selected_revs = positive_revs.head(3)

            if len(selected_revs) < 3:
                remaining_revs = 3 - len(selected_revs)
                selected_revs = pd.concat(
                    [selected_revs, negative_revs.head(remaining_revs)]
                )

            recommondations[user] = {
                "recommendation": best_product,
                "similar_id": selected_revs.index.to_list(),
                "similar_val": selected_revs["cosine_distance"].values,
            }

        recommendation_df = pd.DataFrame.from_dict(recommondations, orient="index")
        recommendation_df.reset_index(inplace=True)
        recommendation_df.rename(columns={"index": "user_id"}, inplace=True)

        return recommendation_df
