import torch

import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalysis:
    """
    A class for performing sentiment analysis.
    """

    def __init__(self, device: torch.device, model_name: str, df: pd.DataFrame) -> None:
        """
        Args:
            device (torch.device): Device to run the model.
            model_name (str): Name of the pretrained model to use.
            df (pd.Dataframe): A DataFrame containing 'text' column for sentiment analysis.
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self.df = df
        self.reviews = df["text"].to_list()

    def predict(self, batch_size: int = 16) -> pd.DataFrame:
        """
        Performs sentiment analysis on the text data.

        Args:
            batch_size (int): Number of samples to process at a time.

        Returns:
            pd.DataFrame: A new DataFrame with sentiment predictions and confidence scores
        """
        labels = ["NEGATIVE", "POSITIVE"]
        results = []

        for i in tqdm(
            range(0, len(self.reviews), batch_size),
            desc="Processing batches",
            unit="batch",
        ):
            batch = self.reviews[i : i + batch_size]

            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

            batch_results = [
                {
                    "sentiment": labels[pred.argmax().item()],
                    "confidence": pred.max().item(),
                }
                for pred in prediction
            ]
            results.extend(batch_results)

        results_df = pd.DataFrame(results)
        merged_df = self.df.reset_index(drop=True).merge(
            results_df, left_index=True, right_index=True
        )

        return merged_df
