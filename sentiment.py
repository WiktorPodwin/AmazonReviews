from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tqdm import tqdm


class SentimentAnalysis:

    def __init__(self, device: torch.device, model_name: str, df: pd.DataFrame) -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        self.df = df
        self.reviews = df["text"].to_list()

    def predict(self, batch_size: int = 16) -> pd.DataFrame:
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
