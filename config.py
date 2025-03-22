from attrs import define
import torch


@define
class BaseConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
