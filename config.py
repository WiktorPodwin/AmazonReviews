import torch
import os

from attrs import define


@define
class BaseConfig:
    """
    BaseConfig class holds configuration settings for the project, including file paths,
    model names, and device settings.

    Attributes:
        BASE_DIR (str): The base directory of the current file.
        SOURCE_DATA (str): Path to the source data file (Cell Phones & Accessories dataset).
        SENTIMENT_CSV (str): Path to the sentiment CSV file.
        RECOMMENDATION_CSV (str): Path to the recommendation CSV file.
        DEVICE (torch.device): The device to use for computation.
        SENTIMENT_MODEL (str): The pre-trained sentiment analysis model to use (DistilBERT).
        EMBEDDING_MODEL (str): The pre-trained sentence transformer model for embeddings (SBERT).
    """

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DATA = os.path.join(BASE_DIR, "data/Cell_Phones_&_Accessories.txt.gz")

    SENTIMENT_CSV = os.path.join(BASE_DIR, "data/sentiment.csv")
    RECOMMENDATION_CSV = os.path.join(BASE_DIR, "data/recommendation.csv")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
