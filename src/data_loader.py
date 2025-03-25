import gzip

import pandas as pd


class ReviewCleaner:
    """
    A class for data extraction and preprocessing.
    """

    def __init__(self, filename: str) -> None:
        """
        Args:
            filename (str): The path to the gzipped file containing product review data.
        """
        self.filename = filename
        self.reviews = pd.DataFrame()

    def _parse(self):
        """
        Parses the gzipped file into a pandas DataFrame.

        Yields:
            dict: A dictionary representing a product review entry.
        """
        f = gzip.open(self.filename, "rt", encoding="utf-8")
        entry = {}
        for line in f:
            line = line.strip()
            colon_pos = line.find(":")
            if colon_pos == -1:
                yield entry
                entry = {}
                continue
            e_name = line[:colon_pos]
            rest = line[colon_pos + 2 :]
            entry[e_name] = rest
        yield entry

    def _convert_to_fraction(self, frac_str: str) -> float:
        """
        Converts a fraction string (e.g. '3/5') to its decimal value.

        Args:
            frac_str (str): Fraction to convert, in the format 'numerator/denominator'.

        Returns:
            float: Decimal value of the fraction.
        """
        try:
            numerator, denominator = frac_str.split("/")
            numerator = float(numerator)
            denominator = float(denominator)
            if denominator == 0:
                return 0.0
            return numerator / denominator
        except:
            return 0.0

    def load_reviews(self):
        """
        Loads and parses the review data into a pandas DataFrame, converting data types.
        """
        self.reviews = pd.DataFrame(list(self._parse()))
        self.reviews.columns = [
            "product_id",
            "product_title",
            "product_price",
            "user_id",
            "profile_name",
            "helpfulness",
            "score",
            "time",
            "summary",
            "text",
        ]
        self.reviews = self.reviews.astype(
            {
                "product_id": str,
                "product_title": str,
                "user_id": str,
                "profile_name": str,
                "helpfulness": str,
                "score": float,
                "summary": str,
                "text": str,
            }
        )

        self.reviews["product_price"] = pd.to_numeric(
            self.reviews["product_price"], errors="coerce"
        ).astype("float64")
        self.reviews["time"] = pd.to_numeric(
            self.reviews["time"], errors="coerce"
        ).astype("Int64")

        self.reviews["helpfulness"] = self.reviews["helpfulness"].apply(
            self._convert_to_fraction
        )

    def clean_reviews(self) -> pd.DataFrame:
        """
        Cleans the review data by removing missing values, duplicates and irrelevant data.

        Returns:
            pd.DataFrame: A preprocessed DataFrame with cleaned review data.
        """
        self.reviews = self.reviews.dropna(
            subset=[
                "product_id",
                "product_title",
                "user_id",
                "profile_name",
                "helpfulness",
                "score",
                "time",
                "summary",
                "text",
            ]
        )

        no_dupes = self.reviews.drop_duplicates()

        cleaned_users = no_dupes[no_dupes["user_id"] != "unknown"]

        sorted_reviews = cleaned_users.sort_values(
            by=["user_id", "product_id", "time"], ascending=[True, True, False]
        )
        cleaned_df = sorted_reviews.drop_duplicates(
            subset=["user_id", "product_id"], keep="first"
        )

        return cleaned_df
