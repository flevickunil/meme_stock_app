  

import re
import string
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from typing import List, Tuple
import pandas as pd
import ast

class ModelUtils:
    """
    A class to encapsulate utility functions for model operations.
    """

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Purpose:
            To assess the similarity between two vectors

        Arguments:
            vec1: first vector (e.g. word from text data)
            vec2: vector from look-up dictionary 
        
        Output:
            nan if vectors are 0, a similarity rating if between -1 and 1;
                -1 = vectors are diametrically opposed
                0 = no similarity
                1 = the same vector
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return np.nan
        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def evaluate_model_performance(true_tickers: List[str], extracted_tickers: List[str]):
        """
        Purpose:
            Assess model performance by analyzing the predicted tickers from ticker/company extraction models with actual answers from test data to calculate precision and sensitivity. 

        Arguments:
            true_tickers: true tickers found in text
            extracted_tickers: extracted tickers from text by models
        
        Output:
            precision metric: measures the percent by which extracted tickers are relevant or true
            sensitivity metric: measures the percent by which the model can recognise tickers 
        """
        true = Counter(true_tickers)
        extracted = Counter(extracted_tickers)

        # Calculate true positives, false positives, and false negatives
        true_positives = sum((true & extracted).values())
        false_positives = sum((extracted - true).values())
        false_negatives = sum((true - extracted).values())

        # Calculate precision
        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0
        
        # Calculate sensitivity
        if true_positives + false_negatives > 0:
            sensitivity = true_positives / (true_positives + false_negatives)
        else:
            sensitivity = 0
        
        return precision, sensitivity

    @staticmethod
    def load_text_files(filepath):
        """
        Purpose:
            Load text files 

        Arguments:
            filepath: filepath for text files
        
        Output:
            txt file saved into object as string
        """
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    
    @staticmethod
    def convert_comma_separated_string_to_list(variable):
        """
        Purpose:
            Convert a comma-separated string into a list of strings.

        Arguments:
            variable: Comma-separated string
        
        Output:
            List of strings
        """
        # Remove leading and trailing whitespace, then split by comma and strip each element
        list_of_strings = [item.strip() for item in variable.split(',')]
        return list_of_strings

    @staticmethod
    def preprocess_text(text, lower_case = True):
        """
        Purpose:
            Preprocess text to remove URLs, punctuation, line breaks, and stopwords

        Arguments:
            text: The text to preprocess.
            lower_case: Whether to convert the text to lowercase. Defaults to True
        
        Output:
            preprocessed_text: A list of words from the text after preprocessing
        """
        # Using compile because repeated pattern on large number of texts it is more efficient than .finditer or .findall
        url_pattern = re.compile(r"http[s]?\://\S+")

        # Remove punctuation from entire text
        text = re.sub(url_pattern,'',text)
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Convert to lowercase if lower_case is True, this is so that we can identify capitalised tickers as an option
        if lower_case:
            text = text.lower()
        
        # replace new line in reddit text with space, so that text1.split() will handle words better
        text = text.replace("/n", ' ')

        words = word_tokenize(text)

        # Compile non-alphanumeric pattern
        non_word_num = re.compile(r'\W+')
        
        # Remove non-alphanumeric characters from each word
        words = [re.sub(non_word_num, '', word) for word in words]
        
        # Remove empty strings resulting from previous step
        words = [word for word in words if word]
        stop_words = set(stopwords.words('english'))

        #Check if each word is in stop_words and remove if they are
        preprocessed_text = [word for word in words if word.lower() not in stop_words]
        return preprocessed_text

    @staticmethod
    def clean_titles(title):
        """
        Purpose:
            Clean the titles of companies extracted from SEC json file to better match reddit raw data 

        Arguments:
            title: full title of company

        Output:
            a cleaned company title as type string
        """
        # List of popular companies suffixes to remove, unlikely to find these in full company titles written in reddit posts/comments
        terms_to_remove = [
            r'\bcorp\b', r'\bco\b', r'\bltd\b', r'\binc\b', r'\bnv\b', r'\bplc\b',
            r'\bsa\b', r'\bag\b', r'\badr\b', r'\bp\.l\.c\b', r'&', r'/de/', r'/mn/',
            r'/can/', r'/uk/', r'/fi/', r'\bn\.v\.', r'\bs\.a\b', r'\bn\.v\b',
            r'\bgroup\b', r'\bhldg\b', r'\bhldgs\b', r'\bhld\b', r'\bholding\b',
            r'\bholdings\b', r'\bsystems\b', r'\btechnology\b', r'\bmotor\b', r'\bmfg\b', r'\bde\b', r'\bse\b'
        ]
        
        # Make Regex pattern to match terms and punctuation to above list
        terms_pattern = re.compile('|'.join(terms_to_remove), flags=re.IGNORECASE)
        punctuation_pattern = re.compile(r'[^\w\s]')
        
        # Create a new dictionary with cleaned company names
        name_cleaned = terms_pattern.sub('', title)
        # Remove punctuation
        name_cleaned = punctuation_pattern.sub('', name_cleaned)
        # Remove extra whitespace
        name_cleaned = re.sub(r'\s+', ' ', name_cleaned).strip()
        
        return name_cleaned

    @staticmethod
    def get_company_information():
        """
        Purpose:
            Load company name and tickers extract from SEC WEBSITE

        Output:
            json file SEC-registered companies 
        """
        with open('company_tickers.json', 'r') as file:
            data = json.load(file)
        return data

class TickerFrequencyProcessor:
    @staticmethod
    def top_tickers(tickers: List[str], number: int = 5):
        """
        Purpose:
            Find x most frequent tickers from list of tickers

        Arguments:
            tickers: list of tickers
            number: The number of top most frequent tickers found

        Output:
            a list of top x (number) tickers as string type
        """
        freq = Counter(tickers).most_common(number)
        keys = [key[0] for key in freq]
        return keys

    @staticmethod
    def process_ticker_frequencies(dataframe: pd.DataFrame, utc_datetime_column: str = 'utc_datetime', ticker_column: str = 'tickers'):
        """
        Purpose:
            Process dataframe with text data & ticker frequency into dataframe for Web Application 

        Arguments:
            dataframe: dataframe that contains ticker frequencies

        Output:
            Dataframe with top tickers per date
        """
        # Create date column from UTC suitable for graphing
        dataframe['date'] = pd.to_datetime(dataframe[utc_datetime_column], unit='s')
        dataframe['d-m-y'] = dataframe['date'].dt.strftime('%d-%m-%Y')

        # Group by date and ticker for all submissions
        grouped_df = dataframe.groupby('d-m-y')[ticker_column].agg(lambda x: [ticker for list in x for ticker in list]).reset_index()
        explode_df = grouped_df.explode(ticker_column)
        explode_df.dropna(subset=[ticker_column], inplace=True)

        # Add count to tickers
        ticker_count = explode_df.groupby(['d-m-y', ticker_column]).size()

        # Create final dataframe with appropriate column headers
        ticker_df = pd.DataFrame(ticker_count).reset_index()
        ticker_df.columns = ['date', 'ticker', 'count']
        ticker_df['date'] = pd.to_datetime(ticker_df['date'], dayfirst=True)
        ticker_df['ticker'] = ticker_df['ticker'].astype(str)
        return ticker_df
