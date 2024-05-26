import os
import re 
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from fuzzywuzzy import fuzz, process
from utils import ModelUtils


class RegexExtraction:
    """
    A class to encapsulate the extraction of company tickers from a list of tokens.
    """
    @staticmethod
    def extract_tickers(tokens: List[str], ticker_dict: List[str], blacklist=None):
        """
        Purpose:
            Extract company tickers from a list of tokens using the provided regex pattern.

        Arguments:
            tokens: List of tokens (words) to scan for tickers
            blacklist: list of words (String) to actively avoid

        Output:
            A string of positively identified tickers
            avg_time: the average time taken to identify a ticker
        """
        ticker_pattern = re.compile(r"\b[A-Z]{1,5}\b")

        if blacklist is None:
            blacklist = []

        # Use the regex pattern to filter out tickers from the tokens
        start_time = time.time()
        tickers = [token.upper() for token in tokens if token.upper() not in blacklist and ticker_pattern.match(token)]

        # Match extracted tickers with those from the dictionary
        positive_matches = [ticker for ticker in tickers if ticker in ticker_dict]
        
        #end time and record avg_time
        end_time = time.time()
        avg_time = (end_time - start_time) / len(tokens)

        return positive_matches, avg_time
    
    @staticmethod
    def create_ticker_list(data: Dict):
        ticker_list = [v['ticker'] for k, v in data.items()]
        return ticker_list

class GloveModel:
    """
    A class for extracting companies titles and tickers using GloVe vector model.
    """
    def __init__(self):
        self.glove_model = {}

    def load_glove_model(self, file_path):
        """
        Purpose:
            Load pre-trained GloVe word vectors created by Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.

        Arguments:
            file_path: file path in meme_Stock folder with GloVe vector representations

        Output:
            txt file of GloVe word vector representations 
        """
        self.glove_model = {}
        if not os.path.exists(file_path):
            print(f'Cannot find filepath: {file_path}')
            return self.glove_model
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                vector = np.array([float(val) for val in split_line[1:]])
                self.glove_model[word] = vector
        
        return self.glove_model
    
    def compute_vector(self, text, model, vector_size=50):
        """
        Purpose:
            Compute vector for company titles 

        Arguments:
            data: json of company details (title & tickers) from the SEC website
            model: GloVe model

        Output:
            an array or vector representing the ticker or the title of the company
        """
        words = text.split()
        word_vectors = [model[word] for word in words if word in model]
        if not word_vectors:
            return np.zeros(vector_size)
        return np.mean(word_vectors, axis=0)

    def create_vector_dicts(self, data, model, vector_size=50):
        """
        Purpose:
            create two look-up dictonaries with the tickers & titles and corresponding vector representations

        Arguments:
            data: json of company details (title & tickers) from the SEC website
            model: the GloVe model dictionary of words & associated vectors
            vector_size: size of vector desired, default is 50 according to txt file loaded and to save memory and computing resources

        Output:
            ticker_to_vector: dictionary with tickers as keys and vectors as values
            title_to_vector: dictionary with company titles as keys and vectors as values
            ticker_to_title: dictionary to look up company name with ticker
        """
        ticker_to_vector = {}
        title_to_vector = {}
        ticker_to_title = {}

        for v in data.values():
            title = ModelUtils.clean_titles(v['title'].lower())
            ticker = v['ticker'].lower()
            
            ticker_to_title[ticker] = title

            title_vector = self.compute_vector(title, model, vector_size)
            ticker_vector = self.compute_vector(ticker, model, vector_size)
            
            title_to_vector[title] = title_vector
            ticker_to_vector[ticker] = ticker_vector

        return ticker_to_vector, title_to_vector, ticker_to_title

    def merge_vector_dicts(self, ticker_to_vector, title_to_vector, ticker_to_title):
        """
        Purpose:
            Combine two vector representations of one company (as ticker & title) expected to be present in reddit text into a dictionary

        Arguments:
            ticker_to_vector: dictionary with tickers as keys and vector as values
            title_to_vector: dictionary with titles as key and vector as values
            ticker_to_title: dictionary with ticker as keys and title of company as values 
        Output:
            combined dict: dictionary with tickers as keys and a tuple of title as a vector and ticker as a vector 
        """
        combined_dict = {}
        for ticker, ticker_vec in ticker_to_vector.items():
            title = ticker_to_title[ticker]  
            title_vec = title_to_vector[title]  
            combined_dict[ticker] = (ticker_vec, title_vec)
        return combined_dict

    def get_glove_vector(self, word, vector_dim=50):
        """
        Purpose:
            Retrieve token vector from GloVe Model Dictionary

        Arguments:
            word: tokenised word from text data
            glove_model: GloVe dictionary storing words and associated vectors 
        
        Output:
            Array (vector or vector of 0's if not found in GloVe model)
        """
        return self.glove_model.get(word, np.zeros(vector_dim))

    def glove_best_match(self, token_vec, combined_vec_dict, threshold=0.95):
        """
        Purpose: to find the best vectorised match from a tokenised word

        Arguments:
            token_vec: vector representing the tokenised word
            combined_vec_dict: dictionary with key (ticker) and values (vectorised ticker, vectorised company title)
            thresholds: thresholds for cosine similarity required for best match

        Output:
            best_match: the key of the token with the best cosine similarity that is above the threshold
        """
        best_match = None
        highest_similarity = -1
        
        for key, (ticker_vec, title_vec) in combined_vec_dict.items():
            similarity_ticker = ModelUtils.cosine_similarity(ticker_vec, token_vec)
            similarity_title = ModelUtils.cosine_similarity(title_vec, token_vec)
            max_similarity = max(similarity_ticker, similarity_title)
            
            if max_similarity > highest_similarity:
                highest_similarity = max_similarity
                best_match = key

        if highest_similarity >= threshold:
            return best_match
        return ""

    def glove_optimum_threshold(self, tokens: List[str], true_tickers: List[str], combined_vec_dict, vector_dim=50, thresholds=None):
        """
        Purpose: to test the optimum threshold level for the GloVe model detection of ticker/companies 

        Arguments:
            tokens: list of tokenised words
            true_tickers: correct tickers found in data
            combined_vec_dict: dictionary with key (ticker) and values (vectorised ticker, vectorised company title)
            vector_dim: vector dimensions
            thresholds: thresholds for cosine similarity required for best match

        Output:
            Tuple: threshold, precision of model, sensitivity of model
        """
        if thresholds is None:
            thresholds = [i / 100 for i in range(50, 100, 5)]
        elif any(threshold > 1 or threshold < -1 for threshold in thresholds):
            print('Threshold values must be between -1 and 1.')
            return []

        token_vectors = [self.get_glove_vector(token, vector_dim) for token in tokens]
        results = []

        for threshold in thresholds:
            extracted_tickers = []
            times = []
            for token_vec in token_vectors:
                start_time = time.time()
                match = self.glove_best_match(token_vec, combined_vec_dict, threshold)
                end_time = time.time()
                if match:
                    extracted_tickers.append(match)
                times.append(end_time - start_time)

            avg_time = np.mean(times)
            precision, sensitivity = ModelUtils.evaluate_model_performance(true_tickers, extracted_tickers)
            results.append(f'Threshold: {threshold}, precision: {precision:.3f}, sensitivity: {sensitivity:.3f}, time taken to match: {avg_time:.4f}s')
        
        return results

    def run_glove_model(self, tokens: List[str], combined_vec_dict, threshold, vector_dim=50):
        """
        Purpose: to run the GloVe model on a list of tokens 

        Arguments:
            tokens: list of tokenised words
            combined_vec_dict: dictionary with key (ticker) and values (vectorised ticker, vectorised company title)
            vector_dim: vector dimensions
            thresholds: desired threshold for cosine similarity in glove_best_match

        Output:
            list of predicted tickers
        """
        if threshold is None:
            print('input threshold')
        elif threshold > 1 or threshold < -1:
            print('Threshold values must be between -1 and 1.')
            return []
        
        predicted_tickers = []

        token_vectors = [self.get_glove_vector(token, vector_dim) for token in tokens]

        for token_vec in token_vectors:
            match = self.glove_best_match(token_vec, combined_vec_dict, threshold)
            if match:
                predicted_tickers.append(match)

        return predicted_tickers

class FuzzModel:
    def create_fuzz_dicts(self, data: dict):
        """
        Purpose:
            Create look-up dictionaries to help match predicted tickers and companies in text to tickers and companies found in SEC json file

        Arguments:
            data: json of company details (title & tickers) from the SEC website

        Output:
            title_to_ticker: dictionary whereby key it company title and ticker is value
            ticker_to_title: dictionary whereby key is ticker and value is company title
            combined_dict: concatenated dictionary of title_to_ticker & ticker_to_title
        """

        title_to_ticker = {ModelUtils.clean_titles(v['title'].lower()): v['ticker'].lower() for v in data.values()}
        ticker_to_title = {v['ticker'].lower(): ModelUtils.clean_titles(v['title'].lower()) for v in data.values()}
        
        combined_dict = {}
        for ticker, title in ticker_to_title.items():
            combined_dict[ticker] = (ticker, title)

        return title_to_ticker, ticker_to_title, combined_dict

    def fuzz_preprocess(self, data: dict): 
        """
        Purpose:
            To preprocess lookup dictionary for fuzzy matching

        Arguments:
            data: a dictionary with a key as the ticker and values and ticker or title of company
        
        Output:
            value_to_key: dictionary with common key for different values of the same company i.e. a positive appl confirmation could be 'Apple' or 'AAPL'
            values: a list of all the possible values to match the tokenised words against. This helps train the fuzzywuzzy matching function
        """
        values_to_key = {}

        for key, value in data.items():
            values_to_key[value[0]] = key
            values_to_key[value[1]] = key
        values = list(values_to_key.keys())

        return values_to_key, values

    def fuzz_best_match(self, token: str, value_to_key: dict, values: List[str], threshold=85): 
        """
        Purpose:
            to match tokenised words from reddit data to pre-processed company tickers and titles

        Arguments:
            word: tokenized word from reddit text
            value_to_key: dictionary with ticker as a key, and all possible values (title of company, ticker) as the values
            values: all the values fuzzywuzzy needs to match for.
            threshold: Extent to which the word must match the values fuzzy is looking for on a scale of 0-100
        
        Output:
            returns the best match fuzzy can find or None if no words meet the threshold
        """
        best_match = process.extractOne(token, values, scorer=fuzz.token_sort_ratio)
        
        if best_match and best_match[1] >= threshold:
            return value_to_key[best_match[0]]
        else:
            return None 

    def fuzz_optimum_threshold(self, tokens: List[str], value_to_key: Dict, values: List[str], true_tickers: List[str], thresholds=None): 
        """
        Purpose:
            Finds the optimum threshold for fuzzy matching based on precision and recall.

        Arguments:
            token: list of tokens from preprocessed data
            value_to_key: dictionary with all ticker key and ticker & company name values
            values: list of all values fuzzy must match for
            true_tickers: true tickers in test data
            thresholds: testing thresholds

        Output:
            list of tuples with threshold and corresponding precision and sensitivity
        """
        if thresholds is None:
            thresholds = np.arange(80, 100, 5)

        result = []

        for threshold in thresholds:
            predicted_tickers = []
            times = []

            for token in tokens:
                start_time = time.time()
                best_match = self.fuzz_best_match(token, value_to_key, values, threshold)
                end_time = time.time()
                times.append(end_time - start_time)

                if best_match:
                    predicted_tickers.append(best_match)

            precision, sensitivity = ModelUtils.evaluate_model_performance(true_tickers, predicted_tickers)
            avg_time = np.mean(times)
            result.append(f'Threshold: {threshold}, precision: {precision:.3f}, sensitivity: {sensitivity:.3f}, time taken to match: {avg_time:.4f}s')
        return result


    def run_fuzz_model(self, tokens: List[str], value_to_key: dict, values: List[str], threshold: int):
        """
        Purpose:
            Run fuzz model on list of tokens

        Arguments:
            token: list of tokens from preprocessed data
            value_to_key: dictionary with all ticker key and ticker & company name values
            values: list of all values fuzzy must match for
            thresholds: desired threshold for a match

        Output:
            list of tickers as type string
        """

        predicted_tickers = []

        for token in tokens:
            best_match = self.fuzz_best_match(token, value_to_key, values, threshold)
            if best_match:
                    predicted_tickers.append(best_match)

        return predicted_tickers