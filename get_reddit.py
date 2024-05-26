import praw
import pandas as pd
from typing import List
from datetime import datetime
import pytz

class RedditSubmissions:
    def __init__(self, reddit_instance):
        self.reddit = reddit_instance

    def extract_submission_data(self, submissions: List[praw.models.Submission]):
        """
        Purpose:
            Extract key data from submission type objects called from Reddit API

        Arguments:
            A list of submissions object type praw.models.Submission

        Output:
            Dataframe: dataframe containing series (columns) of data
            utc_datetime series: Timestamp of the submisssion represented in UNIX time (wanted to keep core data utc, can transform to local timestamp later if required)
            author series: author of post / submission
            title series: title of post / submission
            upvote_ratio series: Percentage of upvotes from votes on submission
            num_comments series: Number of comments created by users on the submission/post
            all_text series: Concatenate the title and text data to parse through for tickers/companies
        """
        data = {'utc_datetime': [], 'author': [], 'title': [], 'text': [], 'upvote_ratio': [], 'num_comments': []} 
        for submission in submissions:
            data['utc_datetime'].append(submission.created_utc)
            data['author'].append(str(submission.author))
            data['title'].append(str(submission.title))
            data['text'].append(str(submission.selftext))
            data['upvote_ratio'].append(float(submission.upvote_ratio))
            data['num_comments'].append(int(submission.num_comments))
        dataframe = pd.DataFrame(data)

        # Fill NaN or None with empty strings before concatenation
        dataframe['title'] = dataframe['title'].astype(str).fillna('')
        dataframe['text'] = dataframe['text'].astype(str).fillna('')

        dataframe['all_text'] = dataframe['title'] + ' ' + dataframe['text']
        return dataframe

    def get_submissions(self, subreddit: str, sort: str = 'top', time_filter: str = 'month', limit: int = 10):
        """
        Purpose:
            Get top submissions from a subreddit, filtered by time range. 

        Arguments:
            subreddit: subreddit of inquiry
            sort: the type of submissions wanted, i.e. top, new, rising, hot 
            time_filter: the time range for which top submissions is queried for 
            limit: the number of submissions desired

        Output:
            submissions: a list of submissions / posts
        """
        try:
            subreddit_instance = self.reddit.subreddit(subreddit)
            
            if sort == 'top':
                submissions = list(subreddit_instance.top(time_filter=time_filter, limit=limit))
            elif sort == 'new':
                submissions = list(subreddit_instance.new(limit=limit))
            elif sort == 'rising':
                submissions = list(subreddit_instance.rising(limit=limit))
            elif sort == 'hot':
                submissions = list(subreddit_instance.hot(limit=limit))
            else:
                print("Sort option not recognised. Use 'top', 'new', 'rising', 'hot'")
            
            return submissions
        except Exception as e:
            print(f"An error occurred: {e}")
            return []


class RedditAPIHelper:
    def __init__(self, reddit_instance):
        self.reddit = reddit_instance

    def get_api_limit(self, timezone: str):
        """
        Purpose:
            Identify how many API calls are left within the limit imposed by Reddit (600 per 10 minutes).

        Arguments:
            timezone: Timezone as a string e.g. ('Europe/Zurich').

        Output:
            A print statement informing when the reset time is for API calls and how many remaining requests there are.
        """
        try:
            # Connect to Reddit authority limits
            limit = self.reddit.auth.limits
            
            if limit is None:
                print("Failed to retrieve API limits.")
                return

            remaining_requests = limit.get('remaining')
            reset_timestamp = limit.get('reset_timestamp')
            
            if remaining_requests is None or reset_timestamp is None:
                print("Failed to retrieve remaining requests or reset timestamp.")
                return

            # Get datetime of when reset will occur & translate into local time
            reset = datetime.utcfromtimestamp(reset_timestamp)
            reset_time = pytz.utc.localize(reset).astimezone(pytz.timezone(timezone))
            print(f'Reset time is {reset_time}, you have {remaining_requests} requests remaining')
            
        except pytz.UnknownTimeZoneError:
            print(f'Unknown timezone: {timezone}')
        except Exception as e:
            print(f'An error occurred: {e}')