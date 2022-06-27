import os
import pandas as pd
import tweepy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import numpy as np 
import plotly.express as px


# confidential
consumer_secret = os.environ['CONSUMER_SECRET']
consumer_key = os.environ['CONSUMER_KEY']
access_token = os.environ['ACCESS_TOKEN']
access_token_secret = os.environ['ACCESS_TOKEN_SECRET']
bearer_token = os.environ['BEARER_TOKEN']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# woied INDIA
woeid = 2282863

analyzer = SentimentIntensityAnalyzer()




# utility functions
def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) #Line removess @ mentions (r tells that the expression is a raw string)
    text = re.sub(r'#', '', text) #Remove #
    text = re.sub(r'RT[\s]+', '', text) #Remove Retweet
    text = re.sub(r'https?:\/\/\S+', '', text) #Remove links
    return text

def updateTweets():
    result = []
    trends = api.get_place_trends(woeid)
    for trend in trends[0]['trends'][:5]:
        public_tweets = api.search_tweets(trend['name'],result_type="recent",tweet_mode = "extended")
        for tweet in public_tweets:
            
            if tweet.full_text.startswith("RT @"):
                # print(tweet.retweeted_status.full_text)
                tweet = tweet.retweeted_status
                # print(f"https://twitter.com/twitter/statuses/{tweet.id}")
                # continue
            # tweet url https://twitter.com/twitter/statuses/{id}
            text = GoogleTranslator(source='auto', target='en').translate(tweet.full_text)
            # analysis = TextBlob(text)
            sentiment_dict = analyzer.polarity_scores(text)
            data = {
            "username":tweet.user.screen_name,
                # "location":
            "polarity":sentiment_dict["compound"]*100,
            "topic":trend['name'],
            "text": tweet.full_text,  
            "date":tweet.created_at ,
            "url":f"https://twitter.com/twitter/statuses/{tweet.id}",
            "action":False,
            "id":tweet.id,
            "retweet":tweet.retweet_count,
            "score":sentiment_dict["compound"]*100,
            "type":"negative"
            }
            result.append(data)
    return result
            

def getTweet(id):
    tweet = api.lookup_statuses(id=[id,])[0]
    text = GoogleTranslator(source='auto', target='en').translate(tweet.text)
    sentiment_dict = analyzer.polarity_scores(text)
    # print(sentiment_dict)
    sentiment_dict["url"] = f"https://twitter.com/twitter/statuses/{tweet.id}"
    sentiment_dict["pos"] = round(sentiment_dict["pos"]*100,2)
    sentiment_dict["neg"] = round(sentiment_dict["neg"]*100,2)
    sentiment_dict["neu"] = round(sentiment_dict["neu"]*100,2)
    return sentiment_dict


def getUser(id):
    user = api.get_user(screen_name=id)
    posts = api.user_timeline(screen_name = id, count = 200, language = "en", tweet_mode = "extended")
    df_children = []
    neg_hash = []
    all_hash = []
    for tweet in posts:
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(tweet.full_text)
        except:
            translated = tweet.full_text
        score = analyzer.polarity_scores(translated)['compound']
        overall = "Postive" if score>0 else "Negative" if score <0 else  "Neutral"
        data = [translated,overall,score]
        df_children.append(data)
        l = [entity["text"] for entity in tweet.entities["hashtags"]]
        all_hash += l
        if score<0:
            neg_hash += l
    df_tweets = pd.DataFrame(df_children,columns = ["Tweets","Polarity","Score"])
    
    # all tweets chart
    freq = nltk.FreqDist(all_hash)
    d = pd.DataFrame({"Hashtag": list(freq.keys()), "Count": list(freq.values())})
    d = d.nlargest(columns = "Count", n = 10)
    fig = px.bar(d,x = "Hashtag", y = "Count",color="Hashtag" ,title="Overall Topics")
    all_graph = fig.to_html(full_html=False)


    # negative tweets chart
    freq = nltk.FreqDist(neg_hash)
    d = pd.DataFrame({"Hashtag": list(freq.keys()), "Count": list(freq.values())})
    d = d.nlargest(columns = "Count", n = 5)
    fig = px.bar(d,x = "Hashtag", y = "Count",color="Hashtag" ,title="Negative Topics")
    neg_graph = fig.to_html(full_html=False)

    # pie chart
    fig = px.pie(df_tweets, names='Polarity',color_discrete_map={'Positive':'#00cc96',
                                 'Negative':'#ef553b',
                                 'Neutral':'#fff'}, hole = 0.3,title='Summary') 
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    fig.update_traces(hoverinfo='label+percent', marker=dict(line=dict(color='#000000', width=2)))  
    pie_chart = fig.to_html(full_html=False)
    result = {
        "img_url": user.profile_image_url_https,
        "name": user.name,
        "description":user.description,
        "location": user.location,
        "created_at": user.created_at,
        "followers": user.followers_count,
        "followings": user.friends_count,
        "is_bot": False,
        "overall": df_tweets['Polarity'].max(),
        "neg_graph":neg_graph,
        "all_graph":all_graph,
        "pie_chart":pie_chart
    }
    return result