from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

ckey="o52C5vZzwJy2gsl0AIafSOq82"
csecret="4OUcE5hLq7PE2DIQRLywMzFo1B3sVmleg6riy1x77NcQ6TG5RC"
atoken="963615242348572673-CvlAMJRR7qmXutvfr8FxKBidmI0A0Qt"
asecret="ymk0PAnPKZPuQ0KGOHhx8wLA7Ik5ObGell3oz1X44mXcK"

class listener(StreamListener):
        def on_data(self,data):
            try:
                all_data=json.loads(data)
                tweet=all_data["text"]
                sentiment_value,confidence=s.sentiment(tweet)
                print(tweet,sentiment_value,confidence)
                if((confidence*100)>80):
                    f=open("output_file.txt","a")
                    f.write(sentiment_value)
                    f.write('\n')
                    f.close()
                return True
            except:
                return True
        def on_error(self,status):
            print(status)

oauth=OAuthHandler(ckey,csecret)
oauth.set_access_token(atoken,asecret)
twitterstream=Stream(oauth,listener())
twitterstream.filter(track=["happy"])
