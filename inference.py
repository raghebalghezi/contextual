
import pickle
from scrape import parse

# url = "https://edition.cnn.com/2020/07/25/health/us-coronavirus-saturday/index.html"

url ="https://edition.cnn.com/2020/07/22/football/liverpool-fc-trophy-presentation-anfield-premier-league-football-spt-intl-gbr/index.html"
# txt = "It is believed that nations that don't treat their childern well, they will become their foolish leaders."

topics = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 
'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 
'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 
'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 
'talk.politics.misc', 'talk.religion.misc']


web_page = parse(url)

classifier = pickle.load(open("classifier.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
sent_vect = vectorizer.transform([web_page])

print("shape of input text {}".format(sent_vect.shape))

class_n = classifier.predict(sent_vect)[0]

print(topics[class_n])

