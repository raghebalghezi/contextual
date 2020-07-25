import urllib.request
from bs4 import BeautifulSoup
import ssl



url = "https://edition.cnn.com/2020/07/24/economy/surviving-coronavirus-job-loss-tips/index.html"



def parse(url):
    # This restores the same behavior as before.
    context = ssl._create_unverified_context()

    html = urllib.request.urlopen(url, context=context).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

# print(parse(url))