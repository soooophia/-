
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    '病史,高血糖,糖尿病.',
   '高血压内科,高血脂',
    '高血脂,病史正常.',
    '心率,异常,高血脂',
 ]
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)


vectorizer = TfidfVectorizer(stop_words=['病史','内科','正常',','])
X=vectorizer.fit_transform(corpus)
print ('vocabulary list:\n')
for key,value in vectorizer.vocabulary_.items():
    print (key,value)