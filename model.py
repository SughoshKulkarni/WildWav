# Importing the libraries
import pandas as pd
import pickle

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:, :4]

y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[5.1, 3.5, 1.4,0.2]]))