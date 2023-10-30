import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle5 as pickle


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(columns=['id','Unnamed: 32'])
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    
    return data

def create_model(data):
    X= data.drop(columns=['diagnosis'])
    y = data['diagnosis']
    
    # scale the data
    scalar = StandardScaler()
    X = scalar.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print('Classification report: \n', classification_report(y_test, y_pred))
    
    return model, scalar


def main():
    data = get_clean_data()
    
    model, scalar = create_model(data)

    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)

    

if __name__ == '__main__':
    main()