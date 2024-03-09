#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Define the Streamlit app
def app():

    classifier = ''
    if st.session_state['selected_model'] == 0:     # decision tree
        report = """Achieves good accuracy, but can be prone to 
        overfitting, leading to lower performance on unseen data.
        Simple and interpretable, allowing visualization of decision rules.
        Susceptible to changes in the training data, potentially 
        leading to high variance in predictions."""
        classifier = 'Decision Tree'
    elif st.session_state['selected_model'] == 1:   # Extreme Random Forest
        report = """Performance: Can achieve similar or slightly better 
        accuracy compared to a random forest, but results can vary 
        depending on hyperparameter tuning. Introduces additional randomness 
        during tree building by randomly selecting features at each split.  Aims to 
        further improve generalization and reduce overfitting by increasing 
        the diversity of trees in the ensemble. Requires careful 
        hyperparameter tuning to achieve optimal performance."""
        classifier = "Extreme Random Forest"
    else:
        report = """Gradient boosting classifiers perform well on the MNIST 
        handwritten digit dataset, typically achieving accuracy in the high 90s 
        (often exceeding 98%). Gradient boosting excels at handling complex, 
        non-linear relationships within the data, which is helpful for
        recognizing the varied shapes of handwritten digits. It can achieve 
        accuracy comparable to other popular methods for MNIST like 
        Support Vector Machines (SVMs) and Random Forests."""
        classifier = "Gradient Boosting"

    st.subheader('Performance of the ' + classifier)

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']
    y_train= st.session_state['y_train']
    y_test = st.session_state['y_test']

    st.sidebar.subheader('Select the classifier')

    # Create the selection of classifier
    clf = DecisionTreeClassifier()
    st.session_state['selected_model'] = 0
    options = ['Decision Tree', 'Extreme Random Forest Classifier', 'Gradient Boosting Classifier']
    selected_option = st.sidebar.selectbox('Select the classifier', options)
    if selected_option=='Extreme Random Forest Classifier':        
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=0)
        st.session_state['selected_model'] = 1
    elif selected_option == 'Gradient Boosting Classifier':
        clf = GradientBoostingClassifier()
        st.session_state['selected_model'] = 2

    # save the clf to the session variable
    st.session_state['clf'] = clf

    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    st.subheader('Confusion Matrix')
    st.write('Confusion Matrix')
    cm = confusion_matrix(y_test, y_test_pred)
    st.text(cm)

    st.subheader('Performance Metrics')
    st.text(classification_report(y_test, y_test_pred))
    
    st.write(classifier)
    st.write(report)

    # save the clf to the session state
    st.session_state['clf'] = clf

#run the app
if __name__ == "__main__":
    app()
