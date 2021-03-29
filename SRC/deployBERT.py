import streamlit as st

import numpy as np
import pandas as pd

import nlu
from sklearn.metrics.pairwise import cosine_similarity
import os

#attach java paths
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]

# load pipeline model and pickled dataframes with embeddings
pipe = nlu.load(path='pipeline2')
embeddings = pd.read_pickle('pipeline2/embeddings_sample1.pkl')
embeddings['newembeddings'] = [embeddings.embeddings.iloc[i][i] for i in embeddings.index]
embeddings.drop('embeddings', axis=1, inplace=True)

st.header("Book Recomendation")
st.subheader("Sample Size :")

st.text(str(embeddings.shape))

st.dataframe(embeddings.head(5))

def calculate_similarity(df, description_embeddings, top=5):
#     embedmat = np.reshape(df.embeddings.iloc[0][0], (-1,1))
    df['similarity'] = df.newembeddings.apply(lambda x : cosine_similarity([x], [description_embeddings]))
    df['similarity'] = df['similarity'].apply(lambda x: x[0][0])
    df.sort_values('similarity', ascending = False, inplace=True)
    topvals = df.head(top)
    del df
    return topvals

@st.cache()
def BookTitles():
    titles = [str(embeddings.title[i]) for i in embeddings.index]
    return titles

def getRecords(titles):
    records = embeddings[embeddings.title==str(titles)]
    return records

alltitles = ['<select>']
alltitles += BookTitles()
title = st.selectbox("Select TITLE", tuple(alltitles))

if title!= '<select>':
    st.write('You selected:', title)
    st.write("Getting its following records : ")
    records = getRecords(title)
    st.dataframe(records)

    st.image(records.image_url.iloc[0])
    st.write("Rating : " + records.average_rating.iloc[0])
    st.write("Description : " + records.description.iloc[0])

    st.write("Getting Similar Books, please wait a bit....")

    topvals = calculate_similarity(embeddings, records.newembeddings.iloc[0])
    for i in topvals.index:
        st.image(topvals.image_url.loc[i])
        st.write("Book Name : ", topvals.title.loc[i])
        st.write("Rating : " + topvals.average_rating.loc[i])
        st.write("Book Description : " + topvals.description.loc[i])
        st.write('')

    st.dataframe(topvals)

st.write("")
st.write("OR")
st.write("")

descripinput = st.text_input('Enter Description: ')
if descripinput:
    st.write("Getting Similar Books, please wait a bit....")
    embedmat = pipe.predict(descripinput)['embed_sentence_bert_embeddings'][0]
    topvals = calculate_similarity(embeddings, embedmat)

    st.dataframe(topvals)

    for i in topvals.index:
        st.image(topvals.image_url.loc[i])
        st.write("Book Name : ", topvals.title.loc[i])
        st.write("Rating : " + topvals.average_rating.loc[i])
        st.write("Book Description : " + topvals.description.loc[i])
        st.write('')


