# check single sentence similarity

import pandas as pd

import nlu
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle(r'pipeline/embeddings_sample1.pkl')

df['newembeddings'] = [df.embeddings.iloc[i][i] for i in df.index]
df.drop('embeddings', axis=1, inplace=True)


def calculate_similarity(df, description_embeddings, top=5):
#     embedmat = np.reshape(df.embeddings.iloc[0][0], (-1,1))
    df['similarity'] = df.newembeddings.apply(lambda x : cosine_similarity([x], [description_embeddings]))
    df['similarity'] = df['similarity'].apply(lambda x: x[0][0])
    df.sort_values('similarity', ascending = False, inplace=True)
    topvals = df.head(top)
    del df
    return topvals

pipe = nlu.load(request='from_disk', path='pipeline')

descpred = pipe.predict("Unabridged CDs, 25 CDs, 30 hoursRead by TBABobbi Anderson and the other good folks of Havenell.")['embed_sentence_bert_embeddings'][0]

resdf = calculate_similarity(df, descpred)

print(resdf)