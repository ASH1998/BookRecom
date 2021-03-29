import gzip
import pandas as pd

import gc
import gzip
import json
from tqdm import tqdm

hugeChunk = list()
ctr = 0
with gzip.open('Data\goodreads_books.json.gz') as fopen:
    for l in tqdm(fopen):
        data = json.loads(l)
#         print(data)
#         break
        # hugeChunk.append([data['book_id'], data['rating'], data['review_text'], data['n_votes']])
        hugeChunk.append(data)
        ctr+=1
        if ctr%100000==0:
            reviewSample = pd.DataFrame(hugeChunk)
            reviewSample.to_hdf('Data\samplereview\sample'+str(ctr)+'.h5', 'data')
            
            hugeChunk = list()
            gc.collect()