from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
import os
import pinecone

def create_data(batch_size = 128):
    for i in tqdm(range(0, len(questions), batch_size)):
        # find end of batch
        i_end = min(i+batch_size, len(questions))
        # create IDs batch
        ids = [str(x) for x in range(i, i_end)]
        # create metadata batch
        metadatas = [{'text': text} for text in questions[i:i_end]]
        # create embeddings
        xc = model.encode(questions[i:i_end])
        # create records list for upsert
        records = zip(ids, xc, metadatas)
        # upsert to Pinecone
        index.upsert(vectors=records)
        
dataset = load_dataset('quora', split='train')

questions = []

for record in dataset['questions']:
    questions.extend(record['text'])
  
# remove duplicates
questions = list(set(questions))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('a68dbd9b-d99f-49ab-902e-b1b9f6a81f8e') or 'a68dbd9b-d99f-49ab-902e-b1b9f6a81f8e'
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('asia-southeast1-gcp') or 'asia-southeast1-gcp'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'semantic-search-fast'

# only create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=model.get_sentence_embedding_dimension(),
        metric='cosine'
    )

# now connect to the index
index = pinecone.GRPCIndex(index_name)

create_data()
print(index.describe_index_stats())
pinecone.deinit()

