from datasets import load_dataset
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
import os
import gradio as gr
import pinecone

def get_queries(query ,k=3):
    # create the query vector
    xq = model.encode(query).tolist()
    # now query
    xc = index.query(xq, top_k=k, include_metadata=True)
    queries = {'id':[], 'text':[], 'score':[]}
    for result in xc['matches']:
        queries['id'].append(result['id'])
        queries['text'].append(result['metadata']['text'])
        queries['score'].append(result['score'])
        
    return pd.DataFrame(queries)

# get api key from app.pinecone.io
PINECONE_API_KEY = os.environ.get('a68dbd9b-d99f-49ab-902e-b1b9f6a81f8e') or 'a68dbd9b-d99f-49ab-902e-b1b9f6a81f8e'
# find your environment next to the api key in pinecone console
PINECONE_ENV = os.environ.get('asia-southeast1-gcp') or 'asia-southeast1-gcp'

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

index_name = 'semantic-search-fast'

# now connect to the index
index = pinecone.GRPCIndex(index_name)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Create the Gradio interface with an additional top_k parameter
query_input = gr.inputs.Textbox(label="Query", placeholder="Enter your query....")
top_k_input = gr.inputs.Slider(minimum=1, maximum=10, step=1, default=5, label="Top K")
output_dataframe = gr.outputs.Dataframe(headers=["Result"], type="pandas")

# Define the Gradio interface with both inputs
interface = gr.Interface(
    fn=get_queries,
    inputs=[query_input, top_k_input],
    outputs=output_dataframe,
    title='Semantic Search'
)

# Run the interface
interface.launch()
pinecone.deinit()