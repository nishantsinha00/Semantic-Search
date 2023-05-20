# Semantic-Search

The Semantic Search project aims to build a search system that understands the meaning and context of textual data to retrieve relevant information. It leverages techniques from natural language processing (NLP) and machine learning to enable more accurate and context-aware search results.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Technologies](#technologies)
- [Links](#links)

## Overview

Traditional keyword-based search systems rely on exact matches between search queries and document contents. However, they may fail to capture the intended meaning behind user queries or miss semantically related documents. Semantic search addresses this limitation by considering the underlying semantics and relationships between words, phrases, and documents.

The Semantic Search project utilizes advanced NLP techniques, such as word embeddings, sentence embeddings, and semantic similarity measures, to capture the semantic meaning of documents and queries. It utilizes the  <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face Logo" width="20">  tranformers library
 to transform text into high-dimensional vector representations that capture semantic information.

## Features

- Semantic Embeddings: Utilizes word embeddings and sentence embeddings to capture the semantic meaning of text.
- Similarity Measurement: Computes the similarity between queries and documents using cosine similarity or other distance metrics.
- Context-Aware Retrieval: Considers the context and meaning of queries to retrieve relevant documents beyond simple keyword matches.
- Scalability: Handles large-scale document collections efficiently by leveraging indexing and search optimization techniques.
- User-Friendly Interface: Provides an intuitive interface for users to input queries and retrieve semantically related results.

## Installation

```bash
# Steps to reproduce
$ git clone https://github.com/nishantsinha00/Semantic-Search.git
$ conda create -n hf_env python=3.8 
$ proceed ([y]/n)? # When conda asks you to proceed, type 'y'
$ conda activate hf_env
$ pip install -r requirements.txt
$ python semantic_data.py # Creates data on pinecone database
$ python app.py # runs app on local machine
```
After running the above mentioned commands. Follow the link that will appear after running the <b>app.py</b> file.

<b>NOTE: Having a Anaconda distribution as a prerequisite is advisable.</b>

## Technologies

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [Pinecone](https://www.pinecone.io/)

## Links
- [Gradio Semantic Search App](https://huggingface.co/spaces/nishantsinha00/Gradio-Semantic-Search-App)
- [LinkedIn](https://www.linkedin.com/in/nishant-sinha-201885191/)
- [Video walkthrough of the project](https://youtu.be/PUvjFX0KNJM)



