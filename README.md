# RAG-1: Build a Retrieval Augmented Generation (RAG) App, Part 1

A Colab notebook that builds a question-answering app over a single blog post,
[LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)
by Lilian Weng. It follows the LangChain "Build a RAG App, Part 1" tutorial, adapted to
run on Google Gemini (free tier).

**Notebook:** [`RagPart1.ipynb`](RagPart1.ipynb)

## The idea

An LLM has never read your documents, so it may guess. RAG works like an open-book exam:
find the right pages first, then hand them to the model together with the question.

## Pipeline

**Indexing** (done once)

| Step | What happens | Code |
|---|---|---|
| Load | Download the blog post and keep only the title, header and content | `WebBaseLoader` |
| Split | Cut it into ~1000-character chunks with 200 characters of overlap | `RecursiveCharacterTextSplitter` |
| Embed + store | Turn each chunk into a vector and save it | Gemini embeddings, `InMemoryVectorStore` |

**Retrieval and generation** (done per question)

| Step | What happens | Code |
|---|---|---|
| Retrieve | Find the 4 chunks closest in meaning to the question | `similarity_search` |
| Generate | Gemini answers using only those chunks | `ChatGoogleGenerativeAI` |

The two steps are wired together as a graph with [LangGraph](https://langchain-ai.github.io/langgraph/).

## Query analysis

The last section adds a step before retrieval. The LLM rewrites the user's question
into a structured search (`query` + `section`), and retrieval is filtered to that
section of the post. Asking "What does the end of the post say about Task Decomposition?"
produces `{'query': 'Task Decomposition', 'section': 'end'}`, and only chunks from the
end of the post come back.

## Key outputs

| Stage | Output |
|---|---|
| Load | The post is one document of about 43k characters |
| Split | 66 chunks |
| Store | One UUID per chunk |
| Invoke | Retrieved context next to the final answer |
| Query analysis | The structured query, then chunks tagged `section: end` |

### Load

The post loads as a single document. The deprecation warning is harmless: LangChain is
moving document loaders out of `langchain-community`, and `WebBaseLoader` still works.

![Load output](images/load-output.png)

## Run it

1. Open `RagPart1.ipynb` in Google Colab.
2. Get a free Gemini key at https://aistudio.google.com/apikey.
3. Add it as a Colab secret named `GOOGLE_API_KEY` and turn on "Notebook access".
4. Run the cells in order. The first cell installs the dependencies. If pip asks for a restart, restart and run it again.

## Stack

LangChain, LangGraph, Google Gemini (chat and embeddings), BeautifulSoup.
