## Artificial Intelligence (AI) - LLAMA

Our LLaMa was created using a pre-processed model called Llama-2-7b-Chat-GGUF. Basically, it converts Llama 2 to a standard called GPT-Generated Unified Format.

We used the 7 billion parameter model, which is the repository of the enhanced 7B model, optimized for dialogue use cases and converted to the Hugging Face Transformers format.

<img style="margin-left: auto; margin-right: auto; width: 100%;" src="/assets/llama.png">

Our model has five main parts:

**EMBEDDINGS**: We used [hkunlp/instructor-large](https://huggingface.co/hkunlp/instructor-large).

[Embeddings](https://www.cloudflare.com/learning/ai/what-are-embeddings/) are representations of values or objects like text, images, and audio that are designed to be consumed by machine learning models and semantic search algorithms. They translate objects like these into a mathematical form according to the factors or traits each one may or may not have, and the categories they belong to.

Essentially, embeddings enable machine learning models to find similar objects. Given a photo or a document, a machine learning model that uses embeddings could find a similar photo or document. Since embeddings make it possible for computers to understand the relationships between words and other objects, they are foundational for artificial intelligence (AI).

**DB** - Database Object responsible for saving training data in memory so that it can be consumed later by the model itself.

**RETRIEVER** - Also known as [Retrieval-Augmented Generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG) is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs' generative process.

**LLM** - Large Language Models are a core component of [LangChain](https://python.langchain.com/docs/get_started/introduction). LangChain does not serve its own LLMs, but rather provides a standard interface for interacting with many different LLMs.

## DataSets

Our AI was trained with data provided by NASA. The training process consisted of a few steps: Data Screening, Data Capture, Data Processing and finally Data Processing by our LLaMa version 2.

<img style="margin-left: auto; margin-right: auto; width: 100%;" src="/assets/storage.png">

The data was taken from the following sources:

#### Internal Data

[LLaMa 2](https://ai.meta.com/llama/): Dataset from Facebook's own model
[Llama-2-7b-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF): Model that uses the GPT-Generated Unified Format

#### External Data to fine tune

[Earth Observatory](https://earthobservatory.nasa.gov/)
[Environmental Performance Index (EPI)](https://epi.yale.edu/)
[Wikipedia](https://en.wikipedia.org/wiki/Water)
[Climatekids - Nasa](https://climatekids.nasa.gov/)
[Climate - Nasa](https://climate.nasa.gov/)
[Center for Science Education](https://ncse.ngo/)
[Earth Data - Nasa](https://www.earthdata.nasa.gov/)
[HydroSheds](https://www.hydrosheds.org/)
[Food and Agriculture Organization of the United Nations - FAO](https://www.fao.org/home/en)

##### Acknowledgements

This project is based on the following repository [localGPT](https://github.com/PromtEngineer/localGPT)
