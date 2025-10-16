# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

# Date: 16-10-2025
# Register no. 212222030007
# Aim: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights with Multiple AI Tools

#AI Tools Required:
<hr>

<h2> Required AI Tools</h2>
<p>
To run this project effectively, you’ll need access to some or all of the following AI tools:
</p>

<h3>1. OpenAI API (GPT, DALL·E, Whisper)</h3>
<ul>
  <li><b>Use case:</b> Text generation, summarization, conversation, embeddings, image generation, speech-to-text.</li>
  <li><b>Integration:</b> <code>openai</code> Python package.</li>
</ul>

<h3>2. Anthropic Claude API</h3>
<ul>
  <li><b>Use case:</b> Structured reasoning, summarization, safer text generation.</li>
  <li><b>Integration:</b> <code>anthropic</code> Python package.</li>
</ul>

<h3>3. Cohere API</h3>
<ul>
  <li><b>Use case:</b> Text classification, embeddings, clustering, semantic search.</li>
  <li><b>Integration:</b> <code>cohere</code> Python package.</li>
</ul>

<h3>4. Hugging Face Transformers</h3>
<ul>
  <li><b>Use case:</b> Open-source models for NLP, vision, and speech tasks.</li>
  <li><b>Integration:</b> <code>transformers</code> + <code>datasets</code>.</li>
</ul>

<h3>5. Google Generative AI (Gemini / PaLM)</h3>
<ul>
  <li><b>Use case:</b> Text processing, code generation, multimodal AI.</li>
  <li><b>Integration:</b> <code>google-generativeai</code> Python package.</li>
</ul>

<h3>6. LangChain</h3>
<ul>
  <li><b>Use case:</b> Orchestration framework for combining multiple AI tools.</li>
  <li><b>Integration:</b> <code>langchain</code> Python package.</li>
</ul>

<h3>7. Vector Databases (Optional)</h3>
<ul>
  <li><b>Examples:</b> Pinecone, Weaviate, ChromaDB.</li>
  <li><b>Use case:</b> Storing embeddings for semantic search and retrieval-augmented generation (RAG).</li>
</ul>

<hr>

<h2> Getting Started</h2>

<p>Follow these steps to set up and run the project:</p>

<h3>1. Clone the Repository</h3>
<pre><code>
git clone https://github.com/your-username/multi-ai-tools-automation.git
cd multi-ai-tools-automation
</code></pre>

<h3>2. Install Dependencies</h3>
<p>Install the required Python packages:</p>
<pre><code>
pip install openai cohere anthropic langchain transformers datasets google-generativeai
</code></pre>

<h3>3. Set API Keys</h3>
<p>Create a <code>.env</code> file in the project root and add your API keys:</p>
<pre><code>
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
</code></pre>

<h3>4. Run the Example</h3>
<p>Run the sample workflow:</p>
<pre><code>
python main.py
</code></pre>

<h3>5. Extend the Project</h3>
<ul>
  <li>Add more AI tools by creating new query functions.</li>
  <li>Integrate vector databases (Pinecone, Weaviate, ChromaDB) for retrieval tasks.</li>
  <li>Enhance the logic layer to analyze and visualize insights.</li>
</ul>


# Explanation:
Experiment the persona pattern as a programmer for any specific applications related with your interesting area. 
Generate the outoput using more than one AI tool and based on the code generation analyse and discussing that. 

# Conclusion:

<h1>Multi-AI Tools Integration & Comparison</h1>

<p>
This project demonstrates how to integrate multiple AI APIs, compare outputs, and generate actionable insights automatically.
The provided Python script shows a practical way to call different providers, normalize their responses, evaluate similarities, and produce a final report.
</p>

<h2>Features</h2>
<ul>
  <li>Send the same prompt to multiple AI providers.</li>
  <li>Normalize outputs into a common structure.</li>
  <li>Compare results using similarity, latency, and token usage.</li>
  <li>Generate insights and recommendations (fastest provider, most verbose, agreement/disagreement).</li>
  <li>Save results in JSON for further use.</li>
</ul>

<h2>Architecture</h2>
<pre><code>
+-----------------+   prompt   +----------------+   +-----------------+
|   Client CLI    | --------> |  Controller    |-->| Provider A API  |
+-----------------+           |  (compare)     |   +-----------------+
                              |                |   +-----------------+
                              |                |-->| Provider B API  |
                              |                |   +-----------------+
                              +----------------+
                                      |
                                      v
                              +-----------------+
                              | Normalizer /    |
                              | Comparator      |
                              +-----------------+
                                      |
                                      v
                              +-----------------+
                              | Insights +      |
                              | Report (JSON)   |
                              +-----------------+
</code></pre>

<h2>Setup</h2>
<ol>
  <li>Create a Python 3.10+ environment.</li>
  <li>Install requirements:</li>
</ol>
<pre><code>pip install -r requirements.txt</code></pre>
<ol start="3">
  <li>Set API keys in <code>.env</code>.</li>
  <li>Run the script:</li>
</ol>
<pre><code>python multi_ai_compare.py --prompt "Summarize the future of AI in 3 bullet points"</code></pre>

<h2>Requirements</h2>
<pre><code>requests
python-dotenv
numpy
scikit-learn
sentence-transformers
tqdm
</code></pre>

<h2>.env Example</h2>
<pre><code>OPENAI_API_KEY=your_openai_key
COHERE_API_KEY=your_cohere_key
</code></pre>

<h2>Python Implementation (<code>multi_ai_compare.py</code>)</h2>
<pre><code># multi_ai_compare.py
import os, time, json, argparse, requests
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

@dataclass
class NormalizedResponse:
    provider: str
    text: str
    token_count: int
    latency_ms: float
    metadata: Dict[str, Any]

def call_openai_like(prompt: str, model="gpt-4o", max_tokens=256) -> NormalizedResponse:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"user","content":prompt}], "max_tokens":max_tokens}
    start = time.time()
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    latency = (time.time() - start) * 1000
    data = res.json()
    text = data["choices"][0]["message"]["content"].strip()
    tokens = data.get("usage",{}).get("total_tokens",-1)
    return NormalizedResponse("openai", text, tokens, latency, {"raw":data})

def call_cohere_like(prompt: str, model="xlarge", max_tokens=256) -> NormalizedResponse:
    url = "https://api.cohere.ai/generate"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {"model":model,"prompt":prompt,"max_tokens":max_tokens}
    start = time.time()
    res = requests.post(url, headers=headers, json=payload, timeout=30)
    latency = (time.time() - start) * 1000
    data = res.json()
    text = data["generations"][0]["text"].strip()
    return NormalizedResponse("cohere", text, -1, latency, {"raw":data})

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def compare_responses(responses: List[NormalizedResponse]) -> Dict[str,Any]:
    texts = [r.text for r in responses]
    embeddings = EMBED_MODEL.encode(texts, convert_to_numpy=True)
    sim = cosine_similarity(embeddings)
    pairwise = {}
    for i in range(len(responses)):
        for j in range(i+1,len(responses)):
            pairwise[f"{responses[i].provider}_vs_{responses[j].provider}"]=float(sim[i,j])
    best_latency = min(responses,key=lambda r:r.latency_ms).provider
    return {"pairwise_similarity":pairwise,"best_latency_provider":best_latency}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt",type=str,required=True)
    args=parser.parse_args()
    responses=[]
    try: responses.append(call_openai_like(args.prompt))
    except Exception as e: print("OpenAI failed",e)
    try: responses.append(call_cohere_like(args.prompt))
    except Exception as e: print("Cohere failed",e)
    if not responses: return
    comparison=compare_responses(responses)
    report={"responses":[asdict(r) for r in responses],"comparison":comparison}
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
</code></pre>

<h2>Comparison Metrics</h2>
<table>
  <tr><th>Metric</th><th>Description</th></tr>
  <tr><td>Cosine Similarity</td><td>Measures semantic similarity between outputs.</td></tr>
  <tr><td>Latency</td><td>Which provider responds fastest.</td></tr>
  <tr><td>Token Count</td><td>Approximate size / cost of response.</td></tr>
</table>

<h2>Extending</h2>
<ul>
  <li>Add more provider wrappers (Anthropic, Hugging Face Inference API, etc.).</li>
  <li>Implement retries and error handling.</li>
  <li>Store results in a database for analytics.</li>
  <li>Visualize similarity and latency with charts.</li>
</ul>

<h2>License</h2>
<p>MIT License</p>

</body>
</html>

## Result: The corresponding Prompt is executed successfully.
