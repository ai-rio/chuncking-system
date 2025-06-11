**Subject: Task: Concrete Steps for Implementing Core RAG Evaluation Framework**

Welcome! Let's get Ragas up and running in our Colab environment. We understand that Colab can have its quirks, so this guide provides explicit, runnable steps to implement the core evaluation framework.


### **1. Objective: Implement Core RAG Evaluation Framework**

The goal is to integrate **Ragas** to quantitatively measure our RAG pipeline's performance.


### **2. Concrete Steps for Implementation**

Follow these steps precisely.


#### **2.1. Update requirements.txt and Install Dependencies**

First, ensure all necessary libraries are installed.

1. **Add/Update requirements.txt:**

- Open requirements.txt in your project.

- Add the following lines (or ensure they have these exact versions):\
  ragas==0.1.1\
  datasets==2.19.1\
  langchain-core==0.1.52\
  langchain-community==0.0.30\
  langchain==0.1.18\
  openai==1.28.1\
  tiktoken==0.6.0\
  sentence-transformers==2.7.0\
  torch==2.3.0+cu121  # IMPORTANT: Match this to your Colab's CUDA version or use a compatible one\
  torchaudio==2.3.0+cu121\
  torchvision==0.18.0+cu121\
  pydantic-settings==2.2.1\
  \# Add any other dependencies you already have

- **Note on torch:** Colab often uses specific CUDA versions. To find the exact torch version that matches your Colab runtime's CUDA, run !nvidia-smi to see the CUDA version, then visit [PyTorch's website](https://pytorch.org/get-started/locally/) to get the correct installation command for your CUDA version. Adjust torch==, torchaudio==, torchvision== accordingly in requirements.txt.

2. **Install from requirements.txt in Colab:**

- In a Colab cell, run:\
  %pip install -r requirements.txt

- _Troubleshooting:_ If you encounter dependency conflicts, try %pip install -r requirements.txt --upgrade or %pip install -r requirements.txt --force-reinstall. If issues persist, temporarily remove problematic lines and install them one by one.


#### **2.2. Prepare Evaluation Data (question, ground\_truths, contexts, answer)**

We will create a small, synthetic dataset for initial evaluation.

1. **Choose a Sample Chunk:** Select one or two well-chunked text sections from data/output/chunks/sample\_image\_document\_chunks.json that you want to evaluate. For instance, the 'Introduction to AI' chunk or a specific content chunk.

2. **Generate Synthetic Data:**

- **Modify main.py temporarily** (or create a new script evaluation\_test.py) to generate this data programmatically using our LLM.

- **Key Idea:** Use the MetadataEnricher (or a direct genai call) to create a question based on a chunk's page\_content and then generate a synthetic answer. The chunk's page\_content itself will serve as the contexts.

\# Example snippet to add to main.py's main\_async function, before evaluation\
\# This is for DEMONSTRATION; refine to generate multiple samples as needed.\
\
\# --- START RAGAS EVALUATION SETUP ---\
from datasets import Dataset # Import Dataset from huggingface datasets\
import google.generativeai as genai\
import asyncio\
\
\# Ensure Gemini API key is set for this generation part\
if not config.GEMINI\_API\_KEY:\
    print("Warning: GEMINI\_API\_KEY not set for Ragas data generation. Using mock data.")\
    # Mock data for demonstration if API key is truly missing\
    eval\_data\_points = \[\
        {"question": "What is AI?", "ground\_truths": \["AI is transforming many aspects of our lives."], "contexts": \["Artificial intelligence (AI) is transforming many aspects of our lives. From smart assistants to complex data analysis, AI is becoming increasingly prevalent."], "answer": "AI is becoming increasingly prevalent and transforms many aspects of our lives."}\
    ]\
else:\
    genai.configure(api\_key=config.GEMINI\_API\_KEY)\
    model\_name = config.LLM\_METADATA\_MODEL # Or a more powerful model like gemini-1.5-flash for QA\
    qa\_model = genai.GenerativeModel(model\_name)\
\
    eval\_data\_points = \[]\
    # Take the first few text chunks for evaluation\
    num\_eval\_chunks = min(3, len(enriched\_chunks)) # Evaluate first 3 or fewer if not enough chunks\
\
    print(f"\n--- Generating synthetic evaluation data for {num\_eval\_chunks} chunks ---")\
    for i in range(num\_eval\_chunks):\
        chunk = enriched\_chunks\[i]\
        if chunk.metadata.get('source\_segment\_type') == 'text' or chunk.metadata.get('chunk\_type') == 'prose':\
            context\_text = chunk.page\_content.strip()\
\
            # Generate a question from the context\
            try:\
                question\_prompt = f"Generate a concise factual question that can be answered ONLY from the following text:\n\nText: {context\_text}\n\nQuestion:"\
                response\_q = await asyncio.to\_thread(qa\_model.generate\_content, \[{"role": "user", "parts": \[{"text": question\_prompt}]}])\
                question\_text = response\_q.candidates\[0].content.parts\[0].text.strip()\
                # Remove any markdown formatting around the question\
                question\_text = re.sub(r'^\["\\'\`]\*(.\*?)(\["\\'\`]\*)$', r'\1', question\_text).strip()\
                if question\_text.startswith("Question:"):\
                    question\_text = question\_text\[len("Question:"):].strip()\
                if question\_text.endswith("?"): # Basic validation\
                    question = question\_text\
                else:\
                    question = question\_text + "?" # Ensure it's a question\
\
\
                # Generate an answer from the context for the question\
                answer\_prompt = f"Based ONLY on the following text, answer the question. If the answer is not in the text, state 'Not found in text.'\n\nText: {context\_text}\n\nQuestion: {question}\n\nAnswer:"\
                response\_a = await asyncio.to\_thread(qa\_model.generate\_content, \[{"role": "user", "parts": \[{"text": answer\_prompt}]}])\
                answer\_text = response\_a.candidates\[0].content.parts\[0].text.strip()\
\
                # Add data point\
                eval\_data\_points.append({\
                    "question": question,\
                    "ground\_truths": \[answer\_text], # For simplicity, synthetic answer is ground truth\
                    "contexts": \[context\_text],\
                    "answer": answer\_text\
                })\
                print(f"  Generated Q/A for chunk {i}: '{question\_text\[:50]}...'")\
            except Exception as e:\
                print(f"  Error generating Q/A for chunk {i}: {e}")\
                # Skip this data point if generation fails\
                continue\
        else:\
            print(f"  Skipping non-text chunk {i} for Ragas Q/A generation.")\
\
if not eval\_data\_points:\
    print("No evaluation data points generated. Skipping Ragas evaluation.")\
    return # Exit main\_async if no data\
\
\# Create Hugging Face Dataset\
eval\_dataset = Dataset.from\_list(eval\_data\_points)\
print("\n--- Evaluation dataset created ---")\
print(eval\_dataset)\
\# --- END RAGAS EVALUATION SETUP ---


#### **2.3. Integrate Ragas Evaluation**

Now, use the eval\_dataset with Ragas.

1. **Add Ragas Import and Evaluation Code:**

- Continue modifying main.py (or your evaluation script) after the dataset generation part.

\# --- START RAGAS EVALUATION EXECUTION ---\
from ragas import evaluate\
from ragas.metrics import (\
    context\_precision,\
    context\_recall,\
    faithfulness,\
    answer\_relevance,\
    # You can add other metrics as needed, e.g.,\
    # answer\_correctness,\
    # answer\_similarity,\
)\
\
print("\n--- Starting Ragas evaluation ---")\
try:\
    result = evaluate(\
        eval\_dataset,\
        metrics=\[\
            context\_precision,\
            context\_recall,\
            faithfulness,\
            answer\_relevance,\
        ],\
        llm=genai.GenerativeModel(model\_name), # Pass the LLM to Ragas\
        embeddings=chunker.embedding\_model # Pass the embedding model to Ragas if needed by metrics\
    )\
\
    print("\n--- Ragas Evaluation Results ---")\
    print(result) # Prints a pandas DataFrame\
\
    # Convert to dictionary for easier logging/saving if desired\
    result\_df = result.to\_pandas()\
    print("\n--- Ragas Evaluation DataFrame ---")\
    print(result\_df)\
\
    # You can save the results to a file here if needed\
    # result\_df.to\_csv("ragas\_evaluation\_results.csv", index=False)\
    # print("\nRagas evaluation results saved to ragas\_evaluation\_results.csv")\
\
except Exception as e:\
    print(f"\nCRITICAL ERROR during Ragas evaluation: {e}")\
    traceback.print\_exc()\
\# --- END RAGAS EVALUATION EXECUTION ---


#### **2.4. Colab Specific Troubleshooting**

- **Runtime Reset:** If Python seems to ignore code changes (especially requirements.txt or module imports), perform Runtime > Restart session in Colab.

- **PYTHONPATH:** The initial Colab setup cells you provided (mounting Drive, setting PYTHONPATH, os.chdir) are crucial. Ensure they are run first every time you open the notebook or restart the runtime.

- **API Key:** Confirm your config.GEMINI\_API\_KEY is correctly set in src/config/settings.py or via your .env file within the Colab environment. Without it, LLM calls will use mock data or fail.


### **3. Expected Outcome & Reporting**

Upon successful execution, you should see:

- Logs indicating synthetic Q\&A generation.

- A Dataset object printed.

- The ragas evaluation results (a pandas DataFrame).

- A summary of metrics like context\_precision, context\_recall, faithfulness, and answer\_relevance.

Your precise observations on these metrics will inform our next steps for optimizing the RAG system. Let's tackle this!
