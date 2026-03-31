from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
import re
from langchain_community.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

def generate_queries(query,llm):
    prompt = f"""
    You are a JIRA expert.

    Rewrite the query into 4 search queries to retrieve relevant issues.

    Focus on:
    - unresolved tickets
    - sprint carry forward issues
    - incomplete stories
    - pending tasks
    - backlog spillover
    - issues not resolved in sprint

    Query: {query}
    """

    response = llm.invoke(prompt)
    return [q.strip() for q in response.content.split("\n") if q.strip()]

def hybrid_search_with_scores(query, vectorstore,k=40):
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    docs_from_vectorstore = [Document(page_content=doc.page_content, metadata=doc.metadata)
                         for doc in vectorstore.similarity_search("", k=100)]

# Build BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(docs_from_vectorstore)
    bm25_retriever.k = 20
    dense_results = dense_retriever.invoke(query)
    sparse_results = bm25_retriever.invoke(query)

    scores = {}
    doc_map = {}

    # Dense scoring
    for rank, doc in enumerate(dense_results):
        key = doc.page_content
        score = 0.7 / (rank + 1)
        scores[key] = scores.get(key, 0) + score
        doc_map[key] = doc

    # Sparse scoring
    for rank, doc in enumerate(sparse_results):
        key = doc.page_content
        score = 0.3 / (rank + 1)
        scores[key] = scores.get(key, 0) + score
        doc_map[key] = doc

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [(doc_map[key], score) for key, score in sorted_docs[:k]]

def dynamic_top_k(results, query, max_k=10):
    filtered_docs = []

    # 🔥 RELAX threshold for analysis queries
    if any(word in query.lower() for word in ["why", "reason", "analysis"]):
        threshold = 0.05   # 🔥 very important change
    else:
        threshold = 0.2

    for doc, score in results:
        if score >= threshold:
            filtered_docs.append(doc)

        if len(filtered_docs) >= max_k:
            break

    return filtered_docs

def rerank_documents(query, docs, top_k=5):
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_k]]

def retrieve_docs(query,apiKey):

    embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True})

    # Load existing Chroma DB
    vectorstore = Chroma(
    persist_directory=".\chroma_db_backup",  # your stored folder
    embedding_function=embedding)
    llm = ChatOpenAI(model="gpt-4",api_key=apiKey)

    # ==============================
    # 🔥 STEP 0: DATE QUERY HANDLING
    # ==============================
    print("1")
    date_match = re.search(r"\d{1,2}/[A-Za-z]{3}/\d{4}", query)

    if date_match:
        target_date = date_match.group()

        docs = vectorstore.similarity_search("", k=200)

        filtered_docs = []

        for doc in docs:
            comments = doc.page_content or ""

            if target_date.lower() in comments.lower():
                filtered_docs.append(doc)

        if len(filtered_docs) == 0:
            return f"No issues found with comments on {target_date}"

        # Return unique issue keys
        issue_keys = list(set([
            doc.metadata.get("issue_key") for doc in filtered_docs
        ]))

        return "\n".join(issue_keys)


    # ==============================
    # 🔹 STEP 1: TICKET QUERY
    # ==============================
    match = re.search(r"[A-Z]+-\d+", query)

    if match:
        issue_key = match.group()

        docs = vectorstore.similarity_search(
            query="",
            filter={"issue_key": issue_key},
            k=5
        )

        if len(docs) == 0:
            return f"Issue {issue_key} not found in dataset."


    # ==============================
    # 🔹 STEP 2: SEMANTIC QUERY (RAG)
    # ==============================
    else:
        queries = generate_queries(query,llm)[:3]

        all_docs = []

        for q in queries:
            results = hybrid_search_with_scores(q, vectorstore, k=40)

            filtered_docs = dynamic_top_k(results, query, max_k=10)

            all_docs.extend(filtered_docs)

        # 🔹 Fallback
        if len(all_docs) == 0:
            print("⚠️ Fallback triggered")

            fallback = hybrid_search_with_scores(query, k=5)
            all_docs = [doc for doc, _ in fallback]

        # 🔹 Deduplicate
        clean_docs = [
            doc if not isinstance(doc, tuple) else doc[0]
            for doc in all_docs
        ]

        unique_docs = list({
            doc.page_content: doc for doc in clean_docs
        }.values())

        # 🔹 Re-rank
        docs = rerank_documents(query, unique_docs, top_k=5)



    # ==============================
    # 🔹 STEP 3: BUILD CONTEXT
    # ==============================
    context = ""

    for doc in docs:
        context += f"""
Issue Key: {doc.metadata.get('issue_key')}
Issue Type: {doc.metadata.get('issue_type')}
Status: {doc.metadata.get('status')}
Project name: {doc.metadata.get('project')}
Project type: {doc.metadata.get('Project type')}
Project url: {doc.metadata.get('Project url')}
Priority: {doc.metadata.get('priority')}
Resolution: {doc.metadata.get('Resolution')}
Created: {doc.metadata.get('Created')}
Updated: {doc.metadata.get('Updated')}
Last Viewed: {doc.metadata.get('Last Viewed')}
Resolved: {doc.metadata.get('Resolved')}
Custom field (Symptom Severity): {doc.metadata.get('Custom field (Symptom Severity)')}

Description: {doc.page_content}
"""


    # ==============================
    # 🔹 STEP 4: INTENT DETECTION
    # ==============================
    if any(word in query.lower() for word in ["why", "reason", "analysis"]):
        mode = "analysis"
    else:
        mode = "extraction"


    # ==============================
    # 🔹 STEP 5: PROMPT
    # ==============================
    if mode == "analysis":
        prompt = f"""
You are a JIRA analyst.

Analyze the context and identify possible reasons.

Look for:
- unresolved tickets
- pending status
- missing resolution
- repeated updates
- high priority not closed

If exact reason is not explicitly stated, infer from patterns.

CONTEXT:
{context}

QUESTION:
{query}

Give a concise answer.
"""
    else:
        prompt = f"""
You are a JIRA structured data assistant.

Extract ONLY requested fields.

RULES:
- Use only provided context
- Do not infer
- Do not add extra text

CONTEXT:
{context}

QUESTION:
{query}

OUTPUT FORMAT:
<Field Name>: <Exact Value>

If not found:
No issue found.
"""

    response = llm.invoke(prompt)
    return response.content
