import re
import time
from typing import List, Dict, Any, Tuple
import config
from vectordb.store import VectorStore
from utils.console import ConsoleLogger

class RAGEngine:
    """
    Retrieval-Augmented Generation engine strictly grounded in uploaded PDF documents.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def answer_question(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Retrieves relevant PDF excerpts and generates an answer strictly grounded in the context.
        """
        ConsoleLogger.header(f"QUERY: \"{question}\"")
        start_time = time.time()
        ConsoleLogger.query(f"Processing question with top_k={top_k}...")
        
        # 1. Retrieve top matching chunks
        chunks = self.vector_store.query(question, top_k=top_k)

        # If no chunks exist in DB
        if not chunks:
            ConsoleLogger.warning("No chunks available to answer the query.")
            return {
                "answer": "No documents have been uploaded yet, or no relevant text was found. Please upload a PDF to ask questions.",
                "sources": [],
                "provider": "none",
                "is_grounded": False
            }

        # Check if the highest similarity score is above a minimal threshold
        top_score = chunks[0]["similarity_score"]
        if top_score < 0.15:
            ConsoleLogger.warning(f"Top similarity ({top_score*100:.1f}%) below minimal threshold (15.0%). Refusing hallucination.")
            return {
                "answer": "I cannot find the answer to this question in the uploaded PDF documents. (The question does not match any content in the uploaded files).",
                "sources": chunks,
                "provider": config.LLM_PROVIDER,
                "is_grounded": False
            }

        # 2. Format context with clear page & document citations
        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            context_blocks.append(
                f"[Source {idx} - Document: {chunk['filename']}, Page {chunk['page_number']}]\n{chunk['text']}"
            )
        formatted_context = "\n\n".join(context_blocks)

        # 3. Generate grounded response using active provider
        answer, provider_used, is_grounded = self._generate_response(question, formatted_context, chunks)
        elapsed = time.time() - start_time
        
        ConsoleLogger.query(f"Answer generated via [{provider_used}] in {elapsed:.2f}s. (Grounded: {is_grounded})")
        preview = answer.replace("\n", " ")[:120]
        ConsoleLogger.info(f"Answer preview: \"{preview}...\"")

        return {
            "answer": answer,
            "sources": chunks,
            "provider": provider_used,
            "is_grounded": is_grounded
        }


    def _generate_response(self, question: str, context: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str, bool]:
        system_prompt = (
            "You are an expert Document Assistant. Your task is to answer questions strictly and solely "
            "based on the provided context extracted from uploaded PDF documents.\n"
            "Rules you must strictly follow:\n"
            "1. Answer ONLY using facts directly mentioned in the context.\n"
            "2. Cite the source document name and page number for every key claim.\n"
            "3. If the answer cannot be determined strictly from the context, respond with:\n"
            "   'I cannot find the answer to this question in the uploaded PDF documents.'\n"
            "4. Never hallucinate, extrapolate, or bring outside knowledge into your answer."
        )

        user_prompt = f"Context from uploaded PDF documents:\n{context}\n\nQuestion:\n{question}\n\nStrictly Grounded Answer:"

        # Attempt OpenAI if configured
        if config.LLM_PROVIDER == "openai" and config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0
                )
                answer = response.choices[0].message.content.strip()
                is_grounded = "cannot find the answer" not in answer.lower()
                return answer, f"OpenAI ({config.OPENAI_MODEL})", is_grounded
            except Exception as e:
                # Fallback on failure
                pass

        # Attempt Anthropic if configured
        if config.LLM_PROVIDER == "anthropic" and config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
                message = client.messages.create(
                    model=config.ANTHROPIC_MODEL,
                    max_tokens=1024,
                    temperature=0.0,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                answer = message.content[0].text.strip()
                is_grounded = "cannot find the answer" not in answer.lower()
                return answer, f"Anthropic ({config.ANTHROPIC_MODEL})", is_grounded
            except Exception as e:
                # Fallback on failure
                pass

        # Built-in Offline Extractive Generator (standalone, zero external API keys needed)
        return self._offline_extractive_generation(question, chunks)

    def _offline_extractive_generation(self, question: str, chunks: List[Dict[str, Any]]) -> Tuple[str, str, bool]:
        """
        Synthesizes a strict grounded response offline directly from top chunks by matching query tokens.
        """
        q_words = set(re.findall(r'\w+', question.lower()))
        # Remove common stop words
        stop_words = {"what", "is", "are", "the", "a", "an", "how", "why", "when", "where", "who", "which", "in", "on", "of", "to", "for", "with", "does", "do", "can"}
        keywords = q_words - stop_words
        if not keywords:
            keywords = q_words

        candidate_sentences = []
        for chunk in chunks:
            filename = chunk["filename"]
            page = chunk["page_number"]
            sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
            
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) < 15:
                    continue
                s_words = set(re.findall(r'\w+', sent_clean.lower()))
                overlap = len(keywords.intersection(s_words))
                if overlap > 0:
                    candidate_sentences.append({
                        "sentence": sent_clean,
                        "filename": filename,
                        "page": page,
                        "overlap": overlap
                    })

        if not candidate_sentences:
            return (
                "I cannot find the answer to this question in the uploaded PDF documents.",
                "Offline Engine (Extractive Grounded)",
                False
            )

        # Sort by keyword overlap descending
        candidate_sentences.sort(key=lambda x: x["overlap"], reverse=True)
        top_sentences = candidate_sentences[:3]

        answer_lines = ["Based on the uploaded documents:"]
        seen = set()
        for c in top_sentences:
            if c["sentence"] not in seen:
                seen.add(c["sentence"])
                answer_lines.append(f"• {c['sentence']} (Source: {c['filename']}, Page {c['page']})")

        answer_text = "\n\n".join(answer_lines)
        return answer_text, "Offline Engine (Extractive Grounded)", True
