import asyncio
import os
import logging
import argparse
from typing import Dict, List
from dotenv import load_dotenv
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService, HuggingFaceEmbeddingService
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from common_utils import (
    SUBSET_PATHS,
    load_corpus_data,
    load_question_data,
    load_existing_results,
    filter_processed_questions,
    save_results_incremental,
    make_error_result,
    fast_graphrag_index_exists,
)
from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper

load_dotenv()

DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]


def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_path: str,
    embed_size: int,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int,
    openai_emb: bool = False
):
    logging.info(f"Processing corpus: {corpus_name}")

    output_dir = f"./results/fast-graphrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    working_dir = os.path.join(base_dir, corpus_name)

    if openai_emb:
        from fast_graphrag._llm import OpenAIEmbeddingService
        embedding_service = OpenAIEmbeddingService(
            model=embed_model_path,
            base_url=llm_base_url,
            api_key=llm_api_key,
            embedding_dim=embed_size,
        )
        logging.info(f"Using OpenAI-compatible embedding service: {embed_model_path}")
    else:
        try:
            embedding_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
            embedding_model = AutoModel.from_pretrained(embed_model_path)
            logging.info(f"Loaded embedding model: {embed_model_path}")
        except Exception as e:
            logging.error(f"Failed to load embedding model: {e}")
            return
        embedding_service = HuggingFaceEmbeddingService(
            model=embedding_model,
            tokenizer=embedding_tokenizer,
            embedding_dim=embed_size,
            max_token_size=8192
        )

    if mode == "ollama":
        ollama_client = OllamaClient(base_url=llm_base_url)
        llm_service = OllamaWrapper(ollama_client, model_name)
        logging.info(f"Using Ollama LLM service: {model_name} at {llm_base_url}")
    else:
        llm_service = OpenAILLMService(
            model=model_name,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        logging.info(f"Using OpenAI-compatible LLM service: {model_name} at {llm_base_url}")

    grag = GraphRAG(
        working_dir=working_dir,
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service,
            n_checkpoints=1,                                               # F2
        ),
    )

    # F1: check if index already exists
    if fast_graphrag_index_exists(working_dir):
        logging.info(f"Index already exists for {corpus_name}, skipping indexing")
    else:
        try:                                                               # F3
            grag.insert(context)
            logging.info(f"Indexed corpus: {corpus_name} ({len(context.split())} words)")
        except Exception as e:
            logging.error(f"Indexing failed for {corpus_name}: {e}")
            if not fast_graphrag_index_exists(working_dir):
                logging.error(f"No index available for {corpus_name}, skipping queries")
                return
            logging.info(f"Partial index detected for {corpus_name}, proceeding with queries")

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"Found {len(corpus_questions)} questions for {corpus_name}")

    # F6: resume
    existing = load_existing_results(output_path)
    pending = filter_processed_questions(corpus_questions, existing)
    results = list(existing)

    if pending:
        logging.info(f"Resuming: {len(existing)} done, {len(pending)} remaining")
    else:
        logging.info(f"All {len(existing)} questions already processed")

    for q in tqdm(pending, desc=f"Answering questions for {corpus_name}"):
        try:
            response = grag.query(q["question"])
            context_chunks = response.to_dict()['context']['chunks']
            contexts = [item[0]["content"] for item in context_chunks]
            predicted_answer = response.response

            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"Failed to process question {q.get('id')}: {e}")
            results.append(make_error_result(q, corpus_name, e))          # F4
        save_results_incremental(output_path, results)                    # F5

    logging.info(f"Saved {len(results)} predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")

    parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./Examples/graphrag_workspace",
                        help="Base working directory for GraphRAG")

    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct",
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/bge-large-en-v1.5",
                        help="Path to embedding model directory")
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Embedding dimension")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")
    parser.add_argument("--openai_emb", action="store_true", help="Use OpenAI-compatible API for embeddings instead of local HuggingFace")

    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="",
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log")
        ]
    )

    logging.info(f"Starting GraphRAG processing for subset: {args.subset}")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    try:
        corpus_data = load_corpus_data(corpus_path)                       # F7
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        _, grouped_questions = load_question_data(questions_path)         # F7
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    async def _run_all():
        tasks = []
        for item in corpus_data:
            tasks.append(asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.mode,
                args.model_name,
                args.embed_model_path,
                args.embed_size,
                args.llm_base_url,
                api_key,
                grouped_questions,
                args.sample,
                args.openai_emb,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"Task failed: {r}")

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
