import os
import asyncio
import argparse
import logging
from typing import Dict, List
from dotenv import load_dotenv
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

from common_utils import (
    SUBSET_PATHS,
    load_corpus_data,
    load_question_data,
    load_existing_results,
    filter_processed_questions,
    save_results_incremental,
    make_error_result,
    hipporag_index_exists,
)

load_dotenv()

import hipporag

DSPY_FILTER_PATH = os.path.join(
    os.path.dirname(hipporag.__file__),
    "prompts", "dspy_prompts", "filter_llama3.3-70B-Instruct.json",
)

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.misc_utils import string_to_bool
from hipporag.utils.config_utils import BaseConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hipporag_processing.log")
    ]
)


def split_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_token_size: int = 256,
    chunk_overlap_token_size: int = 32
) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks


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

    output_dir = f"./results/hipporag2/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    if openai_emb:
        tokenizer_name = "bert-base-uncased"
    else:
        tokenizer_name = embed_model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logging.info(f"Loaded tokenizer: {tokenizer_name}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}")
        return

    chunks = split_text(context, tokenizer)
    logging.info(f"Split corpus into {len(chunks)} chunks")

    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"Found {len(corpus_questions)} questions for {corpus_name}")

    embedding_model_name = embed_model_path if not openai_emb else "text-embedding-3-small"

    # H1/H2: check if graph already exists before indexing (Approach A)
    save_dir = os.path.join(base_dir, corpus_name)
    index_exists = hipporag_index_exists(save_dir, model_name, embedding_model_name)

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=embedding_model_name,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path=DSPY_FILTER_PATH,
        retrieval_top_k=5,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )

    if mode == "ollama":
        config.llm_mode = "ollama"
        logging.info(f"Using Ollama mode: {model_name} at {llm_base_url}")
    else:
        config.llm_mode = "openai"
        logging.info(f"Using OpenAI mode: {model_name} at {llm_base_url}")

    if openai_emb:
        if hasattr(config, 'embedding_dim'):
            config.embedding_dim = embed_size
        logging.info(f"Using OpenAI embedding: {embedding_model_name} (dim={embed_size})")

    hipporag = HippoRAG(global_config=config)

    # H3/H4: skip indexing if graph already exists
    if index_exists:
        logging.info(f"Index already exists for {corpus_name}, skipping indexing")
    else:
        try:
            hipporag.index(docs)
            logging.info(f"Indexed corpus: {corpus_name}")
        except Exception as e:
            logging.error(f"Indexing failed for {corpus_name}: {e}")
            if not hipporag_index_exists(save_dir, model_name, embedding_model_name):
                logging.error(f"No index available for {corpus_name}, skipping queries")
                return
            logging.info(f"Partial index detected for {corpus_name}, proceeding with queries")

    # H6/H7: resume + incremental save
    existing = load_existing_results(output_path)
    pending = filter_processed_questions(corpus_questions, existing)
    results = list(existing)

    if pending:
        logging.info(f"Resuming: {len(existing)} done, {len(pending)} remaining")
    else:
        logging.info(f"All {len(existing)} questions already processed")

    # H5: per-question processing
    for q in tqdm(pending, desc=f"Answering questions for {corpus_name}"):
        try:
            query_solutions, _, _, _, _ = hipporag.rag_qa(
                queries=[q["question"]],
                gold_answers=[[q["answer"]]]
            )
            solution = query_solutions[0].to_dict()
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": solution.get("docs", ""),
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"Failed to process question {q.get('id')}: {e}")
            results.append(make_error_result(q, corpus_name, e))
        save_results_incremental(output_path, results)

    logging.info(f"Saved {len(results)} predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")

    parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./hipporag2_workspace",
                        help="Base working directory for HippoRAG")

    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="gpt-4o-mini",
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/contriever",
                        help="Path to embedding model directory")
    parser.add_argument("--embed_size", type=int, default=768,
                        help="Embedding dimension (used with --openai_emb)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")
    parser.add_argument("--openai_emb", action="store_true", help="Use OpenAI-compatible API for embeddings instead of local HuggingFace")

    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1",
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="",
                        help="API key for LLM service (can also use OPENAI_API_KEY environment variable)")

    args = parser.parse_args()

    logging.info(f"Starting HippoRAG processing for subset: {args.subset}")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")

    if args.openai_emb and api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    os.makedirs(args.base_dir, exist_ok=True)

    try:
        corpus_data = load_corpus_data(corpus_path)
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        _, grouped_questions = load_question_data(questions_path)
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
