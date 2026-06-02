import os
import asyncio
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

from common_utils import (
    SUBSET_PATHS,
    load_corpus_data,
    load_question_data,
    load_existing_results,
    filter_processed_questions,
    save_results_incremental,
    make_error_result,
    digimon_index_exists,
)

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_rag(
    config_path: Path,
    source: str,
    mode: str = "config",
    model_name: str = None,
    llm_base_url: str = None,
    llm_api_key: str = None
) -> GraphRAG:
    logger.info(f"Initializing GraphRAG for source: {source}")

    opt = Config.parse(config_path, dataset_name=source)

    if mode == "ollama":
        if hasattr(opt, 'llm_config'):
            opt.llm_config.model_name = model_name
            opt.llm_config.base_url = llm_base_url
            opt.llm_config.api_key = llm_api_key
            opt.llm_config.mode = "ollama"
        logger.info(f"Ollama configuration: model={model_name}, base_url={llm_base_url}")
    else:
        logger.info(f"Configuration parsed: {opt}")

    rag = GraphRAG(config=opt)
    logger.info(f"GraphRAG initialized for {source}")
    return rag


async def process_corpus(
    rag: GraphRAG,
    corpus_name: str,
    context: str,
    questions: Dict[str, List[dict]],
    sample: int,
    output_dir: str = "./results/GraphRAG"
):
    logger.info(f"Processing corpus: {corpus_name}")

    corpus = [{
        "title": corpus_name,
        "content": context,
        "doc_id": 0,
    }]

    # D6: try/except around insert
    try:
        await rag.insert(corpus)
        logger.info(f"Indexed corpus: {corpus_name} ({len(context.split())} words)")
    except Exception as e:
        logger.error(f"Indexing failed for {corpus_name}: {e}")
        logger.info(f"Attempting to continue with existing index artifacts")

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logger.warning(f"No questions found for corpus: {corpus_name}")
        return

    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
        logger.info(f"Sampled {sample} questions from {len(corpus_questions)} total")

    logger.info(f"Found {len(corpus_questions)} questions for {corpus_name}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{corpus_name}_predictions.json")

    # D8: resume
    existing = load_existing_results(output_path)
    pending = filter_processed_questions(corpus_questions, existing)
    results = list(existing)

    if pending:
        logger.info(f"Resuming: {len(existing)} done, {len(pending)} remaining")
    else:
        logger.info(f"All {len(existing)} questions already processed")

    for q in tqdm(pending, desc=f"Answering questions for {corpus_name}"):
        try:
            # D5: query() returns str, not tuple
            response = await rag.query(q["question"])
            results.append({
                "id": q["id"],
                "source": corpus_name,
                "question": q["question"],
                "context": "",
                "generated_answer": response,
                "evidence": q.get("evidence", ""),                       # D1
                "ground_truth": q.get("answer"),
                "question_type": q.get("question_type", "unknown")
            })
        except Exception as e:
            logger.error(f"Failed to process question {q['id']}: {e}")
            results.append(make_error_result(q, corpus_name, e))          # D2
        save_results_incremental(output_path, results)

    logger.info(f"Saved {len(results)} predictions to: {output_path}")


async def _run_all(rag, corpus_data, grouped_questions, args):
    for item in corpus_data:
        await process_corpus(
            rag=rag,
            corpus_name=item["corpus_name"],
            context=item["context"],
            questions=grouped_questions,
            sample=args.sample,
            output_dir=args.output_dir
        )


def main():
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")

    parser.add_argument("--subset", required=True, choices=["medical", "novel"],
                        help="Subset to process")
    parser.add_argument("--option", default="./config.yml",
                        help="Path to configuration YAML file")           # D4
    parser.add_argument("--output_dir", default="./results/GraphRAG",
                        help="Output directory for results")

    parser.add_argument("--mode", choices=["config", "ollama"], default="config",
                        help="Use config file or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct",
                        help="LLM model identifier (for ollama mode)")
    parser.add_argument("--llm_base_url", default="http://localhost:11434",
                        help="Base URL for LLM API (for ollama mode)")
    parser.add_argument("--llm_api_key", default="",
                        help="API key for LLM service (not needed for ollama)")

    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")

    args = parser.parse_args()

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    try:
        corpus_data = load_corpus_data(corpus_path)                       # D9
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        _, grouped_questions = load_question_data(questions_path)         # D9
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return

    rag = asyncio.run(
        initialize_rag(
            config_path=Path(args.option),                                # D4
            source=args.subset,
            mode=args.mode,
            model_name=args.model_name,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key
        )
    )

    # D7: single async context for all corpora
    asyncio.run(
        _run_all(rag, corpus_data, grouped_questions, args)
    )


if __name__ == "__main__":
    main()
