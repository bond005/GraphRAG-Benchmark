import asyncio
import os
import logging
import argparse
import json
from tqdm import tqdm

from common_utils import (
    SUBSET_PATHS,
    load_corpus_data,
    load_question_data,
    load_existing_results,
    filter_processed_questions,
    save_results_incremental,
    make_error_result,
    ragu_index_exists,
)

from ragu.models.openai import CachedAsyncOpenAI
from ragu.models.llm import LLMOpenAI
from ragu.models.embedder import EmbedderOpenAI
from ragu import KnowledgeGraph, BuilderArguments, Settings, SimpleChunker
from ragu import TwoStageArtifactsExtractorLLM
from ragu.common.prompts import ICLConfig

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def parse_retry_times(value: str) -> tuple[float, ...]:
    return tuple(float(x) for x in value.split(","))


def create_search_engine(engine_type, llm, kg, embedder):
    if engine_type == "local":
        from ragu.search_engine.local_search import LocalSearchEngine
        return LocalSearchEngine(llm=llm, knowledge_graph=kg, embedder=embedder)
    elif engine_type == "global":
        from ragu.search_engine.global_search import GlobalSearchEngine
        return GlobalSearchEngine(llm=llm, knowledge_graph=kg, embedder=embedder)
    elif engine_type == "mix":
        from ragu.search_engine.local_search import LocalSearchEngine
        from ragu.search_engine.global_search import GlobalSearchEngine
        from ragu.search_engine.mix_search import MixSearchEngine
        local = LocalSearchEngine(llm=llm, knowledge_graph=kg, embedder=embedder)
        global_ = GlobalSearchEngine(llm=llm, knowledge_graph=kg, embedder=embedder)
        return MixSearchEngine(llm=llm, engines=[local, global_])
    else:
        raise ValueError(f"Unknown search engine type: {engine_type}")


def convert_gml_to_graphml(working_dir):
    import networkx as nx
    for subdir, dirs, files in os.walk(working_dir):
        for f in files:
            if f == "knowledge_graph.gml":
                gml_path = os.path.join(subdir, f)
                graphml_path = os.path.join(subdir, "knowledge_graph.graphml")
                try:
                    g = nx.read_gml(gml_path)
                    for n, data in g.nodes(data=True):
                        for key, val in data.items():
                            if isinstance(val, (list, dict, tuple, set)):
                                data[key] = json.dumps(val)
                    for u, v, data in g.edges(data=True):
                        for key, val in data.items():
                            if isinstance(val, (list, dict, tuple, set)):
                                data[key] = json.dumps(val)
                    nx.write_graphml(g, graphml_path)
                    logging.info(f"Converted {gml_path} -> {graphml_path}")
                except Exception as e:
                    logging.warning(f"GML->GraphML conversion failed for {gml_path}: {e}")


async def process_corpus(
    corpus_name,
    context,
    base_dir,
    results_dir,
    llm,
    embedder,
    questions,
    sample,
    retrieve_topk,
    search_engine_type,
    icl_enabled=False,
    icl_num_examples=2,
    icl_selection_strategy="semantic",
):
    logging.info(f"Processing corpus: {corpus_name}")

    working_dir = os.path.join(base_dir, corpus_name)
    os.makedirs(working_dir, exist_ok=True)

    output_dir = f"{results_dir}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    Settings.storage_folder = working_dir
    Settings.language = "english"

    chunker = SimpleChunker(max_chunk_size=500, overlap=100)

    icl_config = None
    if icl_enabled:
        icl_config = ICLConfig(
            enabled=True,
            num_examples=icl_num_examples,
            selection_strategy=icl_selection_strategy
        )

    artifact_extractor = TwoStageArtifactsExtractorLLM(
        llm=llm,
        embedder=embedder if icl_enabled else None,
        icl_config=icl_config,
        do_entity_validation=True,
        do_relation_validation=True,
    )
    builder_settings = BuilderArguments(
        use_llm_summarization=True,
        make_community_summary=True,
        remove_isolated_nodes=True,
    )

    kg = KnowledgeGraph(
        llm=llm,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_settings,
    )

    # R1: check if index already exists
    if ragu_index_exists(working_dir):
        logging.info(f"Index already exists for {corpus_name}, skipping indexing")
    else:
        try:
            await kg.build_from_docs([context])
            logging.info(f"Indexed corpus: {corpus_name} ({len(context.split())} words)")
        except Exception as e:
            logging.error(f"Indexing failed for {corpus_name}: {e}")
            if not ragu_index_exists(working_dir):
                logging.error(f"No index available for {corpus_name}, skipping queries")
                return
            logging.info(f"Partial index detected, proceeding with queries")

        convert_gml_to_graphml(working_dir)

    engine = create_search_engine(search_engine_type, llm, kg, embedder)

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    logging.info(f"Found {len(corpus_questions)} questions for {corpus_name}")

    # R3: resume
    existing = load_existing_results(output_path)
    pending = filter_processed_questions(corpus_questions, existing)
    results = list(existing)

    if pending:
        logging.info(f"Resuming: {len(existing)} done, {len(pending)} remaining")
    else:
        logging.info(f"All {len(existing)} questions already processed")

    for q in tqdm(pending, desc=f"Answering questions for {corpus_name}"):
        try:
            if search_engine_type == "local":
                response = await engine.a_query(
                    q["question"],
                    top_k=retrieve_topk,
                    use_summary=True,
                    use_chunks=True,
                )
            else:
                response = await engine.a_query(q["question"])

            context_str = response.retrieval.to_text()
            generated_answer = str(response.response)

            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": context_str,
                "evidence": q["evidence"],
                "question_type": q["question_type"],
                "generated_answer": generated_answer,
                "ground_truth": q.get("answer"),
            })
        except Exception as e:
            logging.error(f"Failed to process question {q.get('id')}: {e}")
            results.append(make_error_result(q, corpus_name, e))

        save_results_incremental(output_path, results)                    # R2

    logging.info(f"Saved {len(results)} predictions to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="RAGU: Process Corpora and Answer Questions"
    )
    parser.add_argument(
        "--subset", required=True, choices=["medical", "novel"],
        help="Subset to process (medical or novel)",
    )
    parser.add_argument(
        "--base_dir", default="./Examples/ragu_workspace",
        help="Base working directory",
    )
    parser.add_argument(
        "--results_dir", default="./Examples/ragu_results",
        help="Directory for generated results",
    )
    parser.add_argument(
        "--model_name", default="gpt-4o-mini",
        help="LLM model name (OpenAI-compatible)",
    )
    parser.add_argument(
        "--embed_model", default="BAAI/bge-large-en-v1.5",
        help="Embedding model name (OpenAI-compatible)",
    )
    parser.add_argument(
        "--embed_size", type=int, default=1024,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--retrieve_topk", type=int, default=5,
        help="Number of top results to retrieve",
    )
    parser.add_argument(
        "--search_engine", default="local",
        choices=["local", "global", "mix"],
        help="Search engine type (default: local)",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Number of questions to sample per corpus",
    )
    parser.add_argument(
        "--llm_base_url", default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API",
    )
    parser.add_argument(
        "--embed_base_url", default=None,
        help="Base URL for embedding API (defaults to --llm_base_url if not set)",
    )
    parser.add_argument(
        "--llm_api_key", default="",
        help="API key (can also use LLM_API_KEY env variable)",
    )
    parser.add_argument(
        "--icl_enabled", action="store_true", default=False,
        help="Enable In-Context Learning (few-shot examples) for extraction",
    )
    parser.add_argument(
        "--icl_num_examples", type=int, default=2,
        help="Number of few-shot examples per extraction call (default: 3)",
    )
    parser.add_argument(
        "--icl_selection_strategy", default="semantic",
        choices=["semantic", "hybrid", "bm25", "random"],
        help="Example selection strategy (default: semantic)",
    )

    parser.add_argument(
        "--llm_rate_max_simultaneous", type=int, default=10,
        help="Max concurrent LLM requests (default: 10)",
    )
    parser.add_argument(
        "--llm_rate_max_per_minute", type=int, default=1000,
        help="Max LLM requests per minute (default: 1000)",
    )
    parser.add_argument(
        "--llm_retry_times", type=parse_retry_times, default="2,4,8,16",
        help="LLM retry wait schedule in seconds, comma-separated (default: 2,4,8,16)",
    )
    parser.add_argument(
        "--embed_rate_max_simultaneous", type=int, default=20,
        help="Max concurrent embedding requests (default: 20)",
    )
    parser.add_argument(
        "--embed_rate_max_per_minute", type=int, default=500,
        help="Max embedding requests per minute (default: 500)",
    )
    parser.add_argument(
        "--embed_retry_times", type=parse_retry_times, default="2,2,2,2",
        help="Embedding retry wait schedule in seconds, comma-separated (default: 2,2,2,2)",
    )
    parser.add_argument(
        "--embed_timeout", type=float, default=120.0,
        help="Per-request timeout for embedding calls in seconds (default: 120)",
    )
    parser.add_argument(
        "--embed_batch_size", type=int, default=500,
        help="Max texts per single embedding API call (default: 500)",
    )
    parser.add_argument(
        "--max_concurrent_embed_batches", type=int, default=5,
        help="Max concurrent embedding batch API calls (default: 5)",
    )
    parser.add_argument(
        "--debug_errors", action="store_true", default=False,
        help="Save failed API call arguments for debugging",
    )

    parser.add_argument(
        "--embedder_token_limit", type=int, default=None,
        help="Max tokens per embedding input (default: Settings.embedder_token_limit = 8192)",
    )
    parser.add_argument(
        "--tokenizer_embedder_backend", default=None,
        choices=["tiktoken", "local"],
        help="Tokenizer backend for embedder truncation (default: Settings default = tiktoken)",
    )
    parser.add_argument(
        "--tokenizer_embedder_name", default=None,
        help="Tokenizer model for embedder truncation (default: Settings default = text-embedding-3-large)",
    )
    parser.add_argument(
        "--llm_context_token_limit", type=int, default=11000,
        help="Max tokens for LLM context in search engines (default: Settings.llm_context_token_limit = 11000)",
    )
    parser.add_argument(
        "--tokenizer_llm_backend", default=None,
        choices=["tiktoken", "local"],
        help="Tokenizer backend for LLM context truncation (default: Settings default = tiktoken)",
    )
    parser.add_argument(
        "--tokenizer_llm_name", default=None,
        help="Tokenizer model for LLM context truncation (default: Settings default = gpt-4o)",
    )

    args = parser.parse_args()

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    if args.embedder_token_limit is not None:
        Settings.embedder_token_limit = args.embedder_token_limit
    if args.tokenizer_embedder_backend is not None:
        Settings.tokenizer_embedder_backend = args.tokenizer_embedder_backend
    if args.tokenizer_embedder_name is not None:
        Settings.tokenizer_embedder_name = args.tokenizer_embedder_name
    if args.llm_context_token_limit is not None:
        Settings.llm_context_token_limit = args.llm_context_token_limit
    if args.tokenizer_llm_backend is not None:
        Settings.tokenizer_llm_backend = args.tokenizer_llm_backend
    if args.tokenizer_llm_name is not None:
        Settings.tokenizer_llm_name = args.tokenizer_llm_name

    try:
        corpus_data = load_corpus_data(corpus_path)                       # R4
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        _, grouped_questions = load_question_data(questions_path)         # R4
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    debug_dir = os.path.join(args.base_dir, "debug") if args.debug_errors else None

    llm_client = CachedAsyncOpenAI(
        base_url=args.llm_base_url,
        api_key=api_key,
        rate_max_simultaneous=args.llm_rate_max_simultaneous,
        rate_max_per_minute=args.llm_rate_max_per_minute,
        retry_times_sec=args.llm_retry_times,
        cache=os.path.join(args.base_dir, "llm_cache"),
        debug_errors_storage=os.path.join(debug_dir, "llm") if debug_dir else None,
    )
    embed_client = CachedAsyncOpenAI(
        base_url=args.embed_base_url or args.llm_base_url,
        api_key=api_key,
        rate_max_simultaneous=args.embed_rate_max_simultaneous,
        rate_max_per_minute=args.embed_rate_max_per_minute,
        retry_times_sec=args.embed_retry_times,
        embed_timeout=args.embed_timeout,
        cache=os.path.join(args.base_dir, "embed_cache"),
        debug_errors_storage=os.path.join(debug_dir, "embed") if debug_dir else None,
    )

    llm = LLMOpenAI(client=llm_client, model_name=args.model_name)
    embedder = EmbedderOpenAI(
        client=embed_client,
        model_name=args.embed_model,
        dim=args.embed_size,
        batch_size=args.embed_batch_size,
        max_concurrent_batches=args.max_concurrent_embed_batches,
        embedder_token_limit=args.embedder_token_limit,
        tokenizer_backend=args.tokenizer_embedder_backend,
        tokenizer_name=args.tokenizer_embedder_name,
    )

    async def _run_all():
        for item in corpus_data:
            await process_corpus(
                corpus_name=item["corpus_name"],
                context=item["context"],
                base_dir=args.base_dir,
                results_dir=args.results_dir,
                llm=llm,
                embedder=embedder,
                questions=grouped_questions,
                sample=args.sample,
                retrieve_topk=args.retrieve_topk,
                search_engine_type=args.search_engine,
                icl_enabled=args.icl_enabled,
                icl_num_examples=args.icl_num_examples,
                icl_selection_strategy=args.icl_selection_strategy,
            )

    asyncio.run(_run_all())


if __name__ == "__main__":
    main()
