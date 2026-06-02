import json
import logging
import os
from typing import Any

from datasets import load_dataset


def load_corpus_data(corpus_path: str) -> list[dict]:
    dataset = load_dataset("parquet", data_files=corpus_path, split="train")
    corpus_data = [
        {"corpus_name": item["corpus_name"], "context": item["context"]}
        for item in dataset
    ]
    logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    return corpus_data


def load_question_data(questions_path: str) -> tuple[list[dict], dict[str, list[dict]]]:
    dataset = load_dataset("parquet", data_files=questions_path, split="train")
    question_data = []
    for item in dataset:
        question_data.append({
            "id": item["id"],
            "source": item["source"],
            "question": item["question"],
            "answer": item["answer"],
            "question_type": item["question_type"],
            "evidence": item["evidence"],
        })
    grouped = group_questions_by_source(question_data)
    logging.info(
        f"Loaded questions with {len(question_data)} entries from {questions_path}"
    )
    return question_data, grouped


def group_questions_by_source(question_list: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(question)
    return grouped


SUBSET_PATHS = {
    "medical": {
        "corpus": "./Datasets/Corpus/medical.parquet",
        "questions": "./Datasets/Questions/medical_questions.parquet",
    },
    "novel": {
        "corpus": "./Datasets/Corpus/novel.parquet",
        "questions": "./Datasets/Questions/novel_questions.parquet",
    },
}


def load_existing_results(output_path: str) -> list[dict]:
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"Could not load existing results from {output_path}: {e}")
        return []


def filter_processed_questions(
    questions: list[dict], existing_results: list[dict]
) -> list[dict]:
    done_ids = {r["id"] for r in existing_results if "id" in r}
    return [q for q in questions if q["id"] not in done_ids]


def save_results_incremental(output_path: str, results: list[dict]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def make_error_result(
    question: dict, corpus_name: str, error: Any
) -> dict:
    return {
        "id": question.get("id", ""),
        "question": question.get("question", ""),
        "source": corpus_name,
        "context": "",
        "evidence": question.get("evidence", ""),
        "question_type": question.get("question_type", ""),
        "generated_answer": "",
        "ground_truth": question.get("answer", ""),
        "error": str(error),
    }


def lightrag_index_exists(working_dir: str) -> bool:
    graph_file = os.path.join(working_dir, "graph_chunk_entity_relation.graphml")
    doc_status = os.path.join(working_dir, "kv_store_doc_status.json")
    return os.path.exists(graph_file) and os.path.exists(doc_status)


def fast_graphrag_index_exists(working_dir: str) -> bool:
    graph_file = os.path.join(working_dir, "graph_igraph_data.pklz")
    chunks_file = os.path.join(working_dir, "chunks_kv_data.pkl")
    return os.path.exists(graph_file) and os.path.exists(chunks_file)


def hipporag_index_exists(save_dir: str, llm_name: str, embed_name: str) -> bool:
    working_dir = os.path.join(save_dir, f"{llm_name}_{embed_name}")
    graph_file = os.path.join(working_dir, "graph.pickle")
    return os.path.exists(graph_file)


def digimon_index_exists(working_dir: str) -> bool:
    graph_file = os.path.join(working_dir, "graph_storage_nx_data.graphml")
    return os.path.exists(graph_file)


def ragu_index_exists(working_dir: str) -> bool:
    for name in ("knowledge_graph.graphml", "knowledge_graph.gml"):
        if os.path.exists(os.path.join(working_dir, name)):
            return True
    return False
