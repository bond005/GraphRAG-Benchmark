# AGENTS.md

## Project overview

GraphRAG-Bench is a research benchmark for evaluating Graph Retrieval-Augmented Generation models. It ships datasets, example runner scripts for various GraphRAG frameworks, and a shared evaluation pipeline. There is no library code to build, no test suite, and no CI.

## Running commands

All scripts expect to be run from the repo root. Paths like `./Datasets/Corpus/` are relative to the project root.

Install shared dependencies:

    pip install -r requirements.txt

The README recommends separate conda environments per framework (e.g., `conda create -n lightrag python=3.10`). Each framework may also require source code patches — see `Examples/README.md` for details.

### Evaluation commands

All three evaluation entrypoints use `python -m`:

    python -m Evaluation.generation_eval --mode API --data_file <results.json> --output_file <out.json>
    python -m Evaluation.retrieval_eval  --mode API --data_file <results.json> --output_file <out.json>
    python -m Evaluation.indexing_eval   --framework lightrag --base_path ./Examples/lightrag_workspace

Common flags: `--mode API|ollama`, `--model`, `--base_url`, `--embedding_model`, `--sample N` (limit dataset size for quick runs).

API mode requires `LLM_API_KEY` env var (all scripts) or `OPENAI_API_KEY` (hipporag2 only).

### Example runner commands

Run from `Examples/` directory with framework-specific scripts:

    python run_lightrag.py --subset medical --mode API --base_dir ./Examples/lightrag_workspace
    python run_fast-graphrag.py --subset medical --base_dir ./Examples/fast-graphrag_workspace
    python run_hipporag2.py --subset medical --mode API --base_dir ./Examples/hipporag2_workspace

DIGIMON is special: copy `run_digimon.py` into the DIGIMON project repo, then run it there.

## Architecture

    Datasets/
      Corpus/          # medical.json/.parquet, novel.json/.parquet
      Questions/       # medical_questions.json/.parquet, novel_questions.json/.parquet
    Examples/          # Runner scripts for each framework (run_lightrag.py, etc.)
    Evaluation/        # Shared evaluation pipeline
      generation_eval.py   # Answer quality metrics (ROUGE-L, correctness, coverage, faithfulness)
      retrieval_eval.py    # Context relevance & evidence recall
      indexing_eval.py     # Graph structure metrics (uses python-igraph)
      llm/                 # Ollama client wrapper
      metrics/             # Individual metric implementations

## Integration: output schema

New framework runners must produce JSON files where each item matches:

    {
      "id": str,
      "question": str,
      "source": str,
      "context": List[str],
      "evidence": List[str],
      "question_type": str,
      "generated_answer": str,
      "ground_truth": str
    }

The four question types are: `Fact Retrieval`, `Complex Reasoning`, `Contextual Summarize`, `Creative Generation`.

## Gotchas

- `Evaluation/` has no `__init__.py` at the top level. It works because scripts are invoked with `python -m Evaluation.<module>`, which treats the directory as a package. Do not add one without testing — internal imports assume this `-m` invocation pattern.
- `requirements.txt` includes `Evaluation==0.0.2` which is a **different PyPI package**, not this repo's `Evaluation/` directory. The local `Evaluation/` shadows it at runtime.
- `.gitignore` excludes `Examples/*_workspace`, `Examples/*_results`, and `Examples/workspace_*` / `Examples/results_*` — generated data and model artifacts.

## Communication

If the user poses a problem in Russian, respond in Russian. Keep all code comments in English regardless of language.
