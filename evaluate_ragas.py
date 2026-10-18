import argparse
import json
from pathlib import Path

import pandas as pd
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy
)
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import EvaluationDataset, evaluate
from support_navigation import detect_institution

from rag_pipeline import(
    EMBEDDING_MODEL,
    init_rag_system,
    run_navigation
)

from service_catalog import as_context, load_catalog

BASE_DIR = Path(__file__).resolve().parent
SCENARIOS_CSV = BASE_DIR/ 'evaluation' /'scenarios.csv'
OUT_DIR = BASE_DIR/ 'eval_results'
TOP_K = 3

JUDGE_MODEL = 'llama3.1:8b'

OUTCOME_TO_ROUTES = {
    'referral': {'answered'},
    'clarify': {'clarify'},
    'out_of_scope': {'out_of_scope', 'abstain', 'refuse'},
    'crisis': {'crisis','escalate','safety'}
}

def split_ids(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [part.strip() for part in str(value).split('|') if part.strip()]


def run_pipeline(df, retriever, llm, safety_guard, catalog):

    rows = []
    for case in df.itertuples():
        result = run_navigation(
            case.question,
            retriever,
            llm,
            safety_guard = safety_guard,
            top_k = TOP_K,
            catalog = catalog
        )

        accepted = split_ids(case.acceptable_service_ids)
        retrieved_ids = [item['id'] for item in result['retrieved']]

        route_ok = result['route'] in OUTCOME_TO_ROUTES[case.expected_outcome]

        selected_ok = (
            result['selected_id'] in accepted if accepted else result['selected_id'] is None
        )

        hit = recall = mrr = None

        if accepted:
            hit = float(any(i in accepted for i in retrieved_ids))
            recall = len(set(retrieved_ids) & set(accepted)) / len(set(accepted))
            mrr = next(
                (1/(rank+1) for rank, i in enumerate(retrieved_ids) if i in accepted),
                0.0
            )

        rows.append(
            {
                'case_id': case.case_id,
                'split': case.split,
                'category': case.category,
                'question': case.question,
                'expected_outcome': case.expected_outcome,
                'accepted_ids': accepted,
                'route': result['route'],
                'route_correct': route_ok,
                'selected_id': result['selected_id'],
                'selected_correct': selected_ok,
                'retrieved_ids': retrieved_ids,
                'retrieved_contexts': [item['content'] for item in result['retrieved']],
                'answer': result['answer'],
                'hit_at_k': hit,
                'recall_at_k': recall,
                'mrr': mrr
            }
        )
    return pd.DataFrame(rows)



def build_ragas_dataset(results, catalog_by_id):

    eligible = results[
        (results['expected_outcome'] == 'referral')
        &(results['route'] == 'answered')
        &(results['retrieved_contexts'].map(len) > 0)
    ]

    samples, case_ids = [],[]
    for row in eligible.itertuples():
        accepted_records = [catalog_by_id[i] for i in row.accepted_ids if i in catalog_by_id]
        if not accepted_records:
            continue
        samples.append(
            SingleTurnSample(
                user_input=row.question,
                retrieved_contexts=row.retrieved_contexts,
                response=row.answer,
                reference='\n\n'.join(as_context(r) for r in accepted_records)
            )
        )
        case_ids.append(row.case_id)
    return EvaluationDataset(samples=samples), case_ids


def run_ragas(dataset):

    judge = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    return evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall()
        ],
        llm=judge,
        embeddings=embeddings,
        run_config=RunConfig(timeout=300, max_workers=2),
        raise_exceptions=False
    )

def raw_retrieval_baseline(df, retriever, TOP_K=TOP_K):
    rows =[]
    for case in df.itertuples():
        accepted = split_ids(case.acceptable_service_ids)
        if not accepted:
            continue
        hits = retriever.retrieve(
            case.question, top_k=TOP_K, institution=detect_institution(case.question)
        )
        ids = [h['id'] for h in hits]
        rows.append({
            'case_id': case.case_id,
            'hit_at_k': float(any(i in accepted for i in ids)),
            'top1_correct': float(bool(ids) and ids[0] in accepted)
        })
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='development',
    choices=['development','final_test','all'])
    parser.add_argument('--skip-ragas', action='store_true',
    help='Only run the deterministic metrics.')
    args = parser.parse_args()

    df = pd.read_csv(SCENARIOS_CSV, encoding='utf-8-sig')

    if args.split != 'all':
        df = df[df['split'] == args.split]

    retriever, llm, safety_guard = init_rag_system()
    catalog = load_catalog()
    catalog_by_id = {r['id']: r for r in catalog}

    results = run_pipeline(df, retriever, llm, safety_guard, catalog)
    OUT_DIR.mkdir(exist_ok=True)

    baseline = raw_retrieval_baseline(df, retriever)
    print("\n=== Baseline: raw embedding retrival ===")
    print(f"Question scored: {len(baseline)}")
    print(f"Baseline hit@{TOP_K}: {baseline['hit_at_k'].mean():.2%}")
    print(f"Baseline top-1 accuracy: {baseline['top1_correct'].mean():.2%}")
    baseline.to_csv(OUT_DIR/ f'basline_{args.split}.csv', index=False)


    print("\n=== Deterministic metric ===")
    print(f"Route accuracy: {results['route_correct'].mean():.2%}")
    print(f"Selected-ID accuracy: {results['selected_correct'].mean():.2%}")
    referral = results[results['expected_outcome'] == 'referral']
    print(f"Retrieval hit@{TOP_K}: {referral['hit_at_k'].mean():.2%}")
    print(f"Retrieval recall@{TOP_K}: {referral['recall_at_k'].mean():.2%}")
    print(f"Retrieval MRR: {referral['mrr'].mean():.3f}")
    print('\nBy category:')
    print(results.groupby('category')[['route_correct','selected_correct']].mean().round(2))

    failures = results[~(results['route_correct'] & results['selected_correct'])]

    if len(failures):
        print('\nFailures:')
        print(failures[['case_id','expected_outcome','route',
        'accepted_ids','selected_id'
        ]].to_string(index=False))


    if not args.skip_ragas:
        dataset, case_ids = build_ragas_dataset(results, catalog_by_id)
        if not case_ids:
            print('\nNo answer referral cases, skipping RAGAS')
        else:
            print(f'\nRunning RAGAS on {len(case_ids)} answered referral cases...')
            ragas_df = run_ragas(dataset).to_pandas()
            ragas_df.insert(0,'case_id',case_ids)
            ragas_df.to_csv(OUT_DIR/f'ragas_{args.split}.csv', index=False)
            print('\n=== RAGAS (mean) ===')
            print(ragas_df.select_dtypes('number').mean().round(3).to_string())

    results.drop(columns=['retrieved_contexts']).to_json(
        OUT_DIR / f'pipeline_{args.split}.json', orient='records', indent=2
    )
    print(f'\nSaved to {OUT_DIR}')



if __name__ == '__main__':
    main()