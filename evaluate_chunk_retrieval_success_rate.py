import pandas as pd
import pytrec_eval
from collections import defaultdict
df = pd.read_csv("draft_ground_truth.csv")
qrel = {}

for index, row in df.iterrows():
    
    query_id = row['query_id']
    doc_id = row['expected_doc_id']

    if doc_id != "NO_DOC_FOUND":
        qrel[query_id] = {doc_id: 1}
    
with open('my_chroma.run','r') as f_run:
    run = pytrec_eval.parse_run(f_run)

evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall_1', 'success_1'})
results = evaluator.evaluate(run)

ranked_docs_by_query = defaultdict(list)
with open('my_chroma.run','r') as f_run:
    for line in f_run:
        parts = line.split()
        if len(parts) < 5:
            continue
        q_id, _, doc_id, rank, score = parts[0], parts[1], parts[2], parts[3], parts[4]
        ranked_docs_by_query[q_id].append((int(rank), doc_id))
for q_id in ranked_docs_by_query:
    ranked_docs_by_query[q_id].sort()

#print("--- Week 7 Retrival Evaluation ---")
#for query_id, query_measure in results.items():
#    print(f"Query: {query_id}")
#    print(f"    Success@1 (Did it find the exact chunk?): {query_measure['success_1']}")

#average_success = sum(m['success_1'] for m in results.values()) / len(results)
#print(f"\nOverall System Effectivness (Success Rate): {average_success * 100:.2f}%")

print("--- Week 7 Retrival Evaluation ---")
success_1_hits = 0
success_3_hits = 0
total = 0

for query_id, expected_docs in qrel.items():
    expected_doc_id = list(expected_docs.keys())[0]
    total += 1

    success_1 = results.get(query_id, {}).get('success_1', 0.0)
    top_docs = [d for _, d in ranked_docs_by_query.get(query_id, [])][:3]
    success_3 = 1.0 if expected_doc_id in top_docs else 0.0

    success_1_hits += success_1
    success_3_hits += success_3

    print(f"Query: {query_id}")
    print(f"    Success@1 (Did it find the exact chunk at rank 1?): {success_1}")
    print(f"    Success@3 (Did it find the exact chunk at rank 3?): {success_3}")

    if success_1 == 0.0:
        rank_of_expected = next(
            (r for r, d in ranked_docs_by_query.get(query_id, []) if d == expected_doc_id),
            None
        )
        top1 = top_docs[0] if top_docs else "N/A"
        if rank_of_expected:
            print(f"    -> Expected doc was retreived at rank {rank_of_expected} instead (top1 was: {top1})")
        else:
            print(f"    -> Expected doc not retreived in top 3 at all (top1 was: {top1})")
print(f"\nOverall Success@1 Rate: {success_1_hits / total * 100:.2f}%")
print(f"Overall Success@3 Rate: {success_3_hits / total * 100:.2f}%")
