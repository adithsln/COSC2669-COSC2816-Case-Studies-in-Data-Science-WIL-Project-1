"""Produce raw, reproducible RAG runs. Never guesses referral correctness or claims RAGAS."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_catalog import load_catalog, snapshot_hash
from navigation_core import detect_institution, allowed_institutions
from document_loader_New_RAGAS_Evaluation import init_rag_system, run_navigation, LLM_NAME, MODEL_NAME


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ALLOWED_OUTCOMES={'referral','clarify','crisis','out_of_scope'}


def validate_scenarios(cases, catalog, final):
    if not cases: raise ValueError('The selected scenario set is empty.')
    ids=[c.get('case_id','').strip() for c in cases]
    if not all(ids) or len(set(ids))!=len(ids):
        raise ValueError('Scenario IDs must all be nonempty and unique.')
    service_ids={r['id'] for r in catalog}
    snap=snapshot_hash(catalog)
    for c in cases:
        cid=c['case_id']
        if c.get('expected_outcome') not in ALLOWED_OUTCOMES:
            raise ValueError(f'{cid}: unknown expected outcome')
        if c.get('split') not in ('development','holdout'):
            raise ValueError(f'{cid}: split must be development or holdout')
        if not c.get('question','').strip():raise ValueError(f'{cid}: empty question')
        acceptable=set(filter(None,c.get('acceptable_service_ids_DRAFT','').split('|')))
        if not acceptable <= service_ids:raise ValueError(f'{cid}: reference IDs absent from source catalog')
        if c['expected_outcome']=='referral' and not acceptable:
            raise ValueError(f'{cid}: referral case lacks reviewed acceptable service IDs')
        if final:
            if c.get('group_review_status')!='APPROVED':
                raise ValueError(f'{cid}: final evaluation requires APPROVED review status')
            a=c.get('reviewer_a','').strip();b=c.get('reviewer_b','').strip()
            if not a or not b or a.casefold()==b.casefold():
                raise ValueError(f'{cid}: two distinct group reviewers are required')
            if c.get('source_snapshot_sha256')!=snap:
                raise ValueError(f'{cid}: scenario snapshot differs from approved catalog; regenerate and reapprove case sheet')
    return snap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', default=str(ROOT/'evaluation'/'scenarios_APPROVED.csv'),
                        help='Group-approved frozen scenarios, not the draft.')
    parser.add_argument('--preview-development', action='store_true',
                        help='Use draft development cases only. These are NOT final holdout evaluation.')
    args = parser.parse_args()
    scenario_path = Path(args.scenarios)
    if args.preview_development:
        scenario_path = ROOT/'evaluation'/'scenarios_DRAFT.csv'
    if not scenario_path.exists():
        parser.error('First have the group approve and save evaluation/scenarios_APPROVED.csv')
    cases = list(csv.DictReader(scenario_path.open(encoding='utf-8', newline='')))
    if args.preview_development:
        cases = [c for c in cases if c['split']=='development']
    catalog = load_catalog()
    if not args.preview_development and any(r['approval_status'] != 'APPROVED' for r in catalog):
        parser.error('Final evaluation requires TWO-reviewer approved official_services.json source records.')
    try:
        validate_scenarios(cases, catalog, final=not args.preview_development)
    except ValueError as exc:
        parser.error(str(exc))
    retriever, llm, guard = init_rag_system()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out_dir=ROOT/'evaluation'/'runs'/ (('DEVELOPMENT_PREVIEW_' if args.preview_development else 'FINAL_')+stamp)
    out_dir.mkdir(parents=True,exist_ok=False)
    meta={'run_utc':datetime.now(timezone.utc).isoformat(),
          'status':'development_preview_not_final' if args.preview_development else 'unreviewed_final_raw_output',
          'source_snapshot_sha256':snapshot_hash(catalog),
          'scenario_file_sha256':file_sha(scenario_path),
          'scenario_id_list':[c['case_id'] for c in cases],
          'pipeline_file_sha256':file_sha(ROOT/'document_loader_New_RAGAS_Evaluation.py'),
          'source_catalog_file_sha256':file_sha(ROOT/'source_catalog.py'),
          'evaluation_runner_sha256':file_sha(Path(__file__)),
          'navigation_core_sha256':file_sha(ROOT/'navigation_core.py'),
          'embedding_model':MODEL_NAME,'chat_model':LLM_NAME,
          'service_count':len(catalog), 'case_count':len(cases),
          'important':'Raw system outputs only. No validated referral accuracy or website baseline is asserted.'}
    (out_dir/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    fields=['case_id','source_snapshot_sha256','split','question','expected_outcome','acceptable_service_ids_DRAFT',
            'route','selected_id','source_url','answer','retrieved_service_ids','retrieved_source_urls','retrieved_contexts_json']
    with (out_dir/'raw_outputs.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
        for c in cases:
            result=run_navigation(c['question'],retriever,llm,guard,catalog=catalog)
            docs=result['retrieved']
            writer.writerow({**{x:c.get(x,'') for x in fields},
                'source_snapshot_sha256':meta['source_snapshot_sha256'],
                'route':result['route'],'selected_id':result['selected_id'] or '',
                'source_url':result['source_url'] or '','answer':result['answer'],
                'retrieved_service_ids':'|'.join(d['id'] for d in docs),
                'retrieved_source_urls':'|'.join(d['metadata']['source_url'] for d in docs),
                'retrieved_contexts_json':json.dumps(docs,ensure_ascii=False)})
            print(f"{c['case_id']}: {result['route']} ({result['selected_id']})")
    print('Raw files:',out_dir)
    print('No accuracy statistics have been generated. Review actual output independently.')

if __name__=='__main__':
    main()
