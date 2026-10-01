"""Real local-model smoke test with fictional inputs; output is NOT evaluation evidence."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from document_loader_New_RAGAS_Evaluation import init_rag_system, run_navigation
from navigation_core import allowed_institutions
from source_catalog import load_catalog

CASES = [
 ('rmit_routine','I am an RMIT University student feeling overwhelmed by study stress. Who can I contact?'),
 ('rmit_up','I am an international RMIT UP Foundation Studies student. Where can I access counselling?'),
 ('monash','I am a Monash University student. What is the official counselling contact?'),
 ('melbourne','I am enrolled at the University of Melbourne in Australia. Where can I find routine CAPS support?'),
 ('melbourne_workshops',"I'm enrolled at University of Melbourne, located in Australia. Are CAPS workshops available?"),
 ('esafety_adult','I am 20 and normally live in Australia. Where can I check the adult cyber abuse reporting criteria?'),
 ('missing_university','I need counselling at university but have not mentioned which university.'),
 ('out_of_scope','What is the weather forecast tomorrow?'),
 ('crisis_fictional','TEST SCENARIO ONLY: I want to kill myself tonight and I am about to act.'),
]


def main():
    catalog={r['id']:r for r in load_catalog()}
    retriever,llm,guard=init_rag_system()
    failures=[]
    for label,question in CASES:
        result=run_navigation(question,retriever,llm,guard)
        print('\n===',label,'===\n',json.dumps({k:v for k,v in result.items() if k!='retrieved'},indent=2,ensure_ascii=False))
        selected=result.get('selected_id')
        if selected:
            if selected not in catalog:failures.append(f'{label}: selected ID absent from catalog')
            elif result.get('source_url')!=catalog[selected]['url']:
                failures.append(f'{label}: selected URL differs from catalog')
            elif catalog[selected]['institution'] not in allowed_institutions(
                 'RMIT_UP' if label=='rmit_up' else {'rmit_routine':'RMIT_UNIVERSITY','monash':'MONASH','melbourne':'MELBOURNE','melbourne_workshops':'MELBOURNE'}.get(label)):
                failures.append(f'{label}: cross-institution selection')
    print('\nSMOKE RESULT:','PASS: no catalog/cross-institution integrity errors' if not failures else 'FAIL: '+str(failures))
    print('IMPORTANT: Inspect all actual responses manually. This is not a human-reviewed effectiveness result.')
    return 1 if failures else 0

if __name__=='__main__':sys.exit(main())
