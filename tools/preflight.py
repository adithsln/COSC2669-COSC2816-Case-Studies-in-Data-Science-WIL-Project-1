"""Non-destructive local setup checks. No real student text and no automatic web scraping."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from source_catalog import load_catalog, snapshot_hash

REQUIRED = ('streamlit', 'chromadb', 'sentence_transformers', 'langchain_core', 'langchain_ollama', 'dotenv', 'numpy', 'pandas')


def inspect(require_live=False):
    problems = []
    print('=== WIL source-navigation preflight ===')
    print('Python:', sys.version.split()[0], '| project:', ROOT)
    if sys.version_info[:2] != (3, 11):
        print('NOTE: Python 3.11 is recommended. Python versions other than 3.11 need independent dependency testing.')
    for name in REQUIRED:
        available = importlib.util.find_spec(name) is not None
        print(f'{name}: {"FOUND" if available else "MISSING"}')
        if require_live and not available:
            problems.append(f'Missing Python package: {name}')
    logo = ROOT / 'assets' / 'wil_chatbot_logo.png'
    print('Logo:', 'FOUND' if logo.is_file() else 'MISSING')
    if not logo.is_file():
        problems.append('Missing bundled logo asset')
    try:
        records = load_catalog()
        status = {s:sum(r['approval_status']==s for r in records) for s in sorted({r['approval_status'] for r in records})}
        print('Service records:', len(records), '| SHA256:', snapshot_hash(records))
        print('Approval status:', status)
        if any(r['approval_status']!='APPROVED' for r in records):
            print('NOTICE: Source catalog is still for group review. Do not label results FINAL.')
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        problems.append(f'Invalid catalog: {exc}')
    if require_live:
        try:
            with urlopen('http://127.0.0.1:11434/api/tags', timeout=4) as response:
                data = json.load(response)
            models = [m.get('name', '') for m in data.get('models', [])]
            print('Local Ollama models:', ', '.join(models) if models else '(none)')
            if not any(x=='llama3.2' or x.startswith('llama3.2:') for x in models):
                problems.append('Missing local llama3.2 model. Run: ollama pull llama3.2')
        except (OSError, URLError, ValueError) as exc:
            problems.append(f'Ollama cannot be reached on localhost:11434: {exc}')
    if problems:
        for problem in problems:
            print('FAIL:', problem)
        return 1
    print('PASS: Preflight checks completed.' + (' Live dependencies and Ollama found.' if require_live else ' Offline check only.'))
    return 0


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--live',action='store_true',help='Require runtime Python packages and local Ollama llama3.2')
    args=parser.parse_args()
    sys.exit(inspect(args.live))
