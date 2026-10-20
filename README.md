# WIL Project

## Tertiary Student Wellbeing and Support Navigation Chatbot

### Group 1

| Full Name | Student ID |
| --- | --- |
| Adithya Lakshmi Narasimhan | S4212064 |
| Akhil Chintakindi | S4184580 |
| Himmat Singh Grewal | S4225515 |
| Maleka Ramchandani | S4225051 |
| Shourya Pratap Singh | S4221078 |

## Project overview

This project investigates how effectively a source-grounded RAG chatbot can interpret a student's question in everyday language and guide them to an appropriate, eligible and verifiable support pathway using published support information.

The prototype runs as a browser-based Streamlit application. It is designed for support navigation only. It is not an AI therapist, diagnostic service or emergency service.

The current source catalogue covers RMIT University, RMIT UP, Monash University and the University of Melbourne. It also includes selected general pathways from eSafety and Lifeline where relevant.

## How the chatbot works

The main flow is:

```text
Student question
 ↓
Safety and scope checks
 ↓
MiniLM embedding
 ↓
Chroma retrieval
 ↓
Llama 3.2 selects from the retrieved service IDs
 ↓
Eligibility, contact details and source are read from the CSV catalogue
 ↓
Response shown in Streamlit
```

The language model does not write the published eligibility or contact details itself. After a service is selected, those details are read from `data/official_services.csv`.

## Project structure

```text
Project-1-WIL/
├── app.py
├── rag_pipeline.py
├── support_navigation.py
├── service_catalog.py
├── evaluate_ragas.py
├── requirements.txt
├── README.md
│
├── assets/
│   ├── chatbot_logo_browser.png
│   └── wil_chatbot_logo.png
│
├── data/
│   └── official_services.csv
│
├── evaluation/
│   └── scenarios.csv
│
├── evaluation_results/
│   ├── baseline_development.csv
│   ├── pipeline_development.json
│   ├── ragas_development.csv
│   └── ragas_environment.txt
│
├── tests/
│   └── test_chatbot.py
│
├── Walert Reproduction/
│
└── archive/
```

### `app.py`

Runs the Streamlit interface and manages the chat session.

### `rag_pipeline.py`

Uses MiniLM and ChromaDB to find relevant services, then uses Llama 3.2 to choose from the retrieved service IDs and return the support information linked to that service.

### `support_navigation.py`

Checks the student's institution, whether the question is in scope, whether more information is needed, and whether the message needs urgent safety handling. Emergency messages bypass the normal retrieval process.

### `service_catalog.py`

Reads and validates the service records in `data/official_services.csv`.

### `data/official_services.csv`

Stores the service information used by the chatbot, including purpose, eligibility, contact details and the official source URL.

### `evaluation/scenarios.csv`

This file has the questions we used for testing. It contains 28 questions: 20 development questions and 8 reserved for the final test.

### `evaluate_ragas.py`

This script runs the development checks and writes the outputs to the results folder. It reports the embedding baseline, routing and retrieval results, and the RAGAS scores.

### `evaluation_results/`

Stores the saved development outputs, including the embedding baseline, pipeline results, RAGAS results and the package environment used for the RAGAS run.

### `tests/test_chatbot.py`

Includes 25 automated tests covering routing, safety behaviour, eligibility rules, catalogue checks, grounded service information and vector-store initialisation.

## Requirements

The chatbot runs locally with Python and Ollama. The main chatbot packages are installed from `requirements.txt`, while the RAGAS evaluation uses the separate environment saved in `evaluation_results/ragas_environment.txt`.

For normal chatbot use, install the project dependencies from the project folder:

```bash
pip install -r requirements.txt
```

The RAGAS evaluation was run in a separate frozen package environment. That environment is recorded in:

```text
evaluation_results/ragas_environment.txt
```

It is only needed if the RAGAS evaluation is being reproduced.

Make sure Ollama is installed. The main chatbot uses:

```bash
ollama pull llama3.2
```

The supplementary RAGAS evaluation uses:

```bash
ollama pull llama3.1:8b
```

## Run the chatbot

From the project folder:

```bash
python -m streamlit run app.py
```

On the first run, the chatbot creates a local Chroma vector store from `data/official_services.csv`.

The generated vector store is stored locally and is not committed to GitHub.

## Run the automated tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The current test suite has 25 automated tests covering routing, safety behaviour, service eligibility, catalogue integrity, source-grounded responses and vector-store initialisation.

## Evaluation

We evaluated the project using 28 fictional student questions. We used 20 during development and kept 8 as new final-test questions.

Out of the 28 questions, 20 required the chatbot to recommend a support service. It chose one of the accepted services in 19 of those 20 cases.

In the remaining recommendation case, the chatbot selected a different RMIT service from the accepted pathway. The contact information shown for the selected service still matched the corresponding official source.

Across all 20 recommendation cases, the contact details shown for the selected service matched the official source.

Of the 8 new final-test questions, 5 required a support recommendation and 4 of those 5 were correct.

The evaluation also exposed a safety weakness. In one emergency scenario, the chatbot asked for more information instead of directing the student to urgent help.

After the evaluation, we fixed that specific emergency-routing issue. When we test the same scenario on the current prototype, it is now sent directly to urgent support. This confirms the fix for that case only, so we still need wider emergency and safety testing before the chatbot can be used with students.

The test set is small and uses fictional scenarios. The results therefore show how the prototype performed on these questions, rather than proving clinical effectiveness or performance across the wider student population.

### Development reproducibility checks

We ran the development set again on the current version of the project and recorded these results:

- raw embedding baseline Hit@3: 100%
- raw embedding baseline Top-1 accuracy: 93.33%
- route accuracy: 100%
- retrieval Hit@3: 100%
- retrieval Recall@3: 90%
- retrieval MRR: 0.967

This was a separate check of the current code. We did not use it to replace the results from the original 28-question evaluation.

## Supplementary RAGAS evaluation

We also ran RAGAS on the 15 development cases where the chatbot returned a support-service answer. This gave us another way to inspect how well the answers stayed grounded in the retrieved service information and how useful the retrieved context was.

For this run, we used:

- `llama3.1:8b` as the local judge model
- `all-MiniLM-L6-v2` as the embedding model
- RAGAS `0.4.3`
- 15 answered development referral cases

The saved results are in:

```text
evaluation_results/ragas_development.csv
```

The package environment used for that run is in:

```text
evaluation_results/ragas_environment.txt
```

The recorded mean scores were:

| Metric | Mean |
| --- | ---: |
| Faithfulness | 0.890 |
| Answer Relevancy | 0.587 |
| Context Precision | 0.972 |
| Context Recall | 0.796 |

These RAGAS scores are supplementary. They describe specific aspects of grounding and retrieval and are not measures of clinical validity or overall chatbot accuracy.

## Walert reproduction

The `Walert Reproduction/` directory contains the supplied Walert quantitative evaluation materials and the outputs from the earlier reproduction work.

Those files are kept separate from the final chatbot implementation.

## Privacy and safety

The Streamlit prototype is intended for fictional testing scenarios, not real private student information.

Disk-based chat saving is off by default. It can only be enabled locally by setting:

```text
WIL_SAVE_CHATS=1
```

If someone is in immediate danger in Australia, call `000`.

Lifeline provides 24/7 crisis support on `13 11 14`.
