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

This project investigates whether a source-grounded RAG chatbot can understand a student's question in everyday language and guide them towards an appropriate support pathway using published support information.

The prototype runs as a browser-based Streamlit application. Its purpose is support navigation only. It is not an AI therapist, diagnostic service or emergency service.

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

The language model does not generate the published eligibility or contact details. Once a service is selected, those fields are taken from `data/official_services.csv`.

## Project structure

```text
Project-1-WIL/
├── app.py
├── rag_pipeline.py
├── support_navigation.py
├── service_catalog.py
├── requirements.txt
├── README.md
│
├── assets/
│   └── wil_chatbot_logo.png
│
├── data/
│   └── official_services.csv
│
├── evaluation/
│   └── scenarios.csv
│
├── tests/
│   └── test_chatbot.py
│
└── Walert Reproduction/
```

### `app.py`

Runs the Streamlit interface and manages the chat session.

### `rag_pipeline.py`

Initialises MiniLM, ChromaDB and Llama 3.2, retrieves relevant service records and produces the final support-navigation response.

### `support_navigation.py`

Handles institution detection, scope checks, clarification rules and emergency signposting before normal retrieval.

### `service_catalog.py`

Loads and validates the service catalogue stored in `data/official_services.csv`.

### `data/official_services.csv`

Contains the support-service records used by the chatbot, including service purpose, eligibility, contact information and official source URL.

### `evaluation/scenarios.csv`

Contains the 28 fictional evaluation questions: 20 development questions and 8 new final-test questions.

### `tests/test_chatbot.py`

Contains automated tests for routing, safety behaviour, catalogue integrity, eligibility restrictions, source-grounded contact information and vector-store initialisation.

## Requirements

The chatbot runs locally using Python and Ollama.

Install the Python dependencies from the project folder:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the local model is available:

```bash
ollama pull llama3.2
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

The current test suite contains 22 automated tests covering routing, safety behaviour, service eligibility, catalogue integrity and response grounding.

## Evaluation

The final project evaluation used 28 fictional questions:

- 20 development questions
- 8 new final-test questions

Twenty questions required a support-service recommendation. The chatbot selected an acceptable service in 19 of those 20 cases.

In the remaining case, it selected a different RMIT service from the expected pathway. The contact information shown for the service it selected still matched the corresponding official source.

Across all 20 recommendation cases, the contact details for the selected service matched the official source.

Of the 8 new final-test questions, 5 required a support recommendation and 4 of those 5 were correct.

The frozen final evaluation also identified an important safety weakness. In one emergency scenario, the chatbot asked for more information instead of directing the student to urgent help.

That specific routing problem was corrected after the evaluation. The current prototype now sends that same scenario directly to urgent support. Broader emergency and safety testing is still needed before student deployment.

These results come from a small fictional test set. They should not be interpreted as clinical validation or population-level performance.

## Walert reproduction

The `Walert Reproduction/` directory contains the supplied Walert quantitative evaluation materials used for baseline reproduction work.

Those files are kept separate from the final chatbot implementation.

## Privacy and safety

The Streamlit prototype is intended for fictional testing scenarios rather than real private student information.

Disk-based chat saving is disabled by default. It can only be enabled locally by setting:

```text
WIL_SAVE_CHATS=1
```

If someone is in immediate danger in Australia, call `000`.

Lifeline provides 24/7 crisis support on `13 11 14`.
