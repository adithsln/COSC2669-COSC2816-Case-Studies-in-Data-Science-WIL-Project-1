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

The prototype is a browser-based Streamlit application. It is designed for support navigation only. It is not an AI therapist, diagnostic service or emergency service.

The current university coverage is limited to RMIT University, RMIT UP, Monash University and the University of Melbourne. The source catalogue also includes selected general pathways from eSafety and Lifeline where relevant.

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
Eligibility, contact details and official source are taken from the CSV catalogue
    ↓
Response shown in Streamlit
```

The LLM does not create the published contact details or eligibility information. After a service ID is selected, those fields are read from `data/official_services.csv`.

## Main project files

```text
app.py
rag_pipeline.py
support_navigation.py
service_catalog.py
requirements.txt
README.md

data/
    official_services.csv

assets/
    wil_chatbot_logo.png

evaluation/
    scenarios.csv

tests/
    test_chatbot.py
```

### `app.py`
Runs the Streamlit user interface.

### `rag_pipeline.py`
Initialises MiniLM, ChromaDB and Llama 3.2, retrieves relevant service records and returns the final support-navigation response.

### `support_navigation.py`
Handles institution detection, scope checks, clarification rules and emergency signposting before normal retrieval.

### `service_catalog.py`
Loads and validates the official support-service catalogue stored in `data/official_services.csv`.

### `data/official_services.csv`
Contains the support-service records used by the chatbot, including service purpose, eligibility, contact information and official source URL.

### `evaluation/scenarios.csv`
Contains the 28 fictional questions used in the project evaluation: 20 development questions and 8 new final-test questions.

### `tests/test_chatbot.py`
Contains automated tests for routing, safety behaviour, catalogue integrity, eligibility restrictions, source-grounded contact information and index initialisation.

## Requirements

The chatbot uses Python and Ollama locally.

Install Ollama and make sure the `llama3.2` model is available.

From the project folder, install the Python packages with:

```bash
pip install -r requirements.txt
```

Pull the local model if required:

```bash
ollama pull llama3.2
```

## Run the chatbot

From the project folder:

```bash
streamlit run app.py
```

The first run creates a local Chroma vector store from `data/official_services.csv`. The vector store is generated locally and is not committed to GitHub.

## Run the automated tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Evaluation

The final project evaluation used 28 fictional questions:

- 20 questions used during development
- 8 new questions used for final testing

Twenty questions required the chatbot to recommend a support service. The chatbot selected an acceptable support service in 19 of those 20 cases.

In the remaining case, it selected a different RMIT service from the expected pathway, but the contact information shown for that selected service still matched the official source. Across all 20 service recommendations, the displayed contact details matched the corresponding official source.

Of the 8 new final-test questions, 5 required a support recommendation and 4 of those 5 were handled correctly.

The evaluation also exposed an important safety limitation. In one emergency test, the chatbot asked for more information instead of directing the student to urgent help. The prototype therefore requires stronger emergency handling and further safety testing before student deployment.

These results come from a small fictional test set and should not be interpreted as clinical validation or population-level performance.

## Walert reproduction

The `Walert Reproduction/` directory contains the supplied Walert quantitative evaluation materials used for baseline reproduction work. Those files are kept separate from the final chatbot implementation.

## Privacy and safety

The Streamlit prototype is intended for fictional testing scenarios.

Disk-based chat saving is off by default. It can only be enabled locally by setting:

```text
WIL_SAVE_CHATS=1
```

If someone is in immediate danger in Australia, call 000. Lifeline provides 24/7 crisis support on 13 11 14.
