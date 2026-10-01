import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rag_pipeline as pipe
from service_catalog import load_catalog, snapshot_hash
from support_navigation import (
    SafetyGuard,
    allowed_institutions,
    detect_institution,
    route_question,
)

class FakeRetriever:
    def __init__(self, records, service_ids):
        self.records = {record['id']: record for record in records}
        self.service_ids = service_ids

    def retrieve(self, question, top_k=3, institution=None, score_threshold=0.15):
        results = []
        for service_id in self.service_ids:
            record = self.records[service_id]
            if record['institution'] in allowed_institutions(institution):
                results.append(
                    {
                        'id': service_id,
                        'metadata': {'source_url': record['url']},
                        'content': record['summary'],
                        'similarity_score': 0.91,
                    }
                )
        return results[:top_k]

class FakeLLM:
    def __init__(self, service_id):
        self.service_id = service_id

    def invoke(self, messages):
        return types.SimpleNamespace(
            content=json.dumps({'service_id': self.service_id})
        )

class Message:
    def __init__(self, content):
        self.content = content

MESSAGE_MODULE = types.ModuleType('langchain_core.messages')
MESSAGE_MODULE.SystemMessage = Message
MESSAGE_MODULE.HumanMessage = Message

LANGCHAIN_CORE_MODULE = types.ModuleType('langchain_core')
LANGCHAIN_CORE_MODULE.messages = MESSAGE_MODULE

class SafetyTests(unittest.TestCase):
    def test_crisis_and_case_insensitive(self):
        guard = SafetyGuard()

        self.assertEqual(
            guard.classify('I WANT TO KILL MYSELF TONIGHT'),
            'immediate',
        )
        self.assertEqual(guard.classify('I feel suicidal'), 'crisis')
        self.assertEqual(
            guard.classify('I took too many pills just now'),
            'immediate',
        )
        self.assertEqual(guard.classify('I might harm myself'), 'crisis')
        self.assertEqual(
            guard.classify(
                'This is a fictional test. The student says they intend to '
                'harm themselves immediately.'
            ),
            'immediate',
        )

    def test_negation_and_routine(self):
        guard = SafetyGuard()

        routine_questions = (
            'I am not suicidal, just overwhelmed with study',
            'I do not want to die; I want help with study stress',
            "I'm not suicidal but would like RMIT counselling",
        )

        for question in routine_questions:
            self.assertEqual(guard.classify(question), 'routine', question)

        self.assertEqual(
            guard.classify('I am not suicidal but I want to hurt myself'),
            'crisis',
        )

    def test_crisis_bypasses_model_and_retrieval(self):
        result = pipe.run_navigation(
            'I want to kill myself tonight',
            None,
            None,
            SafetyGuard(),
        )

        self.assertEqual(result['route'], 'crisis')
        self.assertIn('000', result['answer'])
        self.assertIn('13 11 14', result['answer'])

    def test_routing(self):
        self.assertEqual(
            detect_institution('I am RMIT UP student'),
            'RMIT_UP',
        )
        self.assertEqual(
            detect_institution('I study RMIT University'),
            'RMIT_UNIVERSITY',
        )
        self.assertEqual(
            detect_institution('I study at Monash'),
            'MONASH',
        )
        self.assertEqual(
            detect_institution('University of Melbourne workshops'),
            'MELBOURNE',
        )
        self.assertEqual(
            detect_institution('RMIT or Monash?'),
            'MULTIPLE',
        )
        self.assertEqual(
            route_question('What is the weather forecast?')['route'],
            'out_of_scope',
        )
        self.assertEqual(
            route_question('What is the best laptop for gaming?')['route'],
            'out_of_scope',
        )
        self.assertEqual(
            route_question('I need counselling but where?')['route'],
            'clarify',
        )
        self.assertEqual(
            route_question('I am not suicidal but struggling at RMIT')['route'],
            'retrieve',
        )

    def test_melbourne_workshop_query_is_in_scope(self):
        result = route_question(
            "I'm enrolled at University of Melbourne, located in Australia. "
            'Are CAPS workshops available?'
        )
        self.assertEqual(result['route'], 'retrieve')

    def test_adult_cyber_abuse_criteria_query_is_in_scope(self):
        result = route_question(
            'I am 20 and normally live in Australia. Where can I check the '
            'adult cyber abuse reporting criteria?'
        )
        self.assertEqual(result['route'], 'retrieve')

    def test_monash_threatening_behaviour_query_is_in_scope(self):
        result = route_question(
            'I study at Monash and someone is stalking me and sending '
            'threatening messages.'
        )
        self.assertEqual(result['route'], 'retrieve')

class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.records = load_catalog()

    def test_records_have_valid_sources(self):
        self.assertGreaterEqual(len(self.records), 12)
        self.assertEqual(len(snapshot_hash(self.records)), 64)
        self.assertEqual(
            len({record['id'] for record in self.records}),
            len(self.records),
        )

    def test_only_rmit_up_and_shared(self):
        rmit_up_ids = [
            record['id']
            for record in self.records
            if record['institution'] in allowed_institutions('RMIT_UP')
        ]

        self.assertNotIn('RMIT_COUNSELLING', rmit_up_ids)
        self.assertIn('RMIT_WELFARE', rmit_up_ids)

class ResponseTests(unittest.TestCase):
    def setUp(self):
        self.records = load_catalog()

    def test_contact_from_catalog_not_model(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am at Monash and need routine counselling.',
                FakeRetriever(
                    self.records,
                    ['MONASH_COUNSELLING', 'RMIT_COUNSELLING'],
                ),
                FakeLLM('MONASH_COUNSELLING'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'MONASH_COUNSELLING')
        self.assertIn('(03) 9905 3020', result['answer'])
        self.assertNotIn('RMIT Counselling', result['answer'])
        self.assertIn('https://www.monash.edu/', result['answer'])

    def test_ineligible_selection_fail_closed(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am at RMIT University. I need counselling.',
                FakeRetriever(self.records, ['RMIT_COUNSELLING']),
                FakeLLM('MONASH_COUNSELLING'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'abstain')

    def test_rmit_study_stress_prefers_counselling(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am an RMIT University student feeling overwhelmed by study stress. '
                'Who can I talk to?',
                FakeRetriever(
                    self.records,
                    ['RMIT_CONNECT', 'RMIT_WELFARE', 'RMIT_COUNSELLING'],
                ),
                FakeLLM('RMIT_COUNSELLING'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'RMIT_COUNSELLING')

    def test_rmit_financial_hardship_prefers_welfare(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I study at RMIT University. Financial hardship is making university '
                'difficult. Who can help me find assistance?',
                FakeRetriever(
                    self.records,
                    ['RMIT_CONNECT', 'RMIT_COUNSELLING', 'RMIT_WELFARE'],
                ),
                FakeLLM('RMIT_WELFARE'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'RMIT_WELFARE')

    def test_rmit_unstable_housing_prefers_welfare(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am enrolled at RMIT and my housing situation has become unstable. '
                'I am not asking for counselling, but I need help working out university '
                'and external support options. Where should I start?',
                FakeRetriever(
                    self.records,
                    ['RMIT_CONNECT', 'RMIT_COUNSELLING', 'RMIT_WELFARE'],
                ),
                FakeLLM('RMIT_WELFARE'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'RMIT_WELFARE')

    def test_rmit_rent_centrelink_and_emergency_accommodation_prefers_welfare(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am an RMIT student struggling with rent and I need help understanding '
                'Centrelink support and emergency accommodation options. Which RMIT '
                'service should I start with?',
                FakeRetriever(
                    self.records,
                    ['RMIT_CONNECT', 'RMIT_COUNSELLING', 'RMIT_WELFARE'],
                ),
                FakeLLM('RMIT_WELFARE'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'RMIT_WELFARE')

    def test_rmit_up_counselling_needs_program(self):
        result = pipe.run_navigation(
            'I am an RMIT UP student. Is counselling free?',
            None,
            None,
            SafetyGuard(),
        )

        self.assertEqual(result['route'], 'clarify')
        self.assertIn('Fast Track', result['answer'])

    def test_rmit_up_fast_track_is_recognised_as_published_program(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am an international RMIT UP Fast Track student and want counselling.',
                FakeRetriever(self.records, ['RMIT_UP_COUNSELLING']),
                FakeLLM('RMIT_UP_COUNSELLING'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'RMIT_UP_COUNSELLING')

    def test_rmit_up_counselling_is_not_offered_when_program_or_international_status_is_unverified(self):
        questions = (
            'I am a domestic RMIT UP Foundation student and I feel overwhelmed. '
            'Where do I start?',
            'I am an RMIT UP student who feels overwhelmed. Where should I start?',
        )

        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            for question in questions:
                result = pipe.run_navigation(
                    question,
                    FakeRetriever(self.records, ['RMIT_UP_COUNSELLING']),
                    FakeLLM('RMIT_UP_COUNSELLING'),
                    SafetyGuard(),
                    catalog=self.records,
                )
                self.assertNotEqual(
                    result['selected_id'],
                    'RMIT_UP_COUNSELLING',
                )

    def test_adult_specific_esafety_not_shown_without_known_adult_context(self):
        questions = (
            'I am a student in Australia experiencing online harassment.',
            'I am 16 and in Australia, experiencing online abuse.',
        )

        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            for question in questions:
                result = pipe.run_navigation(
                    question,
                    FakeRetriever(
                        self.records,
                        ['ESAFETY_ADULT', 'ESAFETY_TERTIARY'],
                    ),
                    FakeLLM('ESAFETY_ADULT'),
                    SafetyGuard(),
                    catalog=self.records,
                )
                self.assertNotEqual(result['selected_id'], 'ESAFETY_ADULT')

    def test_routine_melbourne_counselling_excludes_crisis_service(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I study at the University of Melbourne and would like '
                'counselling support. Where should I start?',
                FakeRetriever(self.records, ['MELB_CRISIS', 'MELB_CAPS']),
                FakeLLM('MELB_CAPS'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'MELB_CAPS')
        self.assertNotIn('Crisis Support Service', result['answer'])

    def test_melbourne_workshop_prefers_workshop_record(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                "I'm enrolled at University of Melbourne, located in Australia. "
                'Are CAPS workshops available?',
                FakeRetriever(
                    self.records,
                    ['MELB_CAPS', 'MELB_WORKSHOPS', 'MELB_CRISIS'],
                ),
                FakeLLM('MELB_WORKSHOPS'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'MELB_WORKSHOPS')
        self.assertIn('workshops-and-webinars', result['source_url'])

    def test_monash_threatening_behaviour_prefers_safer_community(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'At Monash, someone keeps stalking me and sending threatening '
                'messages. Which university service handles safety concerns like this?',
                FakeRetriever(
                    self.records,
                    ['MONASH_COUNSELLING', 'MONASH_SAFER'],
                ),
                FakeLLM('MONASH_SAFER'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'MONASH_SAFER')
        self.assertIn('safer-community', result['source_url'])

    def test_explicit_adult_cyber_abuse_criteria_prefers_adult_record(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I am 20 and normally live in Australia. Where can I check the '
                'adult cyber abuse reporting criteria?',
                FakeRetriever(
                    self.records,
                    ['ESAFETY_TERTIARY', 'ESAFETY_ADULT'],
                ),
                FakeLLM('ESAFETY_ADULT'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'answered')
        self.assertEqual(result['selected_id'], 'ESAFETY_ADULT')
        self.assertIn('adult-cyber-abuse', result['source_url'])

    def test_no_cross_university(self):
        with patch.dict(
            sys.modules,
            {
                'langchain_core': LANGCHAIN_CORE_MODULE,
                'langchain_core.messages': MESSAGE_MODULE,
            },
        ):
            result = pipe.run_navigation(
                'I study at RMIT University. I need mental health support.',
                FakeRetriever(self.records, ['MONASH_COUNSELLING']),
                FakeLLM('MONASH_COUNSELLING'),
                SafetyGuard(),
                catalog=self.records,
            )

        self.assertEqual(result['route'], 'abstain')

class IndexInitializationTests(unittest.TestCase):
    def test_clean_index_is_populated_once_and_count_called(self):
        import numpy as np

        dotenv_module = types.ModuleType('dotenv')
        dotenv_module.load_dotenv = lambda: None

        ollama_module = types.ModuleType('langchain_ollama')
        ollama_module.ChatOllama = lambda **kwargs: FakeLLM('NONE')

        class Collection:
            name = 'test_collection'

            def __init__(self):
                self.n = 0

            def count(self):
                return self.n

        class Store:
            instance = None

            def __init__(self, *args, **kwargs):
                self.collection = Collection()
                self.calls = 0
                Store.instance = self

            def add_records(self, records, embeddings, snapshot):
                self.calls += 1
                self.collection.n = len(records)

        class Embed:
            def generate_embeddings(self, texts):
                return np.zeros((len(texts), 3))

        with (
            patch.dict(
                sys.modules,
                {
                    'dotenv': dotenv_module,
                    'langchain_ollama': ollama_module,
                },
            ),
            patch.object(pipe, 'VectorStore', Store),
            patch.object(pipe, 'EmbeddingManager', Embed),
        ):
            retriever, llm, guard = pipe.init_rag_system()

        self.assertEqual(Store.instance.calls, 1)
        self.assertEqual(
            Store.instance.collection.count(),
            len(load_catalog()),
        )
        self.assertIsInstance(guard, SafetyGuard)

if __name__ == '__main__':
    unittest.main()
