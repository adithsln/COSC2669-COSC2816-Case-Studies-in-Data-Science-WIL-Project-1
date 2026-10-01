import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from navigation_core import SafetyGuard, detect_institution, route_question, allowed_institutions, adult_reporting_context_confirmed
from source_catalog import load_catalog, snapshot_hash
import document_loader as pipe


class FakeRetriever:
    def __init__(self, records, ids):
        self.records={r['id']:r for r in records}; self.ids=ids
    def retrieve(self, question, top_k=3, institution=None, score_threshold=0.15):
        return [{'id':key,'metadata':{'source_url':self.records[key]['url']},
                 'content':self.records[key]['summary'],'similarity_score':0.91}
                for key in self.ids if self.records[key]['institution'] in allowed_institutions(institution)][:top_k]

class FakeLLM:
    def __init__(self,id): self.id=id
    def invoke(self,messages):return types.SimpleNamespace(content=json.dumps({'service_id':self.id}))

class Msg:
    def __init__(self,content):self.content=content

message_mod=types.ModuleType('langchain_core.messages')
message_mod.SystemMessage=Msg;message_mod.HumanMessage=Msg
core_mod=types.ModuleType('langchain_core'); core_mod.messages=message_mod


class SafetyTests(unittest.TestCase):
    def test_crisis_and_case_insensitive(self):
        g=SafetyGuard()
        self.assertEqual(g.classify('I WANT TO KILL MYSELF TONIGHT'),'immediate')
        self.assertEqual(g.classify('I feel suicidal'),'crisis')
        self.assertEqual(g.classify('I took too many pills just now'),'immediate')
        self.assertEqual(g.classify('I might harm myself'),'crisis')
        self.assertEqual(g.classify('This is a fictional test. The student says they intend to harm themselves immediately.'),'immediate')
    def test_negation_and_routine(self):
        g=SafetyGuard()
        for s in ('I am not suicidal, just overwhelmed with study',
                  'I do not want to die; I want help with study stress',
                  "I'm not suicidal but would like RMIT counselling"):
            self.assertEqual(g.classify(s),'routine',s)
        self.assertEqual(g.classify('I am not suicidal but I want to hurt myself'),'crisis')
    def test_crisis_bypasses_model_and_retrieval(self):
        g=SafetyGuard()
        result=pipe.run_navigation('I want to kill myself tonight',None,None,g)
        self.assertEqual(result['route'],'crisis')
        self.assertIn('000',result['answer']);self.assertIn('13 11 14',result['answer'])
    def test_routing(self):
        self.assertEqual(detect_institution('I am RMIT UP student'),'RMIT_UP')
        self.assertEqual(detect_institution('I study RMIT University'),'RMIT_UNIVERSITY')
        self.assertEqual(detect_institution('I study at Monash'),'MONASH')
        self.assertEqual(detect_institution('University of Melbourne workshops'),'MELBOURNE')
        self.assertEqual(detect_institution('RMIT or Monash?'),'MULTIPLE')
        self.assertEqual(route_question('What is the weather forecast?')['route'],'out_of_scope')
        self.assertEqual(route_question('What is the best laptop for gaming?')['route'],'out_of_scope')
        self.assertEqual(route_question('I need counselling but where?')['route'],'clarify')
        self.assertEqual(route_question('I am not suicidal but struggling at RMIT')['route'],'retrieve')

    def test_melbourne_workshop_query_is_in_scope(self):
        self.assertEqual(route_question("I'm enrolled at University of Melbourne, located in Australia. Are CAPS workshops available?")['route'],'retrieve')

    def test_adult_cyber_abuse_criteria_query_is_in_scope(self):
        self.assertEqual(route_question('I am 20 and normally live in Australia. Where can I check the adult cyber abuse reporting criteria?')['route'],'retrieve')

    def test_monash_threatening_behaviour_query_is_in_scope(self):
        self.assertEqual(route_question('I study at Monash and someone is stalking me and sending threatening messages.')['route'],'retrieve')

class CatalogTests(unittest.TestCase):
    def setUp(self):self.records=load_catalog()
    def test_records_have_valid_sources(self):
        self.assertGreaterEqual(len(self.records),12)
        self.assertEqual(len(snapshot_hash(self.records)),64)
        self.assertEqual(len({r['id'] for r in self.records}),len(self.records))
    def test_only_rmit_up_and_shared(self):
        self.assertNotIn('RMIT_COUNSELLING',[
             r['id'] for r in self.records if r['institution'] in allowed_institutions('RMIT_UP')])
        self.assertIn('RMIT_WELFARE',[
             r['id'] for r in self.records if r['institution'] in allowed_institutions('RMIT_UP')])

class ResponseTests(unittest.TestCase):
    def setUp(self):self.records=load_catalog()
    def test_contact_from_catalog_not_model(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation('I am at Monash and need routine counselling.',
               FakeRetriever(self.records,['MONASH_COUNSELLING','RMIT_COUNSELLING']),
               FakeLLM('MONASH_COUNSELLING'),SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'answered')
        self.assertEqual(out['selected_id'],'MONASH_COUNSELLING')
        self.assertIn('(03) 9905 3020',out['answer'])
        self.assertNotIn('RMIT Counselling',out['answer'])
        self.assertIn('https://www.monash.edu/',out['answer'])
    def test_ineligible_selection_fail_closed(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation('I am at RMIT University. I need counselling.',
                  FakeRetriever(self.records,['RMIT_COUNSELLING']),FakeLLM('MONASH_COUNSELLING'),
                  SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'abstain')
    def test_rmit_up_counselling_needs_program(self):
        out=pipe.run_navigation('I am an RMIT UP student. Is counselling free?',None,None,SafetyGuard())
        self.assertEqual(out['route'],'clarify');self.assertIn('Fast Track',out['answer'])
    def test_rmit_up_fast_track_is_recognised_as_published_program(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            result=pipe.run_navigation('I am an international RMIT UP Fast Track student and want counselling.',
                                       FakeRetriever(self.records,['RMIT_UP_COUNSELLING']),
                                       FakeLLM('RMIT_UP_COUNSELLING'),SafetyGuard(),catalog=self.records)
        self.assertEqual(result['route'],'answered')
        self.assertEqual(result['selected_id'],'RMIT_UP_COUNSELLING')

    def test_rmit_up_counselling_is_not_offered_when_program_or_international_status_is_unverified(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            for question in (
                'I am a domestic RMIT UP Foundation student and I feel overwhelmed. Where do I start?',
                'I am an RMIT UP student who feels overwhelmed. Where should I start?',
            ):
                result=pipe.run_navigation(question,FakeRetriever(self.records,['RMIT_UP_COUNSELLING']),
                                           FakeLLM('RMIT_UP_COUNSELLING'),SafetyGuard(),catalog=self.records)
                self.assertNotEqual(result['selected_id'],'RMIT_UP_COUNSELLING')
    def test_adult_specific_esafety_not_shown_without_known_adult_context(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            for question in ('I am a student in Australia experiencing online harassment.',
                             'I am 16 and in Australia, experiencing online abuse.'):
                result=pipe.run_navigation(question,
                    FakeRetriever(self.records,['ESAFETY_ADULT','ESAFETY_TERTIARY']),
                    FakeLLM('ESAFETY_ADULT'),SafetyGuard(),catalog=self.records)
                self.assertNotEqual(result['selected_id'],'ESAFETY_ADULT')

    def test_routine_melbourne_counselling_excludes_crisis_service(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation(
                'I study at the University of Melbourne and would like counselling support. Where should I start?',
                FakeRetriever(self.records,['MELB_CRISIS','MELB_CAPS']),
                FakeLLM('MELB_CAPS'),SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'answered')
        self.assertEqual(out['selected_id'],'MELB_CAPS')
        self.assertNotIn('Crisis Support Service',out['answer'])


    def test_melbourne_workshop_prefers_workshop_record(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation(
                "I'm enrolled at University of Melbourne, located in Australia. Are CAPS workshops available?",
                FakeRetriever(self.records,['MELB_CAPS','MELB_WORKSHOPS','MELB_CRISIS']),
                FakeLLM('MELB_WORKSHOPS'),SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'answered')
        self.assertEqual(out['selected_id'],'MELB_WORKSHOPS')
        self.assertIn('workshops-and-webinars',out['source_url'])

    def test_monash_threatening_behaviour_prefers_safer_community(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation(
                'At Monash, someone keeps stalking me and sending threatening messages. Which university service handles safety concerns like this?',
                FakeRetriever(self.records,['MONASH_COUNSELLING','MONASH_SAFER']),
                FakeLLM('MONASH_SAFER'),SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'answered')
        self.assertEqual(out['selected_id'],'MONASH_SAFER')
        self.assertIn('safer-community',out['source_url'])

    def test_explicit_adult_cyber_abuse_criteria_prefers_adult_record(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation(
                'I am 20 and normally live in Australia. Where can I check the adult cyber abuse reporting criteria?',
                FakeRetriever(self.records,['ESAFETY_TERTIARY','ESAFETY_ADULT']),
                FakeLLM('ESAFETY_ADULT'),SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'answered')
        self.assertEqual(out['selected_id'],'ESAFETY_ADULT')
        self.assertIn('adult-cyber-abuse',out['source_url'])

    def test_no_cross_university(self):
        with patch.dict(sys.modules,{'langchain_core':core_mod,'langchain_core.messages':message_mod}):
            out=pipe.run_navigation('I study at RMIT University. I need mental health support.',
                FakeRetriever(self.records,['MONASH_COUNSELLING']),FakeLLM('MONASH_COUNSELLING'),
                SafetyGuard(),catalog=self.records)
        self.assertEqual(out['route'],'abstain')

if __name__=='__main__':unittest.main()

class IndexInitializationTests(unittest.TestCase):
    def test_clean_index_is_populated_once_and_count_called(self):
        import numpy as np
        dotenv=types.ModuleType('dotenv');dotenv.load_dotenv=lambda:None
        ollama=types.ModuleType('langchain_ollama');ollama.ChatOllama=lambda **kwargs: FakeLLM('NONE')
        class Collection:
            name='test_collection'
            def __init__(self):self.n=0
            def count(self):return self.n
        class Store:
            instance=None
            def __init__(self,*args,**kwargs):
                self.collection=Collection();self.calls=0;Store.instance=self
            def add_records(self,records,embeddings,snapshot):
                self.calls+=1;self.collection.n=len(records)
        class Embed:
            def generate_embeddings(self,texts):return np.zeros((len(texts),3))
        with patch.dict(sys.modules,{'dotenv':dotenv,'langchain_ollama':ollama}),\
             patch.object(pipe,'VectorStore',Store),patch.object(pipe,'EmbeddingManager',Embed):
            retriever, llm, guard=pipe.init_rag_system()
        self.assertEqual(Store.instance.calls,1)
        self.assertEqual(Store.instance.collection.count(),len(load_catalog()))
        self.assertIsInstance(guard,SafetyGuard)
