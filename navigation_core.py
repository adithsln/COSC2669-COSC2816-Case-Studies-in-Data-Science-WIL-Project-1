"""Privacy-aware, cautious navigation and crisis signposting; standard library only.

Not a diagnosis, a validated clinical triage tool, or a substitute for human review.
"""
import re

EMERGENCY_SOURCE = 'https://www.lifeline.org.au/131114'

# These are conservative signpost triggers, NOT a comprehensive safety classifier.
SELF_HARM = re.compile(
    r"\b(suicid(?:e|al)|kill (?:myself|herself|himself|themselves)|"
    r"end (?:my|her|his|their) life|want to die|don't want to live|do not want to live|"
    r"hurt (?:myself|yourself|herself|himself|themselves|themself)|harm (?:myself|yourself|herself|himself|themselves|themself)|"
    r"harming (?:myself|yourself|herself|himself|themselves|themself)|intend(?:s|ed)? to harm (?:myself|yourself|herself|himself|themselves|themself)|"
    r"self[ -]?harm|overdos(?:e|ed|ing)|"
    r"better off dead|end it all|no reason to live|took too many pills|can't go on living|cannot go on living|"
    r"(?:i|they|he|she) (?:want|wants|intend|intends|plan|plans) to die)\b", re.I
)
NEGATED_SELF = re.compile(
    r"\b(?:i am|i'm|im|i feel|i was|i'm feeling)\s+(?:not|never)\s+"
    r"(?:suicidal|thinking of suicide|going to hurt myself|self[ -]?harming)\b|"
    r"\b(?:i|they|he|she)\s+(?:do not|don't|does not|doesn't)\s+(?:want to die|intend to harm (?:myself|themselves|himself|herself)|plan to harm (?:myself|themselves|himself|herself))\b", re.I
)
IMMEDIATE = re.compile(
    r"\b(right now|immediately|tonight|today|about to|going to|will do it|"
    r"have a plan|have taken|just took|already took|in immediate danger)\b", re.I
)

RMIT_UP = re.compile(r'\brmit\s*(?:up|university pathways)\b', re.I)
RMIT = re.compile(r'\brmit\b', re.I)
MONASH = re.compile(r'\bmonash\b', re.I)
MELBOURNE = re.compile(r'\b(?:university of melbourne|melbourne uni(?:versity)?|unimelb)\b', re.I)
ONLINE_GENERAL = re.compile(r'\b(?:online abuse|online harassment|cyber(?:bully\w*|bullied| abuse|stalk\w*)|social media|eSafety|report (?:an )?online)\b', re.I)
OUT_OF_SCOPE = re.compile(r'\b(?:weather forecast|stock price|football score|lotto numbers|write me code)\b', re.I)
SUPPORT_SCOPE = re.compile(
    r'\b(?:wellbeing|well-being|mental health|stress|stressed|overwhelm\w*|anxi\w*|depress\w*|distress\w*|counsell?\w*|psycholog\w*|support|help|welfare|unsafe|harass\w*|bully\w*|threat\w*|stalk\w*|concerning|unwanted behavio(?:u)?r|abuse|crisis|suicid\w*|self[ -]?harm|harm (?:myself|yourself|herself|himself|themselves|themself)|talk to someone|where should i start|appointment|workshops?|webinars?|group programs?|student services?|safer community|housing|accommodation|financial support)\b',
    re.I
)


IMMINENT_DANGER = re.compile(r'\b(?:i (?:am|m) in immediate danger|i (?:cannot|can\x27t) keep myself safe|someone is going to hurt me)\b', re.I)
OVERDOSE = re.compile(r'\b(?:overdos(?:e|ed|ing)|took too many pills)\b', re.I)


class SafetyGuard:
    """Detect explicit high-risk language; not a clinically validated sensitivity measure."""
    def __init__(self, llm=None):  # compatibility with the historical init call
        self.llm = llm

    def classify(self, text):
        screened = NEGATED_SELF.sub(' ', text)
        if IMMINENT_DANGER.search(screened) or OVERDOSE.search(screened):
            return 'immediate'
        if not SELF_HARM.search(screened):
            return 'routine'
        return 'immediate' if IMMEDIATE.search(screened) else 'crisis'

    def is_distressed(self, text):
        return self.classify(text) != 'routine'


def safety_response(level):
    if level not in {'crisis', 'immediate'}:
        raise ValueError('Unknown crisis category')
    start = ('If you or someone else may be in immediate danger, call **000** now. '
             if level == 'immediate' else
             'I’m sorry this is happening. You deserve prompt support from a person. '
             'If there is immediate danger, call **000** now. ')
    return (start + 'You can contact **Lifeline Australia**, 24/7: call **13 11 14**, '
            'text **0477 13 11 14**, or use their official website. '
            'If possible, reach out to a trusted person who can stay with you. '
            'This chatbot cannot provide emergency or clinical care.\n\n'
            f'Official crisis-support source: {EMERGENCY_SOURCE}')


def detect_institution(question):
    found = set()
    if RMIT_UP.search(question):
        found.add('RMIT_UP')
    elif RMIT.search(question):
        found.add('RMIT_UNIVERSITY')
    if MONASH.search(question):
        found.add('MONASH')
    if MELBOURNE.search(question):
        found.add('MELBOURNE')
    if len(found) > 1:
        return 'MULTIPLE'
    return next(iter(found), None)


def allowed_institutions(institution):
    if institution == 'RMIT_UP':
        return ['RMIT_UP', 'RMIT_SHARED', 'GENERAL']
    if institution == 'RMIT_UNIVERSITY':
        return ['RMIT_UNIVERSITY', 'RMIT_SHARED', 'GENERAL']
    if institution in ('MONASH', 'MELBOURNE'):
        return [institution, 'GENERAL']
    return ['GENERAL']


def is_explicit_minor(question):
    if re.search(r'\bunder 18\b', question, re.I):
        return True
    age = re.search(r"\b(?:i am|i'm|i’m|aged?|as a)\s+(\d{1,2})\b|\b(\d{1,2})[- ]year[- ]old\b", question, re.I)
    if age:
        return int(age.group(1) or age.group(2)) < 18
    return False


def route_question(question, safety_guard=None):
    guard = safety_guard or SafetyGuard()
    crisis = guard.classify(question)
    if crisis != 'routine':
        return {'route': 'crisis', 'level': crisis, 'institution': None,
                'message': safety_response(crisis)}
    if re.search(r'\bmonash college\b', question, re.I):
        return {'route': 'clarify', 'institution': None,
                'message': 'This source snapshot covers Monash University, not independently verified Monash College services. Please check your specific Monash College support pathway rather than assume university service eligibility.'}
    inst = detect_institution(question)
    if inst == 'MULTIPLE':
        return {'route': 'clarify', 'institution': None,
                'message': 'Which university are you seeking a service from: RMIT, Monash or the University of Melbourne? I should not assume eligibility across institutions.'}
    if OUT_OF_SCOPE.search(question):
        return {'route': 'out_of_scope', 'institution': inst,
                'message': 'I can only help with documented student wellbeing or support-navigation information. For other topics, please use an appropriate service.'}
    # Do not ask for a university when the question itself is clearly unrelated to support navigation.
    if not SUPPORT_SCOPE.search(question) and not ONLINE_GENERAL.search(question):
        return {'route': 'out_of_scope', 'institution': inst,
                'message': 'I can only help with documented student wellbeing or support-navigation information. For other topics, please use an appropriate service.'}
    if inst is None and not ONLINE_GENERAL.search(question):
        return {'route': 'clarify', 'institution': None,
                'message': 'Which university do you attend: RMIT University, RMIT UP, Monash or the University of Melbourne? This helps me avoid recommending a service you cannot use.'}
    return {'route': 'retrieve', 'institution': inst, 'message': None}


def adult_reporting_context_confirmed(question):
    """Only show the adult-specific eSafety route when adulthood and Australia are explicit.

    This does NOT determine eligibility for a formal reporting scheme; users must check
    the published threshold, prior platform report and other conditions on eSafety's page.
    """
    age = re.search(r"\b(?:i am|i'm|i’m|aged?|as a)\s+(\d{1,2})\b|\b(\d{1,2})[- ]year[- ]old\b", question, re.I)
    adult_age = bool(age and int(age.group(1) or age.group(2)) >= 18)
    is_adult = adult_age or bool(re.search(r'\b(?:i am|i.m) (?:an? )?adult\b|\bover 18\b', question, re.I))
    australia = bool(re.search(r'\b(?:live|living|resid(?:e|ing)|located|based) in australia\b|'
                               r'\bin australia\b|\baustralian resident\b', question, re.I))
    return bool(is_adult and australia and not is_explicit_minor(question))
