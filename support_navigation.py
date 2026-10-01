"""Route student questions to safe and relevant support-navigation pathways."""

import re

EMERGENCY_SOURCE = 'https://www.lifeline.org.au/131114'

SELF_HARM = re.compile(
    r"\b(suicid(?:e|al)|kill (?:myself|herself|himself|themselves)|"
    r"end (?:my|her|his|their) life|want to die|don't want to live|do not want to live|"
    r"hurt (?:myself|yourself|herself|himself|themselves|themself)|"
    r"harm (?:myself|yourself|herself|himself|themselves|themself)|"
    r"harming (?:myself|yourself|herself|himself|themselves|themself)|"
    r"intend(?:s|ed)? to harm (?:myself|yourself|herself|himself|themselves|themself)|"
    r"self[ -]?harm|overdos(?:e|ed|ing)|better off dead|end it all|no reason to live|"
    r"(?:took|taken) too many pills|can't go on living|cannot go on living|"
    r"(?:i|they|he|she) (?:want|wants|intend|intends|plan|plans) to die)\b",
    re.I,
)

NEGATED_SELF_HARM = re.compile(
    r"\b(?:i am|i'm|im|i feel|i was|i'm feeling)\s+(?:not|never)\s+"
    r"(?:suicidal|thinking of suicide|going to hurt myself|self[ -]?harming)\b|"
    r"\b(?:i|they|he|she)\s+(?:do not|don't|does not|doesn't)\s+"
    r"(?:want to die|intend to harm (?:myself|themselves|himself|herself)|"
    r"plan to harm (?:myself|themselves|himself|herself))\b",
    re.I,
)

IMMEDIATE_RISK = re.compile(
    r"\b(right now|immediately|tonight|today|about to|going to|will do it|"
    r"have a plan|have taken|just took|already took|in immediate danger)\b",
    re.I,
)

IMMINENT_DANGER = re.compile(
    r"\b(?:i (?:am|m) in immediate danger|i (?:cannot|can\x27t) keep myself safe|"
    r"someone is going to hurt me)\b",
    re.I,
)

OVERDOSE = re.compile(
    r"\b(?:overdos(?:e|ed|ing)|(?:took|taken) too many pills)\b",
    re.I,
)

RMIT_UP = re.compile(r'\brmit\s*(?:up|university pathways)\b', re.I)
RMIT = re.compile(r'\brmit\b', re.I)
MONASH = re.compile(r'\bmonash\b', re.I)
MELBOURNE = re.compile(
    r'\b(?:university of melbourne|melbourne uni(?:versity)?|unimelb)\b',
    re.I,
)

GREETING = re.compile(
    r'^\s*(?:hi|hello|hey|good morning|good afternoon|good evening)'
    r'(?:\s+(?:there|boss|bro|buddy|mate|friend|friemd|'
    r'my\s+(?:friend|friemd|buddy|mate|bro|boss)))?'
    r'[!. ]*$',
    re.I,
)

STUDENT_INTRO = re.compile(
    r"^\s*(?:i am|i'm|im)\s+(?:a|an)\s+(?:university|tertiary)\s+student[.! ]*$",
    re.I,
)

ONLINE_SUPPORT = re.compile(
    r'\b(?:online abuse|online harassment|cyber(?:bully\w*|bullied| abuse|stalk\w*)|'
    r'social media|eSafety|report (?:an )?online)\b',
    re.I,
)

OUT_OF_SCOPE = re.compile(
    r'\b(?:weather forecast|stock price|football score|lotto numbers|write me code)\b',
    re.I,
)

SUPPORT_SCOPE = re.compile(
    r'\b(?:wellbeing|well-being|mental health|stress|stressed|overwhelm\w*|'
    r'anxi\w*|depress\w*|distress\w*|counsell?\w*|psycholog\w*|support|help|'
    r'welfare|unsafe|harass\w*|bully\w*|threat\w*|stalk\w*|concerning|'
    r'unwanted behavio(?:u)?r|abuse|crisis|suicid\w*|self[ -]?harm|'
    r'harm (?:myself|yourself|herself|himself|themselves|themself)|talk to someone|'
    r'where should i start|appointment|workshops?|webinars?|group programs?|'
    r'student services?|safer community|housing|accommodation|financial support)\b',
    re.I,
)

class SafetyGuard:
    """Classify explicit crisis language before normal support retrieval."""

    def __init__(self, llm=None):
        self.llm = llm

    def classify(self, text):
        screened_text = NEGATED_SELF_HARM.sub(' ', text)

        if IMMINENT_DANGER.search(screened_text) or OVERDOSE.search(screened_text):
            return 'immediate'

        if not SELF_HARM.search(screened_text):
            return 'routine'

        if IMMEDIATE_RISK.search(screened_text):
            return 'immediate'

        return 'crisis'

    def is_distressed(self, text):
        return self.classify(text) != 'routine'

def safety_response(level):
    """Return the crisis signposting shown before any retrieval is attempted."""
    if level not in {'crisis', 'immediate'}:
        raise ValueError('Unknown crisis category.')

    if level == 'immediate':
        opening = 'If you or someone else may be in immediate danger, call **000** now. '
    else:
        opening = (
            'I’m sorry this is happening. You deserve prompt support from a person. '
            'If there is immediate danger, call **000** now. '
        )

    return (
        opening
        + 'You can contact **Lifeline Australia**, 24/7: call **13 11 14**, '
        + 'text **0477 13 11 14**, or use their official website. '
        + 'If possible, reach out to a trusted person who can stay with you. '
        + 'This chatbot cannot provide emergency or clinical care.\n\n'
        + f'Official crisis-support source: {EMERGENCY_SOURCE}'
    )

def detect_institution(question):
    """Identify the university named in the student's question."""
    institutions = set()

    if RMIT_UP.search(question):
        institutions.add('RMIT_UP')
    elif RMIT.search(question):
        institutions.add('RMIT_UNIVERSITY')

    if MONASH.search(question):
        institutions.add('MONASH')

    if MELBOURNE.search(question):
        institutions.add('MELBOURNE')

    if len(institutions) > 1:
        return 'MULTIPLE'

    return next(iter(institutions), None)

def canonical_institution_reply(text):
    """Return the canonical institution name for a short institution-only reply."""
    cleaned = text.strip()

    if re.fullmatch(r'rmit\s*(?:up|university pathways)[.!]?', cleaned, re.I):
        return 'RMIT UP'

    if re.fullmatch(r'rmit(?:\s+(?:university|uni))?[.!]?', cleaned, re.I):
        return 'RMIT University'

    if re.fullmatch(r'monash(?:\s+university)?[.!]?', cleaned, re.I):
        return 'Monash University'

    if re.fullmatch(
        r'(?:the\s+)?university\s+of\s+melbourne[.!]?|'
        r'melbourne(?:\s+(?:university|uni))?[.!]?|'
        r'unimelb[.!]?',
        cleaned,
        re.I,
    ):
        return 'University of Melbourne'

    return None

def allowed_institutions(institution):
    """Return the service groups that can be considered for an institution."""
    if institution == 'RMIT_UP':
        return ['RMIT_UP', 'RMIT_SHARED', 'GENERAL']

    if institution == 'RMIT_UNIVERSITY':
        return ['RMIT_UNIVERSITY', 'RMIT_SHARED', 'GENERAL']

    if institution in ('MONASH', 'MELBOURNE'):
        return [institution, 'GENERAL']

    return ['GENERAL']

def is_explicit_minor(question):
    """Return True only when the question explicitly indicates an age below 18."""
    if re.search(r'\bunder 18\b', question, re.I):
        return True

    age_match = re.search(
        r"\b(?:i am|i'm|i’m|aged?|as a)\s+(\d{1,2})\b|"
        r"\b(\d{1,2})[- ]year[- ]old\b",
        question,
        re.I,
    )

    if not age_match:
        return False

    age = int(age_match.group(1) or age_match.group(2))
    return age < 18

def route_question(question, safety_guard=None):
    """Decide whether to retrieve, clarify, decline, or show crisis support."""
    guard = safety_guard or SafetyGuard()
    safety_level = guard.classify(question)

    if safety_level != 'routine':
        return {
            'route': 'crisis',
            'level': safety_level,
            'institution': None,
            'message': safety_response(safety_level),
        }

    if GREETING.search(question):
        return {
            'route': 'clarify',
            'institution': None,
            'message': (
                'Hi! I can help you find verified student wellbeing and support services. '
                'Tell me what kind of support you are looking for and, if relevant, '
                'which university you attend.'
            ),
        }

    if STUDENT_INTRO.search(question):
        return {
            'route': 'clarify',
            'institution': None,
            'message': (
                'What kind of wellbeing or student support are you looking for? '
                'You can describe the situation in your own words.'
            ),
        }

    if re.search(r'\bmonash college\b', question, re.I):
        return {
            'route': 'clarify',
            'institution': None,
            'message': (
                'This source snapshot covers Monash University, not independently '
                'verified Monash College services. Please check your specific Monash '
                'College support pathway rather than assume university service eligibility.'
            ),
        }

    standalone_institution = canonical_institution_reply(question)

    if standalone_institution:
        return {
            'route': 'clarify',
            'institution': detect_institution(standalone_institution),
            'message': (
                'What kind of wellbeing or student support are you looking for? '
                'You can describe the situation in your own words.'
            ),
        }

    institution = detect_institution(question)

    if institution == 'MULTIPLE':
        return {
            'route': 'clarify',
            'institution': None,
            'message': (
                'Which university are you seeking a service from: RMIT, Monash or the '
                'University of Melbourne? I should not assume eligibility across institutions.'
            ),
        }

    if (
        institution is not None
        and not SUPPORT_SCOPE.search(question)
        and not ONLINE_SUPPORT.search(question)
        and not OUT_OF_SCOPE.search(question)
    ):
        return {
            'route': 'clarify',
            'institution': institution,
            'message': (
                'What kind of wellbeing or student support are you looking for? '
                'You can describe the situation in your own words.'
            ),
        }

    if OUT_OF_SCOPE.search(question):
        return {
            'route': 'out_of_scope',
            'institution': institution,
            'message': (
                'I can only help with documented student wellbeing or support-navigation '
                'information. For other topics, please use an appropriate service.'
            ),
        }

    if not SUPPORT_SCOPE.search(question) and not ONLINE_SUPPORT.search(question):
        return {
            'route': 'out_of_scope',
            'institution': institution,
            'message': (
                'I can only help with documented student wellbeing or support-navigation '
                'information. For other topics, please use an appropriate service.'
            ),
        }

    if institution is None and not ONLINE_SUPPORT.search(question):
        return {
            'route': 'clarify',
            'institution': None,
            'message': (
                'Which university do you attend: RMIT University, RMIT UP, Monash or the '
                'University of Melbourne? This helps me avoid recommending a service you '
                'cannot use.'
            ),
        }

    return {
        'route': 'retrieve',
        'institution': institution,
        'message': None,
    }

def adult_reporting_context_confirmed(question):
    """Check whether adult and Australian context are both explicit in the question."""
    age_match = re.search(
        r"\b(?:i am|i'm|i’m|aged?|as a)\s+(\d{1,2})\b|"
        r"\b(\d{1,2})[- ]year[- ]old\b",
        question,
        re.I,
    )

    adult_age = bool(
        age_match
        and int(age_match.group(1) or age_match.group(2)) >= 18
    )

    adult_wording = bool(
        re.search(
            r'\b(?:i am|i.m) (?:an? )?adult\b|\bover 18\b',
            question,
            re.I,
        )
    )

    australia_context = bool(
        re.search(
            r'\b(?:live|living|resid(?:e|ing)|located|based) in australia\b|'
            r'\bin australia\b|\baustralian resident\b',
            question,
            re.I,
        )
    )

    is_adult = adult_age or adult_wording

    return bool(
        is_adult
        and australia_context
        and not is_explicit_minor(question)
    )
