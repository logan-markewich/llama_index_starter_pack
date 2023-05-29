from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

# Text QA templates
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information, directly answer the following question "
    "(if you don't know the answer, use the best of your knowledge): {query_str}\n"
)
TEXT_QA_TEMPLATE = QuestionAnswerPrompt(DEFAULT_TEXT_QA_PROMPT_TMPL)

# Refine templates
DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer "
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Given the new context and using the best of your knowledge, improve the existing answer. "
    "If you can't improve the existing answer, just repeat it again. "
    "Do not include un-needed or un-helpful information that is shown in the new context. "
    "Do not mention that you've read the above context."
)
DEFAULT_REFINE_PROMPT = RefinePrompt(DEFAULT_REFINE_PROMPT_TMPL)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "We have the opportunity to refine the above answer "
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Given the new context and using the best of your knowledge, improve the existing answer. "
        "If you can't improve the existing answer, just repeat it again. "
        "Do not include un-needed or un-helpful information that is shown in the new context. "
        "Do not mention that you've read the above context."
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# refine prompt selector
DEFAULT_REFINE_PROMPT_SEL_LC = ConditionalPromptSelector(
    default_prompt=DEFAULT_REFINE_PROMPT.get_langchain_prompt(),
    conditionals=[(is_chat_model, CHAT_REFINE_PROMPT.get_langchain_prompt())],
)
REFINE_TEMPLATE = RefinePrompt(langchain_prompt_selector=DEFAULT_REFINE_PROMPT_SEL_LC)

DEFAULT_TERM_STR = (
    "Make a list of terms and definitions that are defined in the context, "
    "with one pair on each line. "
    "If a term is missing it's definition, use your best judgment. "
    "Write each line as as follows:\nTerm: <term> Definition: <definition>"
)

DEFAULT_TERMS = {
    "New York City": "The most populous city in the United States, located at the southern tip of New York State, and the largest metropolitan area in the U.S. by both population and urban area.",
    "boroughs": "Five administrative divisions of New York City, each coextensive with a respective county of the state of New York: Brooklyn, Queens, Manhattan, The Bronx, and Staten Island.",
    "metropolitan statistical area": "A geographical region with a relatively high population density at its core and close economic ties throughout the area.",
    "combined statistical area": "A combination of adjacent metropolitan and micropolitan statistical areas in the United States and Puerto Rico that can demonstrate economic or social linkage.",
    "megacities": "A city with a population of over 10 million people.",
    "United Nations": "An intergovernmental organization that aims to maintain international peace and security, develop friendly relations among nations, achieve international cooperation, and be a center for harmonizing the actions of nations.",
    "Pulitzer Prizes": "A series of annual awards for achievements in journalism, literature, and musical composition in the United States.",
    "Times Square": "A major commercial and tourist destination in Manhattan, New York City.",
    "New Netherland": "A Dutch colony in North America that existed from 1614 until 1664.",
    "Dutch West India Company": "A Dutch trading company that operated as a monopoly in New Netherland from 1621 until 1639-1640.",
    "patroon system": "A system instituted by the Dutch to attract settlers to New Netherland, whereby wealthy Dutchmen who brought 50 colonists would be awarded land and local political autonomy.",
    "Peter Stuyvesant": "The last Director-General of New Netherland, who served from 1647 until 1664.",
    "Treaty of Breda": "A treaty signed in 1667 between the Dutch and English that resulted in the Dutch keeping Suriname and the English keeping New Amsterdam (which was renamed New York).",
    "African Burying Ground": "A cemetery discovered in Foley Square in the 1990s that included 10,000 to 20,000 graves of colonial-era Africans, some enslaved and some free.",
    "Stamp Act Congress": "A meeting held in New York in 1765 in response to the Stamp Act, which imposed taxes on printed materials in the American colonies.",
    "Battle of Long Island": "The largest battle of the American Revolutionary War, fought on August 27, 1776, in Brooklyn, New York City.",
    "New York Police Department": "The police force of New York City.",
    "Irish immigrants": "People who immigrated to the United States from Ireland.",
    "lynched": "To kill someone, especially by hanging, without a legal trial.",
    "civil unrest": "A situation in which people in a country are angry and likely to protest or fight.",
    "megacity": "A very large city, typically one with a population of over ten million people.",
    "World Trade Center": "A complex of buildings in Lower Manhattan, New York City, that were destroyed in the September 11 attacks.",
    "COVID-19": "A highly infectious respiratory illness caused by the SARS-CoV-2 virus.",
    "monkeypox outbreak": "An outbreak of a viral disease similar to smallpox, which occurred in the LGBT community in New York City in 2022.",
    "Hudson River": "A river in the northeastern United States, flowing from the Adirondack Mountains in New York into the Atlantic Ocean.",
    "estuary": "A partly enclosed coastal body of brackish water with one or more rivers or streams flowing into it, and with a free connection to the open sea.",
    "East River": "A tidal strait in New York City.",
    "Five Boroughs": "Refers to the five counties that make up New York City: Bronx, Brooklyn, Manhattan, Queens, and Staten Island.",
    "Staten Island": "The most suburban of the five boroughs, located southwest of Manhattan and connected to it by the free Staten Island Ferry.",
    "Todt Hill": "The highest point on the eastern seaboard south of Maine, located on Staten Island.",
    "Manhattan": "The geographically smallest and most densely populated borough of New York City, known for its skyscrapers, Central Park, and cultural, administrative, and financial centers.",
    "Brooklyn": "The most populous borough of New York City, located on the western tip of Long Island and known for its cultural diversity, independent art scene, and distinctive neighborhoods.",
    "Queens": "The largest borough of New York City, located on Long Island north and east of Brooklyn, and known for its ethnic diversity, commercial and residential prominence, and hosting of the annual U.S. Open tennis tournament.",
    "The Bronx": "The northernmost borough of New York",
}
