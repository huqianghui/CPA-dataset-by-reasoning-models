from langchain_core.prompts import (
    HumanMessagePromptTemplate,
)

userPromptTemplateCPA = '''
You will act as a professional tutor for the Certified Public Accountant (CPA) qualification exam. Your task is to answer multiple-choice questions on the subject of accounting—note that these are single-choice questions only!

You must first translate the question accurately into English and analyze it directly in English. Do not provide the translated question in the answer.

Each question has one and only one correct answer. Read the question carefully, fully understand its meaning, and answer strictly based on CPA exam knowledge points. Do not rely on personal interpretation!

If the derived answer is not among the given options, recalculate. The answer must be selected from the four given choices. In this case, only provide the revised reasoning process; do not choose the closest answer through rounding.

Key points to follow:
    •	Ensure that the sign (positive or negative) of the calculated result aligns with the question’s requirements.
    •	Distinguish clearly whether the question is asking for the correct statement(s) or the incorrect statement(s).
    •	If the question mentions “China” or similar terms, analyze according to Chinese accounting standards.

Your answer must follow these structured steps:
    1.	Identify the CPA core knowledge point tested in the question.
    2.	Provide the full solution process. If calculations are involved, include a concise equation setup and each step of the computation.
    3.	Clearly state the correct answer (A, B, C, or D).
    4.	Return the final response in JSON formatin Chinese directly, but don't including the word json in the response. Just Direct give the structured result as follows:
    {{
        "process": "the full solution process",
        "answer": "the correct answer"
    }}
Please start solving questions immediately. 

Question:
${cpa_question}
'''

cpa_prompt = HumanMessagePromptTemplate.from_template(userPromptTemplateCPA)