import streamlit as st
from groq import Groq
from decouple import config
import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)


GROQ_MODEL = 'mixtral-8x7b-32768'
TIMEOUT = 120
groq_client = Groq(
    api_key=config('GROQ_API_KEY'),
)

st.set_page_config(
    page_title='Auto Text Classifier',
    page_icon='🦾',
    initial_sidebar_state='collapsed'
)

st.title('Auto Text Classifier 🦾')
st.caption('Automatically classify text into a class based on the description + examples you provide.')

with st.form('inputs'):
    class_definition = st.text_area(
        'Define the Class you want to classify',
        value='Detect whether a piece of text is spam or not',
        key=1
    )


    with st.expander('Provide Examples'):
        st.markdown('#### Positive ✅ Examples')
        positive_examples = []
        for i in range(5):
            positive_example = st.text_area(f'Positive Example {i+1}', value='', key=f'{i}-pos')
            if positive_example:
                positive_examples.append(positive_example)
        positive_exs_string = ''.join([f'- {example}\n' for example in positive_examples])
        st.markdown('#### Negative ❌ Examples')
        negative_examples = []
        for i in range(5):
            negative_example = st.text_area(f'Negative Example {i+1}', value='', key=f'{i}-neg')
            if negative_example:
                negative_examples.append(negative_example)
        negative_exs_string = ''.join([f'- {example}\n' for example in negative_examples])

    new_text = st.text_area('Enter Text to Classify', value='buy this tv', key=2)

    submit_button = st.form_submit_button('Classify Text')

if submit_button:
    logger.info('Building Classifier')
    context_prompt = f"""
    You're a classifier that can classify the following text into the class: {class_definition}.

    Return the class and the probability of the text belonging to the class based on your confidence.
    """
    if len(positive_examples) > 0:
        context_prompt += f"""
        Here are some positive examples of the class:
        {positive_exs_string}
        """
    if len(negative_examples) > 0:
        context_prompt += f"""
        Here are some negative examples of the class:
        {negative_exs_string}
        """
    logger.info(f'Context Prompt: {context_prompt}')

    new_text_prompt = f"""
        Classify the following text into the class: {class_definition}.
        
        Here's the text: {new_text}
    """

    with st.expander('Prompts'):
        st.write('Context Prompt')
        st.write(context_prompt)
        st.write('User Prompt')
        st.write(new_text_prompt)

    logger.info(f'New Text Prompt: {new_text_prompt}')
    if new_text:
        with st.spinner('Classifying Text'):
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": context_prompt},
                    {"role": "user", "content": new_text_prompt}
                    
                ],
                temperature=0.0,
                timeout=TIMEOUT
            )
            logger.info(f'Response: {response}')
            st.write(response.choices[0].message.content)

    
