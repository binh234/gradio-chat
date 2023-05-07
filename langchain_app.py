import gradio as gr
import requests
import csv
import os
from langchain import ConversationChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

prompt_templates = {
    "Default ChatGPT": "",
    "Helpful Asistant": """
    Assistant is a large language model trained by OpenAI.

    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    """
}

# TODO: Add system prompt input when langchain support multiple inputs for ConversationalChain
chat_template = """
Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

{history}
Human: {input}
Assistant:"""

chat_prompt = PromptTemplate(
    input_variables=["system_prompt", "history", "input"], 
    template=chat_template
)

memory = ConversationBufferWindowMemory(k=2)

def get_empty_state():
    return {"total_tokens": 0, "messages": []}

def download_prompt_templates():
    url = "https://raw.githubusercontent.com/f/awesome-chatgpt-prompts/main/prompts.csv"
    try:
        response = requests.get(url)
        reader = csv.reader(response.text.splitlines())
        next(reader)  # skip the header row
        for row in reader:
            if len(row) >= 2:
                act = row[0].strip('"')
                prompt = row[1].strip('"')
                prompt_templates[act] = prompt

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while downloading prompt templates: {e}")
        return

    choices = list(prompt_templates.keys())
    choices = choices[:1] + sorted(choices[1:])
    return gr.update(value=choices[0], choices=choices)

def on_token_change(user_token):
    os.environ["OPENAI_API_KEY"] = user_token

def on_prompt_template_change(prompt_template):
    if not isinstance(prompt_template, str): return
    return prompt_templates[prompt_template]

def submit_message(chat_history, prompt, prompt_template, temperature, max_tokens, context_length):
    memory.k = context_length
    chatgpt_chain = ConversationChain(
        llm=ChatOpenAI(temperature=temperature, max_tokens=max_tokens), 
        prompt=chat_prompt, 
        verbose=False, 
        memory=memory,
    )

    if not prompt:
        return gr.update(value=''), chat_history, f""
    
    system_prompt = prompt_templates[prompt_template]

    if not os.environ["OPENAI_API_KEY"]:
        return '', chat_history.append((prompt, "Error: OpenAI API Key is not set.")), f"Total tokens used: 0"
    
    try:
        response = chatgpt_chain.predict(
            system_prompt=system_prompt,
            input=prompt
        )

        chat_history.append((prompt, response))

    
    except Exception as e:
        chat_history.append((prompt, f"Error: {e}"))

    total_tokens_used_msg = f""

    return '', chat_history, total_tokens_used_msg

def clear_conversation():
    memory.clear()
    return gr.update(value=None, visible=True), None, ""


css = """
      #col-container {max-width: 90%; margin-left: auto; margin-right: auto;}
      #chatbox {min-height: 400px;}
      #header {text-align: center;}
      #prompt_template_preview {padding: 1rem; border-width: 1px; border-style: solid; border-color: #e0e0e0; border-radius: 4px;}
      #total_tokens_str {text-align: right; font-size: 0.8rem; color: #666;}
      #label {font-size: 0.8rem; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2rem; }
      """

with gr.Blocks(css=css) as demo:
    
    state = gr.State(get_empty_state())


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## OpenAI ChatGPT Demo
                    Using the ofiicial API (gpt-3.5-turbo model)
                    Prompt templates from [awesome-chatgpt-prompts](https://github.com/f/awesome-chatgpt-prompts).""",
                    elem_id="header")

        with gr.Row():
            with gr.Column(scale=0.3):
                gr.Markdown("Enter your OpenAI API Key. You can get one [here](https://platform.openai.com/account/api-keys).", elem_id="label")
                user_token = gr.Textbox(value='', placeholder="OpenAI API Key", type="password", show_label=False)
                prompt_template = gr.Dropdown(label="Set a custom insruction for the chatbot:", choices=list(prompt_templates.keys()))
                prompt_template_preview = gr.Markdown(elem_id="prompt_template_preview", value=prompt_templates["Default ChatGPT"])
                with gr.Accordion("Advanced parameters", open=False):
                    temperature = gr.Slider(minimum=0, maximum=2.0, value=0.7, step=0.1, label="Temperature", info="Higher = more creative/chaotic")
                    max_tokens = gr.Slider(minimum=100, maximum=4096, value=1000, step=1, label="Max tokens per response")
                    context_length = gr.Slider(minimum=1, maximum=10, value=2, step=1, label="Context length", info="Number of previous messages to send to the chatbot. Be careful with high values, it can blow up the token budget quickly.")
            with gr.Column(scale=0.7):
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")
                btn_clear_conversation = gr.Button("🔃 Start New Conversation")

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/dragonSwing/chatgpt-grad?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br>
            <p><img src="https://visitor-badge.glitch.me/badge?page_id=dragonswing.chatgpt_api_grad_hf" alt="visitors"></p></center>''')

    btn_submit.click(submit_message, [chatbot, input_message, prompt_template, temperature, max_tokens, context_length], [input_message, chatbot, total_tokens_str])
    input_message.submit(submit_message, [chatbot, input_message, prompt_template, temperature, max_tokens, context_length], [input_message, chatbot, total_tokens_str])
    btn_clear_conversation.click(clear_conversation, [], [input_message, chatbot, total_tokens_str])
    prompt_template.change(on_prompt_template_change, inputs=[prompt_template], outputs=[prompt_template_preview])
    user_token.change(on_token_change, inputs=[user_token], outputs=[])
    
    demo.load(download_prompt_templates, inputs=None, outputs=[prompt_template], queue=False)


demo.queue(concurrency_count=10)
demo.launch(height='800px')