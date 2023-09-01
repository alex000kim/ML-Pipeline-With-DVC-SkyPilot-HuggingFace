


def llama_prompt_format(prompt, completion):
    text = f"<s>[INST] {prompt} [/INST] {completion} </s>"
    return text

def 