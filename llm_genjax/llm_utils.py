import re
import os
from pathlib import Path

import anthropic
import dotenv


dotenv.load_dotenv(Path(__file__).parent / "../.env")

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def get_completion(messages: list[dict], model_name, system_prompt="", temperature=1.0):
    breakpoint()
    # user and assistant messages MUST alternate, and messages MUST start with a user turn.
    message = client.messages.create(
        model=model_name,
        max_tokens=8192,
        temperature=temperature,
        system=system_prompt,
        messages=messages
    )
    return message.content[0].text



def extract_code_blocks(text):
    tags = re.findall(r'<(\w+)\s*(?:name="([^"]*)")?\s*>(.*?)</\1>', text, re.DOTALL)
    code_blocks = {}
    for tag, name, code in tags:
        if tag == "code":
            code_blocks[name] = code
    return code_blocks


def query(messages, prompt, model_id, save_dir, temperature=1.0, system_prompt=""):
    save_dir.mkdir(parents=True, exist_ok=True)
    # save prompt
    (save_dir / "prompt.txt").write_text(prompt)

    messages += [
        {"role": "user", "content": prompt}
    ]
    if not (save_dir / "answer.txt").exists():
        answer = get_completion(messages, model_id, temperature=temperature, system_prompt=system_prompt)
        # save answer
        (save_dir / "answer.txt").write_text(answer)
    else:
        answer = (save_dir / "answer.txt").read_text()

    messages += [
        {"role": "assistant", "content": answer},
    ]

    if "<stop></stop>" in answer:
        return None

    # extract python code block
    code_blocks = extract_code_blocks(answer)
    functions = {}
    for name, code in code_blocks.items():
        # save the code block
        (save_dir / f"code_{name.replace('/', '_')}.py").write_text(code)

        # exec the code and store it
        local_namespace = {}
        exec(code, globals(), local_namespace)

        assert len(local_namespace) == 1, "Only one function is allowed"

        function = list(local_namespace.values())[0]
        functions[name] = function

    return code_blocks, functions
