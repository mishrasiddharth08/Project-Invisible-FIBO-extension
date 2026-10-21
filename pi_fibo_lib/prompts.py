"""Faithful structured captions. No truncation, hidden embellishment or API calls."""
import json
import re

def prepare(text, *, edit=False):
    text = str(text).strip()
    if not text:
        if edit:
            raise ValueError("Enter an edit instruction")
        return ""
    if re.search(r"<(?:lora|lyco|hypernet):", text, re.I):
        raise ValueError("Adapter tags require a verified FIBO adapter loader; remove this tag for the official baseline")
    if text.startswith(("{", "[", "```")):
        try:
            obj = json.loads(text)
        except ValueError as error:
            raise ValueError(f"The structured prompt is invalid JSON: {error}") from error
        if not isinstance(obj, dict):
            raise ValueError("The structured prompt must be a JSON object")
        if edit and not obj.get("edit_instruction"):
            raise ValueError("An editing JSON prompt needs edit_instruction")
        return text
    obj = {"short_description": text}
    if edit:
        obj["edit_instruction"] = text
    return json.dumps(obj, ensure_ascii=False)

def check_budget(tokenizer, text, limit=3000):
    ids = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
    count = len(ids)
    if count > limit:
        raise ValueError(f"Prompt has {count} tokens; FIBO limit is {limit}. Shorten by {count - limit} tokens. Nothing was truncated.")
    return count
