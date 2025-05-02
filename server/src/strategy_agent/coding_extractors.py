from typing import Optional


def extract_code_from_markdown_code_blocks(text: str) -> Optional[str]:
    """
    Extract code from markdown code blocks in the provided text.

    Supports both triple backtick code blocks with or without language specifiers.

    Args:
        text: The text containing markdown code blocks

    Returns:
        A string containing only the code extracted from code blocks, with blocks
        separated by newlines
    """
    import re

    # Pattern to match code blocks with or without language specifier
    # (?s) enables dot to match newlines
    # (?:```(?:\w+)?\n(.*?)```) matches code blocks with optional language specifier
    pattern = r"(?m)^(?<!`)\`\`\`(\w*)\n([\s\S]*?)^(?<!`)\`\`\`$"

    # Find all code blocks
    matches = re.finditer(pattern, text, re.MULTILINE)

    # Filter out bash/shell blocks and collect valid code blocks
    excluded_langs = {
        "bash",
        "sh",
        "shell",
        "zsh",
        "fish",
        "console",
        "terminal",
        "json",
    }
    code_blocks = []
    for match in matches:
        lang = match.group(1).strip()
        if lang not in excluded_langs:
            code_blocks.append(match.group(2))

    if not code_blocks:
        return None  # Return None if no code blocks found

    # Join all code blocks with newlines
    return "\n".join(code_blocks)
