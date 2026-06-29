"""
clean_sft.py — Clean up SFT training data for KlausGPT.

Fixes:
  1. Windows-1252 / UTF-8 encoding corruption (ΓÇÖ → ', etc.)
  2. Chain-of-thought preamble leakage from Gemma
  3. Checklist/validation text leakage
  4. Malformed turn structure
  5. Separator junk (ΓöÇ repeats)
  6. Bold markers around tags

Usage:
  python clean_sft.py sft_conversations.txt sft_clean.txt
"""

import re
import sys


# ─── ENCODING FIXES ──────────────────────────────────────────────────

ENCODING_MAP = {
    # Windows-1252 apostrophes/quotes decoded as UTF-8
    "ΓÇÖ": "'",
    "ΓÇÿ": "'",
    "ΓÇ£": '"',
    "ΓÇ¥": '"',
    "ΓÇô": "—",
    "ΓÇö": "–",
    "ΓÇª": "...",
    "ΓÇ¢": "'",
    # separator junk
    "ΓöÇ": "",
    # other common mojibake
    "├®": "è",
    "├⌐": "é",
    "├¿": "à",
    "├╝": "ü",
    "├Â": "ö",
    "├ñ": "ä",
    "=fyè": "",
    "=fye": "",
    # smart quotes to ASCII
    "\u2018": "'",   # '
    "\u2019": "'",   # '
    "\u201c": '"',   # "
    "\u201d": '"',   # "
    "\u2013": "-",   # –
    "\u2014": "--",  # —
    "\u2026": "...", # …
}


def fix_encoding(text):
    """Fix all encoding corruption in text."""
    for bad, good in ENCODING_MAP.items():
        text = text.replace(bad, good)
    # catch any remaining ΓÇ or Γö sequences we missed
    text = re.sub(r'Γ[öÇ][^\s]{0,2}', '', text)
    # collapse multiple spaces from removals
    text = re.sub(r'  +', ' ', text)
    return text


# ─── CHAIN-OF-THOUGHT / CHECKLIST DETECTION ──────────────────────────

COT_PATTERNS = [
    # planning preambles
    r'Plan\s*:',
    r'Turn\s+\d+\s*:',
    r'Step\s+\d+\s*:',
    # numbered planning with bold markers
    r'\d+\.\s+\*\*',
    # checklist validation
    r'\?\s*Yes\.\s*\d+\.',
    r'tags?\)?\s*\?\s*Yes',
    r'Persona adherence',
    r'responses?\s+(are\s+)?short\s*\(',
    r'Keep Klaus',
    r'Klaus must be',
    r'Klaus\'s responses must',
    r'Sound natural/casual\?',
    # meta-instructions about format
    r'<user>\s*`\s*and\s*`',
    r'<bot>\s*`\s*and\s*`',
    r'`\s*and\s*`\s*\n?\s*<bot>',
    # bold-wrapped tags
    r'\*\*\s*<user>\s*\*\*',
    r'\*\*\s*<bot>\s*\*\*',
    r'\*\*\s*(Asks|Needs|Might|Can|A final)',
]


def line_is_cot(line):
    """Check if a line looks like chain-of-thought or checklist leakage."""
    stripped = line.strip()
    if not stripped:
        return False
    for pattern in COT_PATTERNS:
        if re.search(pattern, stripped, re.IGNORECASE):
            return True
    return False


def bot_line_is_cot(text):
    """Check if a <bot> response contains checklist/planning junk."""
    # bot lines that start with validation checks
    if re.match(r'^\s*\??\s*Yes\.?\s*\d', text):
        return True
    # bot lines that are just planning
    if re.match(r'^\s*\*\*\s*(Needs|Can|Should|Will|Might)', text):
        return True
    return False


# ─── CONVERSATION PARSING AND CLEANING ────────────────────────────────

def parse_conversations(text):
    """Parse raw text into list of conversations.
    
    Each conversation is a list of (role, content) tuples.
    Conversations are separated by blank lines between the last <bot> 
    of one convo and the next <user>.
    """
    lines = text.split('\n')
    conversations = []
    current_conv = []
    current_role = None
    current_text = []
    
    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith('<user>'):
            # save previous turn
            if current_role and current_text:
                current_conv.append((current_role, ' '.join(current_text).strip()))
            
            # check if this is a new conversation (previous was bot, gap detected)
            content = line_stripped[6:].strip()  # strip <user> tag
            
            # if we had a gap (empty current_conv means fresh start)
            if current_role == 'bot' and not line_stripped:
                continue
                
            current_role = 'user'
            current_text = [content] if content else []
            
        elif line_stripped.startswith('<bot>'):
            # save previous turn
            if current_role and current_text:
                current_conv.append((current_role, ' '.join(current_text).strip()))
            
            content = line_stripped[5:].strip()  # strip <bot> tag
            current_role = 'bot'
            current_text = [content] if content else []
            
        elif not line_stripped:
            # blank line — might be conversation boundary
            if current_role and current_text:
                current_conv.append((current_role, ' '.join(current_text).strip()))
                current_role = None
                current_text = []
            if current_conv:
                conversations.append(current_conv)
                current_conv = []
        else:
            # continuation of current turn
            if current_role:
                current_text.append(line_stripped)
    
    # save final turn/conversation
    if current_role and current_text:
        current_conv.append((current_role, ' '.join(current_text).strip()))
    if current_conv:
        conversations.append(current_conv)
    
    return conversations


def clean_conversation(conv):
    """Clean a single conversation. Returns cleaned conv or None if too damaged."""
    cleaned = []
    
    for role, text in conv:
        # skip lines that are pure chain-of-thought
        if line_is_cot(text):
            continue
        
        # skip bot lines that are checklist output
        if role == 'bot' and bot_line_is_cot(text):
            continue
        
        # skip empty or near-empty turns
        if len(text.strip()) < 3:
            continue
        
        # skip turns that are just punctuation, tags, or formatting artifacts
        if re.match(r'^[\s\?\./,!`\*\-\_\|]+$', text):
            continue
        
        # skip turns that are backtick-and fragments like "` and `"
        if re.match(r'^`?\s*and\s*`?$', text.strip()):
            continue
        
        # skip turns that look like format markers (just /, ?, commas, etc)
        if len(text.strip()) < 5 and not any(c.isalnum() for c in text):
            continue
        
        # fix encoding in the text
        text = fix_encoding(text)
        
        # strip bold markers
        text = text.replace('**', '')
        
        # strip any remaining raw <user> or <bot> in the middle of text
        # (these should only be at the start of lines, not mid-sentence)
        # but be careful not to strip legitimate mentions
        
        # clean up whitespace
        text = text.strip()
        
        if text:
            cleaned.append((role, text))
    
    # validate turn structure: should alternate user/bot, start with user
    if not cleaned:
        return None
    
    # remove leading bot turns
    while cleaned and cleaned[0][0] == 'bot':
        cleaned.pop(0)
    
    if not cleaned:
        return None
    
    # ensure alternating structure — drop turns that break alternation
    validated = [cleaned[0]]
    for i in range(1, len(cleaned)):
        if cleaned[i][0] != validated[-1][0]:
            validated.append(cleaned[i])
        # else: skip duplicate role (broken alternation)
    
    # need at least one user + one bot turn
    if len(validated) < 2:
        return None
    
    # should end on a bot turn for complete conversations
    if validated[-1][0] == 'user':
        validated.pop()
    
    if len(validated) < 2:
        return None
    
    return validated


def format_conversation(conv):
    """Format a cleaned conversation back to text."""
    lines = []
    for role, text in conv:
        tag = '<user>' if role == 'user' else '<bot>'
        lines.append(f"{tag} {text}")
    return '\n'.join(lines)


# ─── MAIN ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_sft.py <input_file> [output_file]")
        print("  input_file:  path to sft_conversations.txt")
        print("  output_file: path for cleaned output (default: sft_clean.txt)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "sft_clean.txt"
    
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    
    # first pass: fix encoding on the whole file
    print("Fixing encoding...")
    raw = fix_encoding(raw)
    
    # parse into conversations
    print("Parsing conversations...")
    conversations = parse_conversations(raw)
    print(f"  Found {len(conversations)} raw conversations")
    
    # clean each conversation
    print("Cleaning...")
    cleaned = []
    dropped = 0
    for conv in conversations:
        result = clean_conversation(conv)
        if result:
            cleaned.append(result)
        else:
            dropped += 1
    
    print(f"  Kept {len(cleaned)}, dropped {dropped}")
    
    # stats
    total_turns = sum(len(c) for c in cleaned)
    user_turns = sum(1 for c in cleaned for role, _ in c if role == 'user')
    bot_turns = sum(1 for c in cleaned for role, _ in c if role == 'bot')
    
    # spot check for remaining encoding issues
    remaining_issues = 0
    for conv in cleaned:
        for _, text in conv:
            if 'ΓÇ' in text or 'Γö' in text or '=fy' in text:
                remaining_issues += 1
    
    print(f"\nStats:")
    print(f"  Conversations: {len(cleaned)}")
    print(f"  Total turns:   {total_turns}")
    print(f"  User turns:    {user_turns}")
    print(f"  Bot turns:     {bot_turns}")
    print(f"  Remaining encoding issues: {remaining_issues}")
    
    # write output
    print(f"\nWriting {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, conv in enumerate(cleaned):
            if i > 0:
                f.write('\n')  # blank line between conversations
            f.write(format_conversation(conv))
            f.write('\n')
    
    print("Done!")
    
    # show a few samples
    print(f"\n--- Sample cleaned conversations ---")
    import random
    samples = random.sample(cleaned, min(3, len(cleaned)))
    for conv in samples:
        print()
        for role, text in conv:
            tag = '<user>' if role == 'user' else '<bot>'
            preview = text[:100] + '...' if len(text) > 100 else text
            print(f"  {tag} {preview}")
        print()


if __name__ == "__main__":
    main()
