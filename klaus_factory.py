"""
KlausCore Training Data Factory
===============================
Generates diverse training data by talking to LM Studio or Ollama APIs.
Supports multiple hosts with different backends for parallel generation.

Outputs:
  1. skill_training.csv    — (input_text, skill_label) for TinyNet
  2. sft_conversations.txt — conversation pairs for KlausGPT SFT
  3. sft_reasoning.txt     — SFT data with <think> blocks (--reasoning)
  4. compose_seeds.txt     — skill_output → natural_language pairs

Requirements:
  - LM Studio and/or Ollama running on one or more machines with a model loaded

Usage:
  python klaus_factory.py --host localhost --backend lmstudio       # single LM Studio host
  python klaus_factory.py --host localhost --backend ollama          # single Ollama host
  python klaus_factory.py --sft-only --reasoning                    # SFT data with think blocks

  # Multi-host parallel with mixed backends:
  python klaus_factory.py --hosts localhost:1234:lmstudio 10.1.10.50:11434:ollama --sft-only --reasoning

  # Three machines: desktop (LM Studio), laptop (LM Studio), server (Ollama):
  python klaus_factory.py --hosts localhost:1234:lmstudio 10.1.10.141:1234:lmstudio 10.1.10.50:11434:ollama --sft-only --reasoning --sft-count 25000
"""

import json
import time
import csv
import os
import sys
import random
import argparse
import threading
from queue import Queue, Empty
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Config ──────────────────────────────────────────────────
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1234
DEFAULT_BACKEND = "lmstudio"  # "lmstudio" or "ollama"
OLLAMA_MODEL = "deepseek-r1:8b"  # default model for Ollama hosts

SKILLS = {
    0: "GREET",
    1: "DEFINE",
    2: "FACTUAL",
    3: "RECALL",
    4: "CONVERSE",
    5: "UNKNOWN"
}

# How many diverse examples to generate per skill
EXAMPLES_PER_SKILL = 200

# How many SFT conversation pairs to generate
SFT_CONVERSATIONS = 15000

# ── API Backends ───────────────────────────────────────────

def chat_lmstudio(prompt, system="You are a helpful assistant.", host="localhost",
                  port=1234, temperature=0.9, max_tokens=1024):
    """Send a chat completion request to LM Studio."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"].strip()
            return {"text": text, "thinking": None}
    except URLError as e:
        print(f"[Factory] LM Studio connection failed: {e}")
        return None
    except Exception as e:
        print(f"[Factory] Error: {e}")
        return None


def chat_ollama(prompt, system="You are a helpful assistant.", host="localhost",
                port=11434, temperature=0.9, max_tokens=1024, model=None):
    """Send a generate request to Ollama. Returns thinking + response separately."""
    url = f"http://{host}:{port}/api/generate"
    payload = json.dumps({
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False
    }).encode("utf-8")

    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "").strip()
            thinking = data.get("thinking", "").strip() if data.get("thinking") else None
            return {"text": text, "thinking": thinking}
    except URLError as e:
        print(f"[Factory] Ollama connection failed: {e}")
        return None
    except Exception as e:
        print(f"[Factory] Error: {e}")
        return None


def chat(prompt, system="You are a helpful assistant.", host=DEFAULT_HOST,
         port=DEFAULT_PORT, temperature=0.9, max_tokens=1024, backend="lmstudio",
         ollama_model=None):
    """Unified chat function — routes to the right backend.
    Returns string for backward compat, or dict if you need thinking."""
    if backend == "ollama":
        result = chat_ollama(prompt, system=system, host=host, port=port,
                            temperature=temperature, max_tokens=max_tokens,
                            model=ollama_model)
    else:
        result = chat_lmstudio(prompt, system=system, host=host, port=port,
                              temperature=temperature, max_tokens=max_tokens)
    if result is None:
        return None
    return result["text"]


def chat_with_thinking(prompt, system="You are a helpful assistant.", host=DEFAULT_HOST,
                       port=DEFAULT_PORT, temperature=0.9, max_tokens=1024,
                       backend="lmstudio", ollama_model=None):
    """Like chat() but returns {"text": ..., "thinking": ...} dict."""
    if backend == "ollama":
        return chat_ollama(prompt, system=system, host=host, port=port,
                          temperature=temperature, max_tokens=max_tokens,
                          model=ollama_model)
    else:
        return chat_lmstudio(prompt, system=system, host=host, port=port,
                            temperature=temperature, max_tokens=max_tokens)

# ── Skill Training Data Generation ─────────────────────────
SKILL_PROMPTS = {
    "GREET": """Generate {n} diverse ways a person might greet an AI chatbot.
Include casual greetings, formal greetings, time-based greetings, slang,
different moods, and different levels of enthusiasm.
One per line, just the greeting, no numbering or labels.
Examples: hey, good morning, what's up dude, hello there, yo""",

    "DEFINE": """Generate {n} diverse ways a person might ask for a definition or explanation.
Include "what is X", "define X", "explain X", "what does X mean", "tell me about X",
and less obvious phrasings like "I don't know what X is" or "never heard of X".
Vary the topics (science, everyday words, technical terms, slang).
One per line, just the question, no numbering or labels.
Examples: what is photosynthesis, define entropy, what does ubiquitous mean""",

    "FACTUAL": """Generate {n} diverse factual questions a person might ask.
Include geography, history, science, math, sports, pop culture, measurements,
comparisons, "how many", "how tall", "when did", "who invented", "where is".
Also include indirect factual questions like "I wonder how far the moon is".
One per line, just the question, no numbering or labels.
Examples: how tall is mount everest, when did ww2 end, who painted the mona lisa""",

    "RECALL": """Generate {n} diverse ways a person might ask an AI to remember something
or recall something from a previous conversation.
Include "remember when", "you told me", "what did I say about", "last time we talked",
"do you remember", "what was that thing", "earlier you mentioned".
One per line, just the question, no numbering or labels.
Examples: do you remember my name, what did we talk about last time, you mentioned a book earlier""",

    "CONVERSE": """Generate {n} diverse casual conversation messages a person might send.
Include opinions, feelings, stories, reactions, venting, small talk, jokes,
philosophical musings, and responses that don't fit neatly into other categories.
These are things someone says when they just want to chat, not ask a specific question.
One per line, just the message, no numbering or labels.
Examples: I had the worst day today, that's pretty cool actually, I think dogs are better than cats"""
}

# Tricky edge cases that test skill boundaries
EDGE_CASE_PROMPT = """Generate {n} ambiguous or tricky user messages that could be
classified as different skills depending on context. For each message, put the message
first, then a pipe |, then the most likely skill from: GREET, DEFINE, FACTUAL, RECALL, CONVERSE.

Examples:
what is up with you today|CONVERSE
what is the meaning of life|CONVERSE
do you know what time it is|FACTUAL
hey do you remember that movie|RECALL
what is a good restaurant nearby|FACTUAL
good morning how are you feeling|GREET

Generate {n} more like these, one per line:"""


def generate_skill_data(host, port, output_path="skill_training.csv"):
    """Generate diverse training examples for each skill."""
    print(f"\n[Factory] Generating skill routing training data...")
    all_examples = []

    for skill_name, prompt_template in SKILL_PROMPTS.items():
        skill_id = [k for k, v in SKILLS.items() if v == skill_name][0]
        print(f"  Generating {EXAMPLES_PER_SKILL} examples for {skill_name}...")

        # Generate in batches of 50 to avoid hitting token limits
        batch_size = 50
        for batch in range(0, EXAMPLES_PER_SKILL, batch_size):
            n = min(batch_size, EXAMPLES_PER_SKILL - batch)
            prompt = prompt_template.format(n=n)
            response = chat(prompt, host=host, port=port,
                          system="Generate exactly what is asked. One example per line. No numbering, no labels, no explanations.",
                          temperature=1.0)
            if response:
                lines = [l.strip() for l in response.split("\n") if l.strip()]
                for line in lines:
                    # Clean up common artifacts
                    line = line.lstrip("0123456789.-) ").strip('"').strip()
                    if line and len(line) > 1 and len(line) < 300:
                        all_examples.append((line, skill_id, skill_name))
                print(f"    Batch: got {len(lines)} examples")
            else:
                print(f"    Batch failed, retrying in 2s...")
                time.sleep(2)

            time.sleep(0.5)  # be nice to the API

    # Generate edge cases
    print(f"  Generating edge cases...")
    edge_response = chat(EDGE_CASE_PROMPT.format(n=100), host=host, port=port,
                        system="Generate exactly what is asked. Follow the format precisely.",
                        temperature=1.0)
    if edge_response:
        for line in edge_response.split("\n"):
            line = line.strip()
            if "|" in line:
                parts = line.rsplit("|", 1)
                text = parts[0].strip().strip('"')
                label = parts[1].strip().upper()
                if label in SKILLS.values() and len(text) > 1:
                    skill_id = [k for k, v in SKILLS.items() if v == label][0]
                    all_examples.append((text, skill_id, label))

    # Shuffle and save
    random.shuffle(all_examples)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "skill_id", "skill_name"])
        for text, sid, sname in all_examples:
            writer.writerow([text, sid, sname])

    print(f"  Saved {len(all_examples)} examples to {output_path}")
    return all_examples


# ── SFT Conversation Data Generation ──────────────────────
SFT_SYSTEM = """You are generating training data for a small AI chatbot named Klaus.
Klaus is friendly, helpful, a bit quirky, and speaks casually.
Klaus has limited knowledge — he knows basic facts, can define words,
and can have casual conversations, but he's not an expert at anything.
When Klaus doesn't know something, he says so honestly.
Generate a realistic multi-turn conversation between a User and Klaus.
Format each turn on its own line like:
<user>the user's message
<bot>Klaus's response
Keep Klaus's responses SHORT — 1-2 sentences max. Klaus is a small model,
not a big AI. He should sound natural, not formal or verbose.
Generate {turns} turns of back-and-forth."""

SFT_REASONING_SYSTEM = """You are generating training data for a small AI chatbot named Klaus.
Klaus is friendly, helpful, a bit quirky, and speaks casually.
Klaus thinks through problems before answering by reasoning step by step.
Klaus has limited knowledge but tries to work through things logically.
When Klaus doesn't know something, he says so honestly.
Generate a realistic multi-turn conversation between a User and Klaus.
Format each turn on its own line like:
<user>the user's message
<bot><think>
Klaus's internal reasoning — break the problem into steps, consider options.
Keep it brief, 2-4 lines max.
</think>
Klaus's actual response to the user.
Keep Klaus's spoken responses SHORT — 1-2 sentences max.
Not every response needs a think block — simple greetings and small talk don't need one.
Only use <think> when Klaus needs to reason through something.
Generate {turns} turns of back-and-forth."""

# Seed topics — Gemma will expand these into thousands of specific scenarios
TOPIC_SEEDS = [
    # ── Daily life & routines ──
    "morning routines and waking up",
    "commuting, traffic, and getting to work",
    "grocery shopping and errands",
    "cooking dinner and meal prep",
    "cleaning, laundry, and household chores",
    "bedtime routines and sleep problems",
    "getting ready for the day",
    "weekend plans and lazy days",
    "running late and time management",
    "moving to a new apartment or house",

    # ── Emotions & venting ──
    "venting about a frustrating day",
    "feeling lonely and wanting to talk",
    "celebrating a personal achievement",
    "dealing with anxiety or nervousness",
    "being bored and looking for something to do",
    "feeling overwhelmed by responsibilities",
    "excitement about upcoming plans",
    "nostalgia and missing the past",
    "feeling stuck in life",
    "recovering from an embarrassing moment",

    # ── Relationships & social ──
    "making new friends as an adult",
    "roommate conflicts and boundaries",
    "planning a surprise for someone",
    "dealing with difficult family members",
    "reconnecting with an old friend",
    "navigating social events as an introvert",
    "gift ideas for different occasions",
    "neighbor interactions and complaints",
    "planning a group hangout",
    "dealing with gossip or drama",

    # ── Work & career ──
    "first day at a new job",
    "dealing with a difficult coworker",
    "asking for a raise or promotion",
    "work-life balance struggles",
    "job interview preparation",
    "quitting a job or giving notice",
    "freelancing and side hustles",
    "workplace meetings and presentations",
    "career change considerations",
    "remote work vs office life",

    # ── School & learning ──
    "studying for a big exam",
    "procrastinating on homework",
    "choosing a college major",
    "group project frustrations",
    "learning a new language",
    "online courses and self-study",
    "teacher or professor stories",
    "graduation and what comes next",
    "learning to code for the first time",
    "math struggles and breakthroughs",

    # ── Food & cooking ──
    "trying a new recipe and it going wrong",
    "debating the best pizza toppings",
    "restaurant recommendations and reviews",
    "picky eating and food preferences",
    "baking desserts and sweets",
    "meal planning on a budget",
    "food from different cultures",
    "coffee and tea preferences",
    "cooking for a date or guests",
    "fast food guilty pleasures",

    # ── Hobbies & interests ──
    "getting into a new hobby",
    "video games — favorites and recommendations",
    "reading books and book recommendations",
    "drawing, painting, and art",
    "playing or learning a musical instrument",
    "gardening and plants",
    "collecting things — coins, cards, stamps",
    "photography tips and gear",
    "board games and card games",
    "crafts, knitting, or DIY projects",

    # ── Entertainment ──
    "movie recommendations and reviews",
    "binge-watching a TV show",
    "music taste and favorite artists",
    "podcast recommendations",
    "youtube rabbit holes",
    "spoilers and plot twists",
    "unpopular entertainment opinions",
    "waiting for a sequel or new season",
    "live concerts and events",
    "animation and cartoons",

    # ── Technology ──
    "phone problems and troubleshooting",
    "choosing between tech products",
    "social media habits and opinions",
    "internet culture and memes",
    "smart home devices",
    "computer building and upgrades",
    "app recommendations",
    "online privacy and security concerns",
    "AI and what it means for the future",
    "old technology and nostalgia",

    # ── Science & nature ──
    "space, planets, and the universe",
    "how weather works and weather complaints",
    "interesting animal facts",
    "dinosaurs and prehistoric life",
    "ocean and marine life",
    "how the human body works",
    "cool chemistry and physics facts",
    "climate and environment",
    "volcanoes, earthquakes, and natural disasters",
    "evolution and biology",

    # ── Geography & travel ──
    "dream vacation destinations",
    "travel planning and packing tips",
    "road trip ideas and stories",
    "cultural differences between countries",
    "local hidden gems and recommendations",
    "airport and airplane experiences",
    "camping and outdoor adventures",
    "living abroad experiences",
    "famous landmarks and monuments",
    "budget travel tips",

    # ── Health & fitness ──
    "starting a workout routine",
    "dealing with minor illness or cold",
    "sleep quality and insomnia",
    "healthy eating habits",
    "running, jogging, or walking",
    "stretching and flexibility",
    "headaches and common ailments",
    "staying hydrated",
    "gym intimidation and etiquette",
    "mental health days and self-care",

    # ── Pets & animals ──
    "getting a new pet — cats vs dogs",
    "funny things pets do",
    "pet training tips",
    "pet health concerns",
    "exotic or unusual pets",
    "adopting vs buying pets",
    "pet names and naming stories",
    "wildlife encounters",
    "missing or lost pets",
    "multi-pet household dynamics",

    # ── Philosophy & hypotheticals ──
    "what would you do with a million dollars",
    "time travel scenarios",
    "meaning of life discussions",
    "moral dilemmas and trolley problems",
    "simulation theory and reality",
    "what makes something art",
    "fate vs free will",
    "if you could have any superpower",
    "desert island scenarios",
    "are we alone in the universe",

    # ── Humor & fun ──
    "telling jokes and puns",
    "funny personal stories and fails",
    "would you rather questions",
    "roasting each other playfully",
    "absurd hypothetical scenarios",
    "bad pickup lines",
    "tongue twisters and word games",
    "dad jokes appreciation or hatred",
    "funny misunderstandings",
    "rating things on a scale of 1-10",

    # ── Klaus self-awareness ──
    "asking Klaus what he is and how he works",
    "testing Klaus's limits and knowledge",
    "asking Klaus about his feelings or preferences",
    "comparing Klaus to bigger AI models",
    "asking Klaus to do something he can't do",
    "asking Klaus his opinion on being an AI",
    "trying to confuse or trick Klaus",
    "Klaus admitting he doesn't know something",
    "asking Klaus about his creator",
    "asking if Klaus is conscious or alive",

    # ── Corrections & pushback ──
    "correcting Klaus when he's wrong",
    "disagreeing with Klaus's suggestion",
    "user is annoyed with a response",
    "asking Klaus to rephrase or simplify",
    "user changes their mind mid-conversation",
    "clarifying a misunderstanding",
    "user provides additional context after initial question",
    "asking the same question different ways",
    "following up on a previous answer",
    "user says thanks and wraps up",

    # ── Definitions & explanations ──
    "asking what everyday words mean",
    "explaining technical jargon simply",
    "slang and informal language meanings",
    "word origins and etymology",
    "difference between similar words",
    "acronyms and abbreviations",
    "scientific terms explained simply",
    "legal or financial terms for beginners",
    "internet slang and abbreviations",
    "explaining idioms and expressions",

    # ── Factual questions ──
    "world capitals and geography trivia",
    "historical events and dates",
    "math questions and mental math",
    "unit conversions and measurements",
    "sports facts and records",
    "population and demographic questions",
    "how everyday objects work",
    "famous people and what they did",
    "world records and extremes",
    "calendar and time zone questions",

    # ── Seasonal & events ──
    "holiday traditions and celebrations",
    "new year's resolutions",
    "summer vs winter preferences",
    "birthday planning and ideas",
    "back to school feelings",
    "seasonal allergies and weather changes",
    "halloween costumes and spooky stuff",
    "valentines day opinions",
    "spring cleaning motivation",
    "end of year reflections",

    # ── Coding & programming ──
    "learning to code and where to start",
    "debugging a frustrating bug",
    "which programming language to learn first",
    "web development vs game development",
    "explaining what an API is",
    "version control and git basics",
    "coding project ideas for beginners",
    "frontend vs backend development",
    "open source and contributing to projects",
    "coding bootcamps vs self-teaching vs college",

    # ── Money & finances ──
    "budgeting and saving money",
    "credit cards and credit scores",
    "investing basics for beginners",
    "splitting bills with friends or roommates",
    "student loans and paying them off",
    "taxes and tax season stress",
    "negotiating prices or deals",
    "impulse buying and buyer's remorse",
    "side income and making extra money",
    "saving up for a big purchase",

    # ── Cars & driving ──
    "learning to drive and driving tests",
    "road rage and bad drivers",
    "car maintenance and repairs",
    "car shopping and what to look for",
    "electric vs gas vehicles",
    "long drives and road trip playlists",
    "parking struggles and tickets",
    "first car stories and memories",
    "gas prices and fuel economy",
    "car modifications and customization",

    # ── Sports & athletics ──
    "favorite sports teams and rivalries",
    "playing a sport casually vs competitively",
    "watching the big game",
    "fantasy sports and leagues",
    "extreme sports and adrenaline",
    "sports rules explained simply",
    "gym culture and workout routines",
    "pick-up games and casual sports",
    "sports injuries and recovery",
    "esports and competitive gaming",

    # ── Dating & romance ──
    "first date ideas and nerves",
    "online dating experiences",
    "how to ask someone out",
    "long-distance relationships",
    "relationship advice and communication",
    "moving in together",
    "anniversary ideas and surprises",
    "breakups and moving on",
    "love languages and compatibility",
    "meeting your partner's family",

    # ── DIY & home improvement ──
    "painting a room and choosing colors",
    "building furniture from scratch or kits",
    "fixing things around the house",
    "decorating on a budget",
    "organizing closets and storage",
    "plumbing and electrical basics",
    "lawn care and landscaping",
    "smart home setup and automation",
    "renting vs buying a home",
    "apartment hunting and what to look for",

    # ── Parenting & kids ──
    "babysitting stories and tips",
    "dealing with kids and patience",
    "choosing names for babies",
    "teaching kids new things",
    "kid-friendly activities and games",
    "school supplies and back to school",
    "explaining adult concepts to children",
    "childhood memories and growing up",
    "sibling dynamics and birth order",
    "helicopter parenting vs free-range",

    # ── Fashion & style ──
    "outfit ideas and putting looks together",
    "thrift shopping and vintage finds",
    "fashion trends that make no sense",
    "dressing for different occasions",
    "shoe collections and favorites",
    "comfort vs style debate",
    "seasonal wardrobe changes",
    "accessories and jewelry",
    "uniform or dress code frustrations",
    "personal style evolution over time",

    # ── Shopping & deals ──
    "black friday and holiday shopping",
    "online vs in-store shopping",
    "deal hunting and coupons",
    "product reviews and recommendations",
    "returning items and customer service",
    "subscription services worth keeping",
    "impulse purchases and regrets",
    "comparing brands and quality",
    "wish lists and want lists",
    "secondhand and resale shopping",

    # ── Dreams & sleep ──
    "weird dreams and what they mean",
    "recurring dreams and nightmares",
    "lucid dreaming attempts",
    "sleep schedules and night owls",
    "napping habits and power naps",
    "sleepwalking and sleep talking stories",
    "falling asleep tips and tricks",
    "dream journaling",
    "the feeling of deja vu",
    "oversleeping and missing alarms",

    # ── Productivity & organization ──
    "to-do lists and task management",
    "procrastination and how to beat it",
    "morning vs night productivity",
    "digital vs paper planners",
    "decluttering and minimalism",
    "focus techniques and deep work",
    "multitasking — does it actually work",
    "setting and achieving goals",
    "time blocking and scheduling",
    "motivation and staying disciplined",

    # ── Creative pursuits ──
    "writing stories or poetry",
    "starting a youtube channel or blog",
    "making music or beats",
    "film-making and video editing",
    "creative block and getting unstuck",
    "worldbuilding for fiction",
    "standup comedy and humor writing",
    "journaling and personal writing",
    "fan fiction and fan communities",
    "collaborative creative projects",

    # ── Culture & society ──
    "generational differences and stereotypes",
    "urban vs rural vs suburban life",
    "cultural traditions and customs",
    "language barriers and funny translations",
    "superstitions and old wives tales",
    "etiquette and manners in different places",
    "social norms that are weird when you think about it",
    "internet culture and how it changes language",
    "nostalgia for a specific decade",
    "things that are different in other countries",

    # ── Awkward & relatable ──
    "awkward silences and small talk",
    "accidentally sending a text to the wrong person",
    "forgetting someone's name mid-conversation",
    "waving back at someone who wasn't waving at you",
    "oversharing and then regretting it",
    "social battery running out at events",
    "being put on the spot unexpectedly",
    "misreading a social situation",
    "saying goodbye then walking the same direction",
    "accidentally liking an old social media post",

    # ── Pet peeves & opinions ──
    "pet peeves that drive you crazy",
    "unpopular opinions and hot takes",
    "things that are overrated",
    "things that are underrated",
    "minor inconveniences that ruin your day",
    "sounds that are annoying vs satisfying",
    "habits you wish you could break",
    "things people do that make no sense",
    "guilty pleasures you won't apologize for",
    "hills you're willing to die on",

    # ── Weather & environment ──
    "complaining about the heat or cold",
    "favorite type of weather",
    "rain and staying inside",
    "snow days and winter activities",
    "natural disasters and preparedness",
    "climate and seasons in different places",
    "thunderstorms — scary or calming",
    "humidity complaints",
    "perfect weather for different activities",
    "weather affecting mood and energy",

    # ── Life milestones & transitions ──
    "turning a milestone age",
    "moving away from home for the first time",
    "quarter-life or mid-life reflections",
    "getting your first real paycheck",
    "losing a childhood home or place",
    "learning to adult and figuring life out",
    "comparing yourself to peers",
    "feeling behind in life",
    "unexpected life changes",
    "looking back and appreciating growth",

    # ── Random & miscellaneous ──
    "conspiracy theories that are fun to think about",
    "things you'd tell your younger self",
    "if animals could talk what would they say",
    "weirdest thing you've ever seen",
    "useless talents and party tricks",
    "things you learned embarrassingly late",
    "the best and worst inventions ever",
    "what aliens would think of humans",
    "everyday mysteries nobody talks about",
    "unprompted fun facts and did-you-knows",
]

TOPIC_GEN_PROMPT = """Generate {n} SPECIFIC and UNIQUE conversation scenarios between a user and a chatbot.
Category: {category}
Each scenario should be a single sentence describing a specific situation.
Make each one different — vary the mood, context, and what the user wants.
One per line, no numbering.

Examples for "casual small talk":
- user tells the bot about their terrible commute this morning
- user is bored at work and wants someone to chat with
- user just woke up and is being grumpy but friendly
- user is excited because they just adopted a puppy
- user complains about their noisy neighbors"""


def generate_diverse_topics(host, port, total_topics=200, backend="lmstudio"):
    """Have the LM generate hundreds of specific conversation scenarios."""
    print(f"\n[Factory] Generating {total_topics} diverse topic scenarios...")
    all_topics = []
    per_seed = max(5, total_topics // len(TOPIC_SEEDS))

    for seed in TOPIC_SEEDS:
        prompt = TOPIC_GEN_PROMPT.format(n=per_seed, category=seed)
        response = chat(prompt, host=host, port=port, backend=backend,
                       system="Generate exactly what is asked. Be creative and specific. One scenario per line.",
                       temperature=1.0)
        if response:
            lines = [l.strip().lstrip("- ").strip() for l in response.split("\n") if l.strip()]
            lines = [l for l in lines if len(l) > 10 and len(l) < 200]
            all_topics.extend(lines)
            print(f"  {seed}: {len(lines)} scenarios")
        time.sleep(0.5)

    random.shuffle(all_topics)
    print(f"  Total: {len(all_topics)} unique scenarios")
    return all_topics

def strip_conversation(raw):
    """Strip Gemma's preamble/postamble, keeping only <user>/<bot> turns.

    Gemma often starts with chain-of-thought like:
      'The user wants me to generate an 8-turn conversation. Topic: ...'
    and ends with:
      'Note: This conversation demonstrates...'
    We only want the actual conversation turns.
    """
    # Normalize common alternative formats Gemma might use
    raw = raw.replace("User:", "<user>").replace("Klaus:", "<bot>")
    raw = raw.replace("**User:**", "<user>").replace("**Klaus:**", "<bot>")
    raw = raw.replace("**User**:", "<user>").replace("**Klaus**:", "<bot>")

    # Find the first <user> tag — everything before it is preamble
    first_user = raw.find("<user>")
    if first_user == -1:
        return None
    raw = raw[first_user:]

    # Walk through and keep only lines that start with <user> or <bot>
    cleaned_lines = []
    in_think = False
    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("<user>") or line.startswith("<bot>"):
            cleaned_lines.append(line)
            in_think = False
        elif "<think>" in line:
            # Think block starting — attach to previous <bot> line or add as new
            if cleaned_lines and cleaned_lines[-1].startswith("<bot>"):
                cleaned_lines[-1] += "\n" + line
            else:
                cleaned_lines.append(line)
            in_think = True
        elif "</think>" in line:
            if cleaned_lines:
                cleaned_lines[-1] += "\n" + line
            in_think = False
        elif in_think:
            # Inside a think block — keep the reasoning content
            if cleaned_lines:
                cleaned_lines[-1] += "\n" + line
        elif cleaned_lines:
            # If it doesn't start with a tag but we've already started collecting,
            # it might be a continuation of the previous turn (Gemma sometimes
            # wraps long responses). Append to previous line.
            # But if it looks like meta-text (contains keywords), skip it.
            meta_keywords = ["constraint", "checklist", "confidence score",
                           "the user wants", "generate a", "note:", "this conversation",
                           "persona:", "topic:", "turn conversation"]
            if any(kw in line.lower() for kw in meta_keywords):
                continue
            if line and cleaned_lines:
                cleaned_lines[-1] += " " + line

    if not cleaned_lines:
        return None

    # Must have at least one <user> and one <bot>
    has_user = any(l.startswith("<user>") for l in cleaned_lines)
    has_bot = any(l.startswith("<bot>") for l in cleaned_lines)
    if not has_user or not has_bot:
        return None

    return "\n".join(cleaned_lines)


def generate_sft_data(hosts, output_path="sft_conversations.txt", reasoning=False):
    """Generate conversation pairs for KlausGPT SFT.

    Args:
        hosts: list of (host, port, backend) tuples — one worker thread per host
        reasoning: if True, use reasoning system prompt with think blocks
    """
    mode_label = "reasoning" if reasoning else "standard"
    print(f"\n[Factory] Generating SFT conversation data ({mode_label} mode)...")
    print(f"  Using {len(hosts)} host(s): {', '.join(f'{h}:{p} ({b})' for h,p,b in hosts)}")

    # Step 1: generate diverse topics (uses first host — this is fast)
    topics = generate_diverse_topics(hosts[0][0], hosts[0][1], total_topics=SFT_CONVERSATIONS,
                                     backend=hosts[0][2])

    # Step 2: generate conversations in parallel across all hosts
    topic_queue = Queue()
    for i, topic in enumerate(topics[:SFT_CONVERSATIONS]):
        topic_queue.put((i, topic))

    all_conversations = []
    stats = {"done": 0, "stripped": 0, "failed": 0}
    lock = threading.Lock()

    def worker(host, port, backend, worker_id):
        while True:
            try:
                i, topic = topic_queue.get(timeout=1)
            except Empty:
                return

            turns = random.choice([3, 4, 5, 6, 8])
            prompt = (f"Generate a conversation about: {topic}\n\n"
                     f"Remember: use <user> and <bot> tags, keep Klaus's responses short (1-2 sentences).")

            if reasoning:
                system = SFT_REASONING_SYSTEM.format(turns=turns)
                prompt += ("\nInclude <think> blocks for turns where Klaus needs to reason. "
                          "Skip <think> for simple greetings or small talk.")
            else:
                system = SFT_SYSTEM.format(turns=turns)

            response = chat(prompt, system=system, host=host, port=port,
                          temperature=1.0, max_tokens=2000, backend=backend)

            with lock:
                if response:
                    cleaned = strip_conversation(response)
                    if cleaned:
                        if cleaned != response.strip():
                            stats["stripped"] += 1
                        all_conversations.append(cleaned)
                    else:
                        stats["failed"] += 1
                else:
                    stats["failed"] += 1

                stats["done"] += 1
                if stats["done"] % 100 == 0:
                    good = len(all_conversations)
                    total = stats["done"]
                    stripped = stats["stripped"]
                    failed = stats["failed"]
                    print(f"    [{total}/{SFT_CONVERSATIONS}] {good} good, {stripped} stripped, {failed} failed")

            topic_queue.task_done()

    # Launch one thread per host
    threads = []
    for idx, (host, port, backend) in enumerate(hosts):
        t = threading.Thread(target=worker, args=(host, port, backend, idx), daemon=True)
        t.start()
        threads.append(t)
        print(f"  Worker {idx} started on {host}:{port} ({backend})")

    # Wait for all work to finish
    for t in threads:
        t.join()

    # Shuffle to mix outputs from different hosts
    random.shuffle(all_conversations)

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        for convo in all_conversations:
            f.write(convo + "\n===CONV_BREAK===\n")

    print(f"  Saved {len(all_conversations)} conversations to {output_path}")
    print(f"  Stripped preamble from {stats['stripped']}, failed {stats['failed']}")
    return all_conversations

# ── Compose Seed Data Generation ──────────────────────────
COMPOSE_PROMPT = """Generate {n} examples of turning a structured AI skill output
into natural, casual language. Format: structured output first, then a pipe |,
then the natural version.

The structured output is what an AI's internal system might produce.
The natural version is how a friendly chatbot would actually say it.

Examples:
GREET: morning greeting, user_name=Mike|hey Mike! good morning, what's going on?
DEFINE: quasar = massive celestial object emitting energy|a quasar is basically this huge thing in space that blasts out insane amounts of energy
FACTUAL: mount_everest.height = 8849m|mount everest is about 8,849 meters tall
RECALL: no_memory_found|hmm I don't think we've talked about that before
CONVERSE: sentiment=positive, topic=weekend|that sounds awesome!

Generate {n} more like these, one per line:"""


def generate_compose_data(host, port, output_path="compose_seeds.txt"):
    """Generate compose seed → natural language pairs."""
    print(f"\n[Factory] Generating compose training data...")

    all_pairs = []
    batches = 10
    per_batch = 30

    for i in range(batches):
        response = chat(COMPOSE_PROMPT.format(n=per_batch), host=host, port=port,
                       system="Generate exactly what is asked. Follow the format precisely.",
                       temperature=1.0)
        if response:
            for line in response.split("\n"):
                line = line.strip()
                if "|" in line and len(line) > 10:
                    all_pairs.append(line)
        print(f"  Batch {i+1}/{batches}: {len(all_pairs)} pairs so far")
        time.sleep(0.5)

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(pair + "\n")

    print(f"  Saved {len(all_pairs)} compose pairs to {output_path}")
    return all_pairs


# ── Labeling / Verification Pass ──────────────────────────
def verify_labels(examples, host, port, sample_size=100):
    """Have the LM verify a sample of skill labels for quality control."""
    print(f"\n[Factory] Verifying {sample_size} random labels...")

    sample = random.sample(examples, min(sample_size, len(examples)))
    mismatches = 0

    for text, skill_id, skill_name in sample:
        prompt = f"""Classify this user message into exactly one category.
Categories: GREET, DEFINE, FACTUAL, RECALL, CONVERSE, UNKNOWN
Message: "{text}"
Reply with ONLY the category name, nothing else."""

        response = chat(prompt, host=host, port=port,
                       system="You are a text classifier. Reply with only the category name.",
                       temperature=0.1, max_tokens=20)
        if response:
            predicted = response.strip().upper().split()[0] if response.strip() else ""
            if predicted != skill_name:
                mismatches += 1
                if mismatches <= 10:  # show first 10
                    print(f"    Mismatch: \"{text[:50]}\" — labeled {skill_name}, LM says {predicted}")

        time.sleep(0.3)

    accuracy = (sample_size - mismatches) / sample_size * 100
    print(f"  Label agreement: {accuracy:.1f}% ({mismatches} mismatches in {sample_size} samples)")
    if accuracy < 80:
        print(f"  ⚠ Low agreement — consider reviewing the training data")
    return accuracy


# ── Main ──────────────────────────────────────────────────
def main():
    global SFT_CONVERSATIONS

    parser = argparse.ArgumentParser(description="KlausCore Training Data Factory")
    parser.add_argument("--host", default=None, help="Single host (legacy)")
    parser.add_argument("--hosts", nargs="+", default=None,
                       help="Hosts as host:port:backend (e.g. localhost:1234:lmstudio 10.1.10.50:11434:ollama)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Default port (for legacy --host)")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, help="Default backend: lmstudio or ollama")
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL, help=f"Ollama model name (default: {OLLAMA_MODEL})")
    parser.add_argument("--sft-count", type=int, default=SFT_CONVERSATIONS,
                       help=f"Number of SFT conversations to generate (default: {SFT_CONVERSATIONS})")
    parser.add_argument("--reasoning", action="store_true",
                       help="Generate SFT data with <think> reasoning blocks")
    parser.add_argument("--skills-only", action="store_true", help="Only generate skill data")
    parser.add_argument("--sft-only", action="store_true", help="Only generate SFT data")
    parser.add_argument("--compose-only", action="store_true", help="Only generate compose data")
    parser.add_argument("--verify", action="store_true", help="Run label verification")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    args = parser.parse_args()

    # Build host list: (host, port, backend)
    if args.hosts:
        hosts = []
        for h in args.hosts:
            parts = h.split(":")
            if len(parts) == 3:
                hosts.append((parts[0], int(parts[1]), parts[2]))
            elif len(parts) == 2:
                # Could be host:port or host:backend
                try:
                    port = int(parts[1])
                    hosts.append((parts[0], port, args.backend))
                except ValueError:
                    # Second part is backend name
                    backend = parts[1]
                    default_port = 11434 if backend == "ollama" else 1234
                    hosts.append((parts[0], default_port, backend))
            else:
                hosts.append((parts[0], DEFAULT_PORT, args.backend))
    elif args.host:
        hosts = [(args.host, args.port, args.backend)]
    else:
        hosts = [(DEFAULT_HOST, args.port, args.backend)]

    # Override global SFT count
    SFT_CONVERSATIONS = args.sft_count

    print("=" * 60)
    print("  KlausCore Training Data Factory")
    print("=" * 60)
    print(f"  Hosts: {', '.join(f'{h}:{p} ({b})' for h,p,b in hosts)}")
    print(f"  SFT target: {SFT_CONVERSATIONS} conversations")
    if args.reasoning:
        print(f"  Mode: REASONING (with <think> blocks)")
    print(f"  Output: {args.output_dir}")

    # Test all connections
    for host, port, backend in hosts:
        test = chat("Say 'ok' and nothing else.", host=host, port=port,
                    temperature=0.1, max_tokens=10, backend=backend)
        if not test:
            print(f"\n[Factory] Cannot connect to {host}:{port} ({backend}). Exiting.")
            return
        print(f"  {host}:{port} ({backend}) OK (responded: '{test[:50]}')")

    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Use first host for single-threaded tasks
    primary_host, primary_port, primary_backend = hosts[0]

    do_all = not (args.skills_only or args.sft_only or args.compose_only)

    if do_all or args.skills_only:
        examples = generate_skill_data(
            primary_host, primary_port,
            os.path.join(args.output_dir, "skill_training.csv"))
        if args.verify and examples:
            verify_labels(examples, primary_host, primary_port)

    if do_all or args.sft_only:
        output_name = "sft_reasoning.txt" if args.reasoning else "sft_conversations.txt"
        generate_sft_data(
            hosts,
            os.path.join(args.output_dir, output_name),
            reasoning=args.reasoning)

    if do_all or args.compose_only:
        generate_compose_data(
            primary_host, primary_port,
            os.path.join(args.output_dir, "compose_seeds.txt"))

    print(f"\n[Factory] Done! Training data saved to {args.output_dir}/")
    print(f"  Next steps:")
    print(f"    1. Review data — spot check quality")
    if args.reasoning:
        print(f"    2. Clean with clean_sft.py if needed")
        print(f"    3. SFT train KlausGPT on reasoning data")
        print(f"    4. Then RL with process reward scoring")
    else:
        print(f"    2. Feed skill data through feature vector pipeline")
        print(f"    3. Train TinyNet on (feature_vector, skill_label) pairs")
        print(f"    4. Use sft_conversations.txt to fine-tune KlausGPT")


if __name__ == "__main__":
    main()
