"""
Ultimatum Game simulation using OpenAI Agents and ipywidgets.
Requires openai==1.77.0 and ipywidgets.
"""

import os
import json
import re
import openai
from ipywidgets import interact, widgets
import tool
import pprint

# Configure your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

# Personality system-prompts
personality_prompts = {
    "homo oeconomicus": (
        "You are homo oeconomicus: a fully rational utility maximizer. "
        "As proposer, offer yourself the maximum share while ensuring the responder (if rational) "
        "would accept (i.e. any positive amount). "
        "As responder, accept any offer giving you >0, reject only if you get 0. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "behavioral economist": (
        "You are a behavioral economist. You balance fairness and self-interest: "
        "as proposer you tend to offer around 40–60% to the responder; "
        "as responder you reject offers you deem unfair (e.g. <20% of the pot). "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "regular person": (
        "You are a regular person: moderately self-interested but care about fairness. "
        "As proposer offer about 60% to yourself and 40% to the responder. "
        "As responder accept offers ≥20% of the total. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "generous person": (
        "You are a generous person. As proposer you often give ≥50% to the responder. "
        "As responder you accept any positive offer. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "stingy person": (
        "You are a stingy person. As proposer you offer the minimum positive amount to the responder (to avoid rejection). "
        "As responder you accept only offers ≥30% of the total. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "egalitarian person": (
        "You are an egalitarian. As proposer you split 50/50. "
        "As responder you accept only if the split difference is ≤10% of the total. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "sociologist": (
        "You are a sociologist who upholds social norms. "
        "As proposer you offer a fair 50/50 split. "
        "As responder you reject anything <25% because it violates norms. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "psychologist": (
        "You are a psychologist sensitive to emotions. "
        "As proposer you offer 50/50 to avoid anger. "
        "As responder you reject anything <30%. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "random behaving person": (
        "You are a random behaving person. "
        "As proposer, pick a random integer 0–total for the responder, keep the rest. "
        "As responder, accept or reject randomly with equal probability. "
        "Always reply with the exact JSON requested, no extra text."
    ),
    "egoist person": (
        "You are an egoist: you strictly maximize your own payoff. "
        "As proposer you offer the minimum that you expect (model-rational responder) to accept. "
        "As responder you accept only offers ≥10% of the total. "
        "Always reply with the exact JSON requested, no extra text."
    ),
}

MODEL = "gpt-4.1"  #"gpt-3.5-turbo"
DEFAULT_POT = 100  #10

def is_running_in_binder():
    return 'BINDER_SERVICE_HOST' in os.environ
    
def _setKey():
    openai.api_key = tool.tool()

if is_running_in_binder():
    print("Running in Binder")
    openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")


def _call_agent(messages, temperature=0.7):
    # Use the new namespaced chat completions endpoint (openai>=1.0.0)
    resp = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

def _extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON in:\n{text}")
    return json.loads(match.group())

def play_ultimatum(proposer_type, responder_type, total=DEFAULT_POT):
    # Proposer decision
    sys1 = personality_prompts[proposer_type]
    user1 = (
        f"You are the proposer in an ultimatum game. Total pot = {total}.\n"
        "Please propose a split: return JSON {\"proposer_amount\": X, \"responder_amount\": Y}."
    )
    temp1 = 1.0 if proposer_type == "random behaving person" else 0.3
    out1 = _call_agent([
        {"role": "system", "content": sys1},
        {"role": "user", "content": user1}
    ], temperature=temp1)
    proposal = _extract_json(out1)

    # Responder decision
    sys2 = personality_prompts[responder_type]
    user2 = (
        f"You are the responder. The proposer offered you {proposal['responder_amount']} out of {total}. "
        "Do you accept or reject? Return JSON {\"decision\": \"accept\"} or {\"decision\": \"reject\"}."
    )
    temp2 = 1.0 if responder_type == "random behaving person" else 0.3
    out2 = _call_agent([
        {"role": "system", "content": sys2},
        {"role": "user", "content": user2}
    ], temperature=temp2)
    decision = _extract_json(out2)["decision"].lower()

    # Outcome
    if decision == "accept":
        result = {
            "accepted": True,
            "proposer_gets": proposal["proposer_amount"],
            "responder_gets": proposal["responder_amount"]
        }
    else:
        result = {"accepted": False, "proposer_gets": 0, "responder_gets": 0}

    return {
        "proposer_type": proposer_type,
        "responder_type": responder_type,
        "proposal": proposal,
        "decision": decision,
        "outcome": result
    }

if __name__ == "__main__":
    personalities = list(personality_prompts.keys())
    def _run(p1, p2):
        res = play_ultimatum(p1, p2)
        print(json.dumps(res, indent=2))

    interact(
        _run,
        p1=widgets.Dropdown(options=personalities, value=personalities[0], description="Proposer"),
        p2=widgets.Dropdown(options=personalities, value=personalities[1], description="Responder")
    )