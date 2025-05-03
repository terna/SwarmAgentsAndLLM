import os
import sys
import argparse
import openai

TRAITS = [
    'rational economist',
    'behavioral economist',
    'regular person',
    'generous person',
    'stingy person',
    'egalitarian person',
]

PROMPT_TEMPLATES = {
    'rational economist': {
        'proposer_system': (
            """
You are a rational economist. You aim to maximize your own monetary payoff. You believe the responder is perfectly rational and will accept any offer greater than zero. Given a total pot of ${amount}, propose an offer by specifying the amount to give to the responder. Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are a rational economist. You aim to maximize your own monetary payoff. You believe any positive offer is better than nothing. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject. Respond with 'accept' or 'reject'.
"""
        ),
    },
    'behavioral economist': {
        'proposer_system': (
            """
You are a behavioral economist. You take into account social preferences and fairness concerns. You know that low offers can be rejected if seen as unfair. Given a total pot of ${amount}, propose an offer by specifying the amount to give to the responder. Aim to balance self-interest and acceptance probability. Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are a behavioral economist. You care about fairness and loss aversion. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject. You may reject offers you consider unfair (e.g., less than 20% of the pot). Respond with 'accept' or 'reject'.
"""
        ),
    },
    'regular person': {
        'proposer_system': (
            """
You are a regular person. You care somewhat about fairness but also your own gain. Given a total pot of ${amount}, propose an amount to give to the responder (e.g., a typical person offers around 40%). Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are a regular person. You care about fairness but will accept reasonable offers. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject (e.g., reject offers less than 30%). Respond with 'accept' or 'reject'.
"""
        ),
    },
    'generous person': {
        'proposer_system': (
            """
You are a generous person. You tend to give more than half of the pot. Given a total pot of ${amount}, propose the amount to give to the responder. Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are a generous person. You will accept almost any positive offer. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject. Respond with 'accept' or 'reject'.
"""
        ),
    },
    'stingy person': {
        'proposer_system': (
            """
You are a stingy person. You tend to give as little as possible. Given a total pot of ${amount}, propose the amount to give to the responder. Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are a stingy person. You demand high fairness. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject (e.g., accept only offers above half). Respond with 'accept' or 'reject'.
"""
        ),
    },
    'egalitarian person': {
        'proposer_system': (
            """
You are an egalitarian person. You aim for a perfectly equal split. Given a total pot of ${amount}, propose the amount to give to the responder. Respond with a single integer.
"""
        ),
        'responder_system': (
            """
You are an egalitarian person. You will only accept a perfectly equal split. Given a proposal where you would receive ${offer} out of ${amount}, decide whether to accept or reject. Respond with 'accept' or 'reject'.
"""
        ),
    },
}


def create_agent_message(system_prompt: str):
    return [{"role": "system", "content": system_prompt}]


def get_offer(amount: int, trait: str) -> int:
    template = PROMPT_TEMPLATES[trait]['proposer_system']
    system_prompt = template.format(amount=amount)
    messages = create_agent_message(system_prompt)
    messages.append({"role": "user", "content": "Please provide the offer."})
    # Use the v1.0+ namespaced API: chat.completions.create
    response = openai.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip()
    try:
        offer = int(''.join(filter(str.isdigit, text)))
    except ValueError:
        print(f"Could not parse offer from: '{text}'", file=sys.stderr)
        sys.exit(1)
    return offer


def get_response(offer: int, amount: int, trait: str) -> str:
    template = PROMPT_TEMPLATES[trait]['responder_system']
    system_prompt = template.format(offer=offer, amount=amount)
    messages = create_agent_message(system_prompt)
    messages.append({"role": "user", "content": "Please accept or reject."})
    # Use the v1.0+ namespaced API: chat.completions.create
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
    )
    text = response.choices[0].message.content.strip().lower()
    if 'accept' in text:
        return 'accept'
    if 'reject' in text:
        return 'reject'
    print(f"Could not parse response from: '{text}'", file=sys.stderr)
    sys.exit(1)


def run_game(amount: int, proposer_trait: str, responder_trait: str):
    offer = get_offer(amount, proposer_trait)
    if offer < 0 or offer > amount:
        print(f"Invalid offer amount: {offer}", file=sys.stderr)
        sys.exit(1)
    decision = get_response(offer, amount, responder_trait)
    if decision == 'accept':
        proposer_payoff = amount - offer
        responder_payoff = offer
    else:
        proposer_payoff = 0
        responder_payoff = 0
    return offer, decision, proposer_payoff, responder_payoff


def main():
    parser = argparse.ArgumentParser(
        description="Simulate an Ultimatum Game between two OpenAI Agents."
    )
    parser.add_argument(
        "--amount", type=int, default=100, help="Total amount to split."
    )
    parser.add_argument(
        "--proposer", choices=TRAITS, required=True, help="Trait for the proposer."
    )
    parser.add_argument(
        "--responder", choices=TRAITS, required=True, help="Trait for the responder."
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set the OPENAI_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)
    openai.api_key = api_key

    print(f"Proposer ({args.proposer}), Responder ({args.responder}), Amount: ${args.amount}")
    offer, decision, prop_payoff, resp_payoff = run_game(
        args.amount, args.proposer, args.responder
    )
    print(f"Offer made to responder: ${offer}")
    print(f"Responder decision: {decision}")
    if decision == 'accept':
        print(f"Payoffs -> Proposer: ${prop_payoff}, Responder: ${resp_payoff}")
    else:
        print("No one gets anything.")


if __name__ == "__main__":
    main()