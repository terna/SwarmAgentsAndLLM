import os
from openai import OpenAI

def get_env_var(name):
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(f"Environment variable '{name}' not set")
    return value

client = OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
from serpapi import GoogleSearch


def agent_refine_query(topic):
    """Agent A: Refines the user topic into an effective web search query."""
    prompt = (
        "You are Agent A. Your task is to refine the user's topic into "
        "a concise, effective web search query.\n"
        f"User topic: \"{topic}\"\n"
        "Provide only the search query."
    )
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=64)
    return response.choices[0].message.content.strip()

def agent_search_query(query, api_key, num_results=5):
    """Agent B: Executes the web search using SerpAPI and returns results."""
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": num_results,
    }
    try:
        search = GoogleSearch(params)
        response = search.get_dict()
    except Exception as e:
        raise RuntimeError(f"SerpAPI request failed: {e}")
    # Check for API errors
    if isinstance(response, dict):
        # Standard error key
        if "error" in response:
            raise RuntimeError(f"SerpAPI error: {response.get('error')}")
        # Some versions may use 'error_message'
        if "error_message" in response:
            raise RuntimeError(f"SerpAPI error: {response.get('error_message')}")
        # Return organic search results, if any
        return response.get("organic_results", [])
    # Unexpected response format
    raise RuntimeError(f"Unexpected SerpAPI response: {response}")

def agent_analyze_results(topic, results):
    """Agent A: Analyzes the search results and summarizes key findings."""
    summary_input = "\n".join(
        f"- {item.get('title')}: {item.get('snippet')}" for item in results
    )
    prompt = (
        "You are Agent A. Here are the search results for the topic:\n"
        f"Topic: \"{topic}\"\n\n"
        "Search Results:\n"
        f"{summary_input}\n\n"
        "Provide a concise summary of the main points relevant to the topic."
    )
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a knowledgeable research assistant."},
        {"role": "user", "content": prompt},
    ],
    temperature=0.7,
    max_tokens=256)
    return response.choices[0].message.content.strip()

def main():
    # Load SerpAPI key (allow SERPAPI_API_KEY or SERPAPI_KEY)
    serpapi_api_key = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")
    if not serpapi_api_key:
        print("Error: Environment variable SERPAPI_API_KEY (or SERPAPI_KEY) not set. Exiting.")
        return

    # Get user topic
    topic = input("Enter the topic or argument to search: ").strip()
    if not topic:
        print("No topic entered. Exiting.")
        return

    # Agent A refines the query
    print("\nAgent A is refining your topic into a search query...")
    query = agent_refine_query(topic)
    print(f"Refined Query: {query}\n")

    # Agent B performs the search
    print("Agent B is searching the web...")
    try:
        results = agent_search_query(query, serpapi_api_key, num_results=5)
    except RuntimeError as e:
        print(f"Error during web search: {e}")
        return
    if not results:
        print("No results found. Please check your API key or try refining your query.")
        return

    # Agent A analyzes the results
    print("\nAgent A is analyzing the results...")
    summary = agent_analyze_results(topic, results)
    print("\nSummary of findings:")
    print(summary)

if __name__ == "__main__":
    main()
