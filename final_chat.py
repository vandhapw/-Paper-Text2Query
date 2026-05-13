import time

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from pymongo import MongoClient

from config_utils import (
    get_benchmark_model,
    get_db_name,
    get_ollama_host,
    get_ollama_key,
    required_env,
)
from s1_decomposer import QuestionDecomposer, format_decomposition
from s2_query import run_stage2
from s3_answer import run_stage3
from s4_visualization import run_stage4


load_dotenv(override=True)


def _build_ollama_client(temperature: float = 0.3) -> ChatOllama:
    api_key = get_ollama_key()
    kwargs = {
        "model": get_benchmark_model(),
        "base_url": get_ollama_host(),
        "temperature": temperature,
    }
    if api_key:
        kwargs["api_key"] = api_key
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return ChatOllama(**kwargs)


def setup_connections() -> dict:
    mongo_client = MongoClient(required_env("MONGODB_URI"), serverSelectionTimeoutMS=5000)
    db = mongo_client[get_db_name()]
    s1_llm = _build_ollama_client()
    s3_llm = _build_ollama_client()
    return {
        "mongo_client": mongo_client,
        "klaen_col": db["plalion_klaen_sensor"],
        "company_col": db["plalion_company_sensor"],
        "weather_col": db["lighting_weatherapi"],
        "decomposer": QuestionDecomposer(groq_llm=s1_llm),
        "s3_llm": s3_llm,
    }


def close_connections(connections: dict) -> None:
    client = connections.get("mongo_client")
    if client is not None:
        client.close()


def run_tag_pipeline(question: str, verbose: bool = True, connections: dict | None = None) -> str:
    owns_connections = connections is None
    if connections is None:
        connections = setup_connections()

    try:
        if verbose:
            print(f"\n{'='*60}\n Question: {question}\n{'='*60}")

        if verbose:
            print("\n[1/4] Query Synthesis (Ollama + IR)...")
        dq = connections["decomposer"].decompose(question)
        if verbose:
            print(format_decomposition(dq))

        if verbose:
            print("\n[2/4] Query Execution (MongoDB)...")
        context = run_stage2(
            dq,
            connections["klaen_col"],
            connections["company_col"],
            connections["weather_col"],
        )
        if verbose:
            print(context)

        if verbose:
            print("\n[3/4] Answer Generation (LLM)...")
        t3 = time.monotonic()
        answer = run_stage3(question, context, dq, connections["s3_llm"])
        if dq.plan:
            dq.plan.stage3_latency_ms = (time.monotonic() - t3) * 1000

        if verbose:
            print("\n[4/4] Visualization...")
        t4 = time.monotonic()
        chart_path = run_stage4(context, dq, question)
        if dq.plan:
            dq.plan.stage4_latency_ms = (time.monotonic() - t4) * 1000

        if chart_path:
            print(f"  Chart -> {chart_path}")
            answer += f"\n\nVisualization: `{chart_path}`"

        if verbose and dq.plan:
            p = dq.plan
            total = p.stage1_latency_ms + p.stage2_latency_ms + p.stage3_latency_ms + p.stage4_latency_ms
            print(
                f"\n  Latency -> S1:{p.stage1_latency_ms:.0f}ms  "
                f"S2:{p.stage2_latency_ms:.0f}ms  S3:{p.stage3_latency_ms:.0f}ms  "
                f"S4:{p.stage4_latency_ms:.0f}ms  Total:{total:.0f}ms"
            )
            print(f" Confidence: {p.validation.confidence:.0%}  |  Source: {p.plan_source}")

        return answer
    finally:
        if owns_connections:
            close_connections(connections)


def main():
    print("\n Start Sensor Chat - type 'exit' to quit\n")
    print("Collections: plalion_klaen_sensor | plalion_company_sensor | lighting_weatherapi")
    print("-" * 60)

    connections = setup_connections()
    try:
        while True:
            try:
                question = input("\n You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n Goodbye!")
                break

            if not question:
                continue
            if question.lower() in {"exit", "quit", "bye"}:
                print(" Goodbye!")
                break

            answer = run_tag_pipeline(question, verbose=True, connections=connections)
            print(f"\n Answer:\n{answer}")
    finally:
        close_connections(connections)


if __name__ == "__main__":
    main()
