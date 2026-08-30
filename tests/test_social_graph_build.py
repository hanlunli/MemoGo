from langchain_ollama import ChatOllama

from agents.social_creative_engagement_training.graph import build_session_graph


def test_graph_compiles_without_calling_the_api():
    llm = ChatOllama(model="llama3.3")
    graph = build_session_graph(llm)
    assert hasattr(graph, "invoke")

    node_names = set(graph.get_graph().nodes.keys())
    assert {
        "schedule_check",
        "prepare_content",
        "deliver_and_capture",
        "adaptive_adjustment",
        "session_close",
    }.issubset(node_names)
