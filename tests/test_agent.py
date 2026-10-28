import json

from app.agent import InvestigationAgent, _parse_final
from app.llm import ChatResult, LLMError, Usage


class ScriptedLLM:
    """Plays back a fixed list of responses; records the messages it was sent."""
    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.seen = []

    def chat(self, messages, tools=None, temperature=0.0, max_tokens=1024):
        self.seen.append(messages)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def tool_call(name, **args):
    return ChatResult(content=None, tool_calls=[{"id": "c1", "name": name, "arguments": json.dumps(args)}],
                      usage=Usage(1, 100, 10, 0.1))


def final(**obj):
    return ChatResult(content=json.dumps(obj), usage=Usage(1, 200, 50, 0.1))


def test_multi_hop_with_tool_call_and_citation_validation(loaded_engine):
    llm = ScriptedLLM([
        tool_call("search_documents", query="Ledger team lead"),
        final(answer="Ledger-service v4.2 caused it [S1]; the team lead is Nino Kapanadze [S99].",
              short_answer="Nino Kapanadze", citations=["S1", "S99"], conflicts=[], insufficient=False),
    ])
    a = InvestigationAgent(loaded_engine, llm).ask("Who leads the team behind INC-1042?")
    assert a.status == "ok" and a.short_answer == "Nino Kapanadze"
    assert [c.label for c in a.citations] == ["S1"]
    assert a.invalid_citations == ["S99"]  # hallucinated label is dropped, not shown as a source
    assert any(t["tool"] == "search_documents" and t["step"] == 1 for t in a.trace)
    tool_msg = [m for m in llm.seen[1] if m["role"] == "tool"][0]
    assert "<passage>" in tool_msg["content"]
    assert a.usage["llm_calls"] == 2


def test_tool_call_extras_are_echoed_back(loaded_engine):
    # Gemini 3 rejects the follow-up request (400) unless each tool call's thought_signature is sent back.
    sig = {"google": {"thought_signature": "abc"}}
    call = ChatResult(content=None, usage=Usage(1, 100, 10, 0.1), tool_calls=[
        {"id": "c1", "name": "search_documents", "arguments": json.dumps({"query": "INC-1042"}), "extra_content": sig}])
    llm = ScriptedLLM([call, final(answer="Done [S1].", short_answer="", citations=["S1"], conflicts=[],
                                   insufficient=False)])
    InvestigationAgent(loaded_engine, llm).ask("What happened in INC-1042?")
    echoed = [m for m in llm.seen[1] if m["role"] == "assistant"][0]["tool_calls"][0]
    assert echoed["extra_content"] == sig
    assert echoed["function"]["name"] == "search_documents"

def test_conflicts_are_kept_with_valid_labels(loaded_engine):
    llm = ScriptedLLM([final(answer="Resolved at 14:20 [S1] or 11:05 [S2].", short_answer="",
                             citations=["S1", "S2"], insufficient=False,
                             conflicts=[{"description": "resolution time differs", "citations": ["S1", "S2", "S50"]}])])
    a = InvestigationAgent(loaded_engine, llm).ask("When was INC-1042 resolved?")
    assert a.conflicts == [{"description": "resolution time differs", "citations": ["S1", "S2"]}]


def test_insufficient_and_no_citations(loaded_engine):
    llm = ScriptedLLM([final(answer="The documents do not say.", short_answer="", citations=[], conflicts=[],
                             insufficient=True)])
    a = InvestigationAgent(loaded_engine, llm).ask("Who is the CEO?")
    assert a.insufficient and a.citations == []


def test_llm_failure_degrades_gracefully(loaded_engine):
    a = InvestigationAgent(loaded_engine, ScriptedLLM([LLMError("rate limited")])).ask("When was INC-1042 resolved?")
    assert a.status == "degraded" and a.insufficient and a.citations
    a = InvestigationAgent(loaded_engine, None).ask("anything about ledger")
    assert a.status == "degraded"


def test_prose_answer_is_reformatted_then_forced(loaded_engine):
    llm = ScriptedLLM([ChatResult(content="It was v4.2 [S1]."), ChatResult(content="still prose [S1]"),
                       tool_call("read_document", doc="syn::runbook-ledger-service"),
                       tool_call("search_documents", query=""),
                       final(answer="v4.2 [S1]", short_answer="v4.2", citations=["S1"], conflicts=[],
                             insufficient=False)])
    a = InvestigationAgent(loaded_engine, llm, max_steps=4).ask("Which version was deployed in CHG-3301?")
    assert a.short_answer == "v4.2"
    assert a.trace[-1]["args"] == {"forced": True}


def test_cache_hit(loaded_engine):
    llm = ScriptedLLM([final(answer="x [S1]", short_answer="x", citations=["S1"], conflicts=[], insufficient=False)])
    agent = InvestigationAgent(loaded_engine, llm)
    agent.ask("cache me please")
    assert agent.ask("Cache me please").cached  # second call never reaches the LLM


def test_parse_final_handles_fences():
    assert _parse_final('```json\n{"answer": "a"}\n```') == {"answer": "a"}
    assert _parse_final("no json here") is None
