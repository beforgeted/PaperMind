"""PaperMind 学术研究子 Agent 注册表。"""

from typing import Dict, List

from app.agent.agents.base_agent import AgentSpec
from app.agent.agents.retrieval_agent import RETRIEVAL_AGENT
from app.agent.agents.writing_agent import WRITING_AGENT
from app.agent.agents.summary_agent import SUMMARY_AGENT
from app.agent.agents.profile_agent import PROFILE_AGENT
from app.agent.agents.chat_agent import CHAT_AGENT


AGENT_SPECS: List[AgentSpec] = [
    RETRIEVAL_AGENT,
    WRITING_AGENT,
    SUMMARY_AGENT,
    PROFILE_AGENT,
]

CANONICAL_AGENT_ORDER = [agent.agent_id for agent in AGENT_SPECS]


def list_agent_specs() -> List[AgentSpec]:
    return list(AGENT_SPECS)


def list_registry_items() -> List[Dict[str, object]]:
    return [
        {
            "agent_id": agent.agent_id,
            "name": agent.name,
            "description": agent.description,
            "promptConfig": {"prompt": agent.system_prompt},
            "modelConfig": agent.model_config,
        }
        for agent in AGENT_SPECS
    ]


__all__ = ["AGENT_SPECS", "CANONICAL_AGENT_ORDER", "list_agent_specs", "list_registry_items"]
