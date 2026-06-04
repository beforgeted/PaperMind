"""PaperMind Agent 基础定义。"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    name: str
    description: str
    system_prompt: str
    model_config: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """供后续工具增强型智能体使用的轻量封装。"""

    spec: AgentSpec

    def as_registry_item(self) -> Dict[str, Any]:
        return {
            "agent_id": self.spec.agent_id,
            "name": self.spec.name,
            "description": self.spec.description,
            "promptConfig": {"prompt": self.spec.system_prompt},
            "modelConfig": self.spec.model_config,
        }
