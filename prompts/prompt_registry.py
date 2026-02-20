from dataclasses import dataclass
from typing import Dict, Optional

from prompts.models import PromptComponent, PromptComponentVersion


@dataclass(frozen=True)
class PromptSelection:
    component_key: str
    version: int
    content: str
    description: str
    score: Optional[float]


class PromptRegistry:
    SYSTEM_COMPONENT_KEY = "system.main"
    EVALUATION_COMPONENT_KEY = "evaluation.response_quality"
    RUNTIME_MAIN_COMPONENT_KEY = "runtime.main"

    def get_active_prompt(self, component_key: str) -> PromptSelection:
        component = PromptComponent.objects.filter(key=component_key).first()
        if not component:
            raise RuntimeError(f"Prompt component '{component_key}' not found.")
        if component.active_version is None:
            raise RuntimeError(
                f"Prompt component '{component_key}' has no active_version."
            )
        version = PromptComponentVersion.objects.filter(
            component=component,
            version=component.active_version,
            status="active",
        ).first()
        if not version:
            raise RuntimeError(
                f"Active prompt version not found for component '{component_key}'."
            )
        return PromptSelection(
            component_key=component.key,
            version=version.version,
            content=version.content,
            description=version.description,
            score=version.score,
        )

    def get_system_prompt(self) -> PromptSelection:
        return self.get_active_prompt(self.SYSTEM_COMPONENT_KEY)

    def get_evaluation_prompt(self) -> PromptSelection:
        return self.get_active_prompt(self.EVALUATION_COMPONENT_KEY)

    def get_runtime_prompt_for_mode(self, mode: str) -> PromptSelection:
        component_key = f"runtime.mode.{mode}"
        return self.get_active_prompt(component_key)

    def get_runtime_main_prompt(self) -> PromptSelection:
        return self.get_active_prompt(self.RUNTIME_MAIN_COMPONENT_KEY)

    def get_runtime_mode_objective_for_mode(self, mode: str) -> PromptSelection:
        component_key = f"runtime.mode_objective.{mode}"
        return self.get_active_prompt(component_key)

    def get_runtime_prompts_for_modes(self) -> Dict[str, PromptSelection]:
        selections: Dict[str, PromptSelection] = {}
        components = PromptComponent.objects.filter(
            component_type="runtime",
            scope="mode",
            active_version__isnull=False,
        ).order_by("key")
        for component in components:
            mode = component.mode
            if not mode:
                continue
            version = PromptComponentVersion.objects.filter(
                component=component,
                version=component.active_version,
                status="active",
            ).first()
            if not version:
                continue
            selections[mode] = PromptSelection(
                component_key=component.key,
                version=version.version,
                content=version.content,
                description=version.description,
                score=version.score,
            )
        return selections
