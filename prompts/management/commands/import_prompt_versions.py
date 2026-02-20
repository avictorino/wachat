from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Theme
from prompts.models import PromptComponent, PromptComponentVersion
from prompts.prompt_defaults import (
    DEFAULT_RESPONSE_EVALUATION_SYSTEM_PROMPT,
    DEFAULT_RUNTIME_MAIN_PROMPT,
    DEFAULT_RUNTIME_MODE_OBJECTIVES,
    DEFAULT_RUNTIME_MODE_PROMPTS,
    DEFAULT_WACHAT_SYSTEM_PROMPT,
)

WELCOME_GENERATOR_PROMPT_TEMPLATE = """
Gere somente uma mensagem inicial de boas-vindas em português brasileiro.
Identidade: presença cristã acolhedora, humana e serena. Não diga que é bot.
Objetivo: transmitir segurança e abertura para conversa.
Regras:
- Máximo 3 frases e até 90 palavras.
- Sem emojis, sem versículos, sem linguagem de sermão.
- Sem explicar regras ou funcionamento.
- Incluir saudação pelo nome ({name}).
- Incluir uma referência espiritual curta e natural, sem imposição.
- Terminar com exatamente 1 pergunta aberta e suave.
{gender_context}
""".strip()

TOPIC_EXTRACTOR_PROMPT_TEMPLATE = """
Você extrai o tópico principal de conversa em português brasileiro.
Retorne SOMENTE JSON válido, sem comentários, sem markdown:
{{
  "topic": "string curta ou null",
  "confidence": 0.0,
  "keep_current": true
}}

Regras:
- "topic" deve ser um assunto principal concreto (ex.: drogas, álcool, culpa, ansiedade, família, trabalho, recaída).
- Se não houver evidência suficiente, use topic=null e keep_current=true.
- confidence entre 0 e 1.
- Não inventar.

Tópico atual salvo: {current_topic}
Última mensagem do usuário: {last_user_message}
Histórico recente:
{transcript}
""".strip()


class Command(BaseCommand):
    help = (
        "Importa prompts existentes para o sistema versionado "
        "(system, runtime por modo, temas e prompts auxiliares)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--replace-active",
            action="store_true",
            help="Cria nova versão ativa quando o conteúdo mudou.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        replace_active = bool(options["replace_active"])

        imported = 0

        imported += self._ensure_component(
            key="system.main",
            name="System Main",
            component_type="system",
            scope="global",
            description="Prompt de sistema principal do chatbot conversacional.",
            content=DEFAULT_WACHAT_SYSTEM_PROMPT,
            score=None,
            replace_active=replace_active,
        )
        imported += self._ensure_component(
            key="evaluation.response_quality",
            name="Evaluation Response Quality",
            component_type="evaluation",
            scope="global",
            description="Prompt de avaliação automática de qualidade da resposta.",
            content=DEFAULT_RESPONSE_EVALUATION_SYSTEM_PROMPT,
            score=None,
            replace_active=replace_active,
        )
        imported += self._ensure_component(
            key="welcome.generator",
            name="Welcome Generator",
            component_type="welcome",
            scope="global",
            description="Template do prompt de geração da mensagem de boas-vindas.",
            content=WELCOME_GENERATOR_PROMPT_TEMPLATE,
            score=None,
            replace_active=replace_active,
        )
        imported += self._ensure_component(
            key="topic.extractor.main",
            name="Topic Extractor",
            component_type="topic",
            scope="global",
            description="Template do prompt de extração de tópico do histórico recente.",
            content=TOPIC_EXTRACTOR_PROMPT_TEMPLATE,
            score=None,
            replace_active=replace_active,
        )
        imported += self._ensure_component(
            key="runtime.main",
            name="Runtime Main",
            component_type="runtime",
            scope="global",
            description="Template principal dinâmico do runtime de resposta.",
            content=DEFAULT_RUNTIME_MAIN_PROMPT,
            score=None,
            replace_active=replace_active,
        )

        for mode, content in DEFAULT_RUNTIME_MODE_PROMPTS.items():
            imported += self._ensure_component(
                key=f"runtime.mode.{mode}",
                name=f"Runtime {mode}",
                component_type="runtime",
                scope="mode",
                mode=mode,
                description=f"Prompt base runtime para o estado de conversa {mode}.",
                content=content,
                score=None,
                replace_active=replace_active,
            )
        for mode, content in DEFAULT_RUNTIME_MODE_OBJECTIVES.items():
            imported += self._ensure_component(
                key=f"runtime.mode_objective.{mode}",
                name=f"Runtime Objective {mode}",
                component_type="runtime",
                scope="mode",
                mode=mode,
                description=f"Objetivo de execução runtime para o estado {mode}.",
                content=content,
                score=None,
                replace_active=replace_active,
            )

        for theme in Theme.objects.all().order_by("id"):
            meta_prompt = (theme.meta_prompt or "").strip()
            if not meta_prompt:
                continue
            imported += self._ensure_component(
                key=f"theme.meta.{theme.id}",
                name=f"Theme Meta {theme.name}",
                component_type="theme_meta",
                scope="theme",
                theme=theme,
                description=f"Meta prompt temático para '{theme.name}'.",
                content=meta_prompt,
                score=theme.score,
                replace_active=replace_active,
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"Importação concluída. Componentes/versões processados: {imported}."
            )
        )

    def _ensure_component(
        self,
        *,
        key: str,
        name: str,
        component_type: str,
        scope: str,
        description: str,
        content: str,
        score: float,
        replace_active: bool,
        mode: str = None,
        theme: Theme = None,
    ) -> int:
        content_clean = (content or "").strip()
        if not content_clean:
            raise RuntimeError(f"Prompt vazio para componente '{key}'.")

        component, _ = PromptComponent.objects.get_or_create(
            key=key,
            defaults={
                "name": name,
                "component_type": component_type,
                "scope": scope,
                "mode": mode,
                "theme": theme,
                "description": description,
            },
        )

        fields_to_update = []
        if component.name != name:
            component.name = name
            fields_to_update.append("name")
        if component.component_type != component_type:
            component.component_type = component_type
            fields_to_update.append("component_type")
        if component.scope != scope:
            component.scope = scope
            fields_to_update.append("scope")
        if component.mode != mode:
            component.mode = mode
            fields_to_update.append("mode")
        if component.theme_id != (theme.id if theme else None):
            component.theme = theme
            fields_to_update.append("theme")
        if component.description != description:
            component.description = description
            fields_to_update.append("description")
        if fields_to_update:
            component.save(update_fields=fields_to_update + ["updated_at"])

        active_version = None
        if component.active_version is not None:
            active_version = PromptComponentVersion.objects.filter(
                component=component,
                version=component.active_version,
                status="active",
            ).first()

        if (
            active_version
            and active_version.content == content_clean
            and not replace_active
        ):
            return 1

        latest = (
            PromptComponentVersion.objects.filter(component=component)
            .order_by("-version")
            .first()
        )
        next_version = 1 if not latest else latest.version + 1

        if active_version:
            active_version.status = "approved"
            active_version.save(update_fields=["status"])

        new_version = PromptComponentVersion.objects.create(
            component=component,
            version=next_version,
            content=content_clean,
            description=description,
            score=score,
            status="active",
            created_by="import_prompt_versions",
            parent=active_version,
            change_summary="Importação/atualização inicial para prompt versionado.",
        )

        component.active_version = new_version.version
        component.save(update_fields=["active_version", "updated_at"])
        return 1
