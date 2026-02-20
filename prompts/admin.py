from django.contrib import admin, messages
from django.db.models import FloatField, OuterRef, Subquery

from prompts.models import PromptComponent, PromptComponentVersion
from prompts.prompt_evolution import evaluate_prompt_content, regenerate_prompt_content


class PromptComponentVersionInline(admin.TabularInline):
    model = PromptComponentVersion
    extra = 0
    fields = [
        "version",
        "status",
        "score",
        "improvement",
        "description",
        "created_by",
        "created_at",
    ]
    readonly_fields = ["created_at"]
    ordering = ["-version"]


@admin.register(PromptComponent)
class PromptComponentAdmin(admin.ModelAdmin):
    change_form_template = "admin/prompts/promptcomponent/change_form.html"
    list_display = [
        "key",
        "component_type",
        "scope",
        "mode",
        "theme",
        "active_score",
        "active_version",
        "updated_at",
    ]
    list_filter = ["component_type", "scope", "mode", "theme"]
    search_fields = ["key", "name", "description"]
    readonly_fields = ["created_at", "updated_at"]
    inlines = [PromptComponentVersionInline]
    actions = ["regenerate_and_evaluate_prompts"]

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        active_score_subquery = PromptComponentVersion.objects.filter(
            component=OuterRef("pk"),
            version=OuterRef("active_version"),
        ).values("score")[:1]
        return queryset.annotate(
            active_score_value=Subquery(
                active_score_subquery, output_field=FloatField()
            )
        )

    @admin.display(description="Active Score", ordering="active_score_value")
    def active_score(self, obj):
        return obj.active_score_value

    @admin.action(description="Regenerar prompt ativo + avaliar + ativar nova versão")
    def regenerate_and_evaluate_prompts(self, request, queryset):
        processed = 0
        for component in queryset:
            self._regenerate_component(component)
            processed += 1

        self.message_user(
            request,
            f"{processed} componente(s) regenerado(s), avaliado(s) e ativado(s).",
            level=messages.SUCCESS,
        )

    def response_change(self, request, obj):
        if "_regenerate_prompt" in request.POST:
            self._regenerate_component(obj)
            self.message_user(
                request,
                "Componente regenerado, avaliado e ativado com sucesso.",
                level=messages.SUCCESS,
            )
            return super().response_change(request, obj)
        return super().response_change(request, obj)

    def _regenerate_component(self, component):
        description_command, regenerated_prompt = regenerate_prompt_content(component)
        score, analysis, improvement = evaluate_prompt_content(
            component=component,
            description_command=description_command,
            prompt_content=regenerated_prompt,
        )

        current_active = PromptComponentVersion.objects.filter(
            component=component,
            version=component.active_version,
            status="active",
        ).first()
        latest = (
            PromptComponentVersion.objects.filter(component=component)
            .order_by("-version")
            .first()
        )
        next_version = 1 if not latest else latest.version + 1

        if current_active:
            current_active.status = "approved"
            current_active.save(update_fields=["status"])

        PromptComponentVersion.objects.create(
            component=component,
            version=next_version,
            content=regenerated_prompt,
            description=description_command,
            score=score,
            improvement=improvement,
            score_details={"analysis": analysis},
            status="active",
            change_summary=("Regenerado via action do admin com avaliação automática."),
            created_by="admin:regenerate_and_evaluate",
            parent=current_active,
        )
        component.active_version = next_version
        component.description = description_command
        component.save(update_fields=["active_version", "description", "updated_at"])


@admin.register(PromptComponentVersion)
class PromptComponentVersionAdmin(admin.ModelAdmin):
    list_display = [
        "component",
        "version",
        "status",
        "score",
        "improvement",
        "created_by",
        "created_at",
    ]
    list_filter = ["status", "component__component_type", "component__scope"]
    search_fields = ["component__key", "description", "change_summary", "content"]
    readonly_fields = ["created_at"]
    actions = ["approve_versions", "activate_versions"]

    @admin.action(description="Aprovar versões selecionadas")
    def approve_versions(self, request, queryset):
        updated = queryset.update(status="approved")
        self.message_user(
            request,
            f"{updated} versão(ões) marcada(s) como approved.",
            level=messages.SUCCESS,
        )

    @admin.action(description="Ativar versões selecionadas")
    def activate_versions(self, request, queryset):
        updated = 0
        for item in queryset.select_related("component"):
            PromptComponentVersion.objects.filter(component=item.component).exclude(
                id=item.id
            ).filter(status="active").update(status="approved")
            item.status = "active"
            item.save(update_fields=["status"])
            item.component.active_version = item.version
            item.component.save(update_fields=["active_version", "updated_at"])
            updated += 1
        self.message_user(
            request,
            f"{updated} versão(ões) ativada(s).",
            level=messages.SUCCESS,
        )
