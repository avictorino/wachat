from django.contrib import admin
from django.contrib import messages

from core.models import KnowledgeDocument, Message, Profile
from core.services.embeddings import embed_pdf_document


class MessageInline(admin.TabularInline):
    """
    Inline admin for displaying messages within Profile admin.

    Messages are read-only in the admin because they should only be created
    through the application logic (webhook handlers, conversation flows, etc.)
    to maintain data integrity and proper channel tracking.
    """

    model = Message
    extra = 0
    readonly_fields = ["role", "content", "channel", "created_at"]
    can_delete = False
    fields = ["role", "channel", "content", "created_at"]
    ordering = ["-created_at"]

    def has_add_permission(self, request, obj=None):
        """Disable adding messages directly from admin - they should be created via app logic."""
        return False


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    """Admin interface for Profile model."""

    list_display = [
        "id",
        "telegram_user_id",
        "name",
        "phone_number",
        "inferred_gender",
        "created_at",
    ]
    list_filter = ["inferred_gender", "created_at"]
    search_fields = ["telegram_user_id", "name", "phone_number"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["-created_at"]
    inlines = [MessageInline]


@admin.register(KnowledgeDocument)
class KnowledgeDocumentAdmin(admin.ModelAdmin):
    """Admin interface for KnowledgeDocument model."""

    list_display = ["title", "uploaded_at"]
    readonly_fields = ["uploaded_at"]
    ordering = ["-uploaded_at"]

    def save_model(self, request, obj, form, change):
        """
        Override save_model to trigger embedding generation after file upload.

        This ensures that whenever a PDF is uploaded through the admin,
        it is automatically indexed for RAG.
        """
        super().save_model(request, obj, form, change)

        # Call embedding logic after file is saved
        # TODO: Consider moving this to an async task (Celery, Django-Q, etc.)
        # to avoid blocking the admin UI for large PDFs
        if obj.file:
            try:
                embed_pdf_document(obj.file.path)
                messages.success(
                    request, f"PDF '{obj.title}' uploaded and indexed successfully."
                )
            except Exception as e:
                messages.error(
                    request,
                    f"PDF '{obj.title}' was uploaded but indexing failed: {str(e)}. "
                    f"The document was saved but may need to be re-indexed.",
                )
                # Log the error for debugging
                import logging

                logger = logging.getLogger(__name__)
                logger.exception(f"Failed to index PDF {obj.file.path}")
