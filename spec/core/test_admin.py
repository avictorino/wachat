"""
Tests for the core admin configurations.
"""

from django.contrib import admin
from django.test import TestCase

from core.admin import RagChunkAdmin
from core.models import RagChunk


class RagChunkAdminTest(TestCase):
    """Tests for the RagChunk admin configuration."""

    def test_ragchunk_admin_is_registered(self):
        """Test that RagChunk model is registered in the admin."""
        self.assertIn(RagChunk, admin.site._registry)

    def test_ragchunk_admin_excludes_embedding_field(self):
        """Test that the embedding field is excluded from the admin form."""
        admin_class = admin.site._registry[RagChunk]
        self.assertIn("embedding", admin_class.exclude)

    def test_ragchunk_admin_list_display(self):
        """Test that the admin list display includes expected fields."""
        admin_class = admin.site._registry[RagChunk]
        expected_fields = ["id", "source", "page", "chunk_index", "type", "created_at"]
        self.assertEqual(admin_class.list_display, expected_fields)

    def test_ragchunk_admin_list_filter(self):
        """Test that the admin list filter includes expected fields."""
        admin_class = admin.site._registry[RagChunk]
        expected_filters = ["type", "source", "created_at"]
        self.assertEqual(admin_class.list_filter, expected_filters)

    def test_ragchunk_admin_search_fields(self):
        """Test that the admin search fields include expected fields."""
        admin_class = admin.site._registry[RagChunk]
        expected_search = ["id", "source", "raw_text", "text"]
        self.assertEqual(admin_class.search_fields, expected_search)
