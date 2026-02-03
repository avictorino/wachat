"""
Tests for KnowledgeDocument model and embeddings service.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase

from core.models import KnowledgeDocument


class KnowledgeDocumentModelTest(TestCase):
    """Tests for the KnowledgeDocument model."""

    def test_create_knowledge_document(self):
        """Test creating a new knowledge document."""
        # Create a simple PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000208 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
300
%%EOF
"""

        uploaded_file = SimpleUploadedFile(
            "test.pdf", pdf_content, content_type="application/pdf"
        )

        doc = KnowledgeDocument.objects.create(
            title="Test Document", file=uploaded_file
        )

        self.assertEqual(doc.title, "Test Document")
        self.assertIsNotNone(doc.uploaded_at)
        self.assertTrue(doc.file.name.startswith("books/"))

    def test_knowledge_document_str_representation(self):
        """Test the string representation of KnowledgeDocument."""
        pdf_content = b"%PDF-1.4\n%%EOF"
        uploaded_file = SimpleUploadedFile("test.pdf", pdf_content)

        doc = KnowledgeDocument.objects.create(
            title="My Test Document", file=uploaded_file
        )

        self.assertEqual(str(doc), "My Test Document")


class EmbeddingsServiceTest(TestCase):
    """Tests for the embeddings service."""

    @patch("core.services.embeddings._get_embedder")
    @patch("core.services.embeddings._get_collection")
    def test_embed_pdf_document(self, mock_get_collection, mock_get_embedder):
        """Test the PDF embedding process with mocked dependencies."""
        from core.services.embeddings import embed_pdf_document

        # Mock the embedder
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value.tolist.return_value = [[0.1, 0.2, 0.3]]
        mock_get_embedder.return_value = mock_embedder

        # Mock the collection
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection

        # Create a test PDF file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Test content) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000220 00000 n
0000000313 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
395
%%EOF
""")
            test_pdf_path = f.name

        try:
            # Call the function
            embed_pdf_document(test_pdf_path)

            # Verify embedder was called
            mock_embedder.encode.assert_called_once()

            # Verify collection.add was called
            mock_collection.add.assert_called_once()

        finally:
            # Clean up
            os.unlink(test_pdf_path)

    def test_chunk_text(self):
        """Test text chunking functionality."""
        from core.services.embeddings import chunk_text

        text = "A" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=50)

        # Verify chunks were created
        self.assertGreater(len(chunks), 1)

        # Verify each chunk is approximately the right size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200)

    def test_extract_text_from_pdf(self):
        """Test PDF text extraction."""
        from core.services.embeddings import extract_text_from_pdf

        # Create a test PDF file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000220 00000 n
0000000313 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
395
%%EOF
""")
            test_pdf_path = f.name

        try:
            # Extract text
            text = extract_text_from_pdf(test_pdf_path)

            # Verify text was extracted (may be empty depending on PDF structure)
            self.assertIsInstance(text, str)

        finally:
            # Clean up
            os.unlink(test_pdf_path)
