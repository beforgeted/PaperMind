"""Docling PDF parser wrapper.

Docling's `DocumentConverter` is heavyweight to construct (loads layout +
table-structure models). We instantiate it lazily once per worker process and
reuse it for every task.

The wrapper returns a `ParsedDocument` with Markdown + light metadata. The
chunking phase (Phase 4) consumes the Markdown.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Optional

import logging; logger = logging.getLogger(__name__)
from app.core.schemas import ParsedDocument


class DoclingParseError(RuntimeError):
    """Raised when Docling fails to convert a PDF."""


class DoclingService:
    def __init__(self) -> None:
        self._converter = None  # lazy

    # ---- lazy init: Docling import is expensive; defer until first use ----
    def _get_converter(self):
        if self._converter is None:
            try:
                import torch
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                from docling.document_converter import DocumentConverter, PdfFormatOption

                accelerator_module = import_module(
                    "docling.datamodel.accelerator_options"
                )
                AcceleratorDevice = accelerator_module.AcceleratorDevice
                AcceleratorOptions = accelerator_module.AcceleratorOptions
            except ImportError as exc:  # pragma: no cover
                raise DoclingParseError(
                    "Docling GPU configuration is unavailable. "
                    "Install a recent Docling version, e.g. `pip install -U docling`."
                ) from exc

            if not torch.cuda.is_available():
                raise DoclingParseError(
                    "CUDA is not available for PyTorch. Install a CUDA-enabled "
                    "PyTorch build before running Docling on GPU."
                )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4,
                device=AcceleratorDevice.CUDA,
            )

            logger.info(
                "Initializing Docling DocumentConverter on CUDA device: {}",
                torch.cuda.get_device_name(0),
            )
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                    )
                }
            )
        return self._converter

    def warmup(self) -> None:
        """主线程预热：提前完成 `import torch` / Docling 模型初始化。

        必须在「主线程」调用，避免 PyTorch 在 Windows 上由
        `asyncio.to_thread` 等子线程首次加载 `c10.dll` 时触发
        `[WinError 1114] DLL 初始化例程失败`。
        """
        self._get_converter()

    def parse_pdf(
        self,
        pdf_path: str | Path,
        task_id: str,
        original_filename: str,
    ) -> ParsedDocument:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise DoclingParseError(f"PDF not found: {pdf_path}")

        converter = self._get_converter()
        try:
            result = converter.convert(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Docling failed on {}: {}", pdf_path, exc)
            raise DoclingParseError(str(exc)) from exc

        document = getattr(result, "document", None)
        if document is None:
            raise DoclingParseError("Docling returned a result without a document.")

        try:
            markdown = document.export_to_markdown()
        except Exception as exc:  # noqa: BLE001
            raise DoclingParseError(f"export_to_markdown failed: {exc}") from exc

        title = self._extract_title(document)
        num_pages = self._safe_count(document, ("pages",))
        num_tables = self._safe_count(document, ("tables",))

        parsed = ParsedDocument(
            task_id=task_id,
            markdown=markdown,
            original_filename=original_filename,
            title=title,
            num_pages=num_pages,
            num_tables=num_tables,
            num_chars=len(markdown),
        )
        logger.info(
            "Parsed {} -> {} chars, {} pages, {} tables",
            original_filename,
            parsed.num_chars,
            parsed.num_pages,
            parsed.num_tables,
        )
        return parsed

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_title(document) -> Optional[str]:
        # Docling models expose .name or document metadata; fall back to None.
        for attr in ("name", "title"):
            value = getattr(document, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        meta = getattr(document, "metadata", None)
        if meta is not None:
            for attr in ("title", "name"):
                value = getattr(meta, attr, None)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    @staticmethod
    def _safe_count(document, attr_chain: tuple[str, ...]) -> Optional[int]:
        node = document
        for attr in attr_chain:
            node = getattr(node, attr, None)
            if node is None:
                return None
        try:
            return len(node)
        except TypeError:
            return None


_service_singleton: Optional[DoclingService] = None


def get_docling_service() -> DoclingService:
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = DoclingService()
    return _service_singleton
