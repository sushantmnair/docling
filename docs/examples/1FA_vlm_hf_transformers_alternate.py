from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, VlmConvertOptions
from docling.datamodel.vlm_engine_options import TransformersVlmEngineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from pathlib import Path

source = Path("tests/data/pdf/sources/2305.03393v1-pg9.pdf")

vlm_options = VlmConvertOptions.from_preset(
    "granite_docling",
    engine_options=TransformersVlmEngineOptions()
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=VlmPipelineOptions(vlm_options=vlm_options)
        )
    }
)

print(converter.convert(source=source).document.export_to_markdown())