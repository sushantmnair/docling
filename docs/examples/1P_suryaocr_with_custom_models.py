# %% [markdown]
# Integrating SuryaOCR with Docling for PDF OCR and Markdown Export
#
# Overview:
#
# - Configures SuryaOCR options for OCR.
# - Executes PDF pipeline with SuryaOCR integration.
# - Models auto-download from Hugging Face on first run.
#
# Prerequisites:
#
# - Install: `pip install docling-surya`
# - Ensure `docling` imports successfully.
#
# Execution:
#
# - Run from repo root: `python docs/examples/suryaocr_with_custom_models.py`
# - Outputs Markdown to stdout.
#
# Notes:
#
# - Default source: EPA PDF URL; substitute with local path as needed.
# - Models cached in `~/.cache/huggingface`; override with HF_HOME env var.
# - Use proxy config for restricted networks.
# - **Important Licensing Note**: The `docling-surya` package integrates SuryaOCR, which is licensed under the GNU General Public License (GPL).
#
# Using this integration may impose GPL obligations on your project. Review the license terms carefully.

# Requires `pip install docling-surya`
# See [https://pypi.org/project/docling-surya/](https://pypi.org/project/docling-surya/)

# %%

"""
from docling_surya import SuryaOcrOptions

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    source = "https://19january2021snapshot.epa.gov/sites/static/files/2016-02/documents/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf"

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_model="suryaocr",
        allow_external_plugins=True,
        ocr_options=SuryaOcrOptions(lang=["en"]),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )

    result = converter.convert(source)
    print(result.document.export_to_markdown())


if __name__ == "__main__":
    main()
"""

# %%
"""
Surya OCR: This example was not run due to a dependency conflict. A resolution was not pursued since it was deemed to be not worth the effort. This model needs GPU even more than Easy OCR, since it is a full-fledged Transformer model as opposed to the CRNN architecture used by EasyOCR. The only difference between the two is that Easy OCR was trained on scenes and text while Surya OCR was trained specifically on documents, which makes it better in detecting document sections. If no other OCR model is able to perform the task, Surya OCR may be considered. Since such a situation is rare, further efforts are dropped.
An alternative is to use docling's VLM API pipeline with the HF endpoint
"""

# %%
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions, VlmConvertOptions
from docling.datamodel.vlm_engine_options import ApiVlmEngineOptions, VlmEngineType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    source = "https://19january2021snapshot.epa.gov/sites/static/files/2016-02/documents/epa_sample_letter_sent_to_commissioners_dated_february_29_2015.pdf"

    vlm_options = VlmConvertOptions.from_preset(
        "qwen", # The closest general vision preset
        engine_options=ApiVlmEngineOptions(
            runtime_type=VlmEngineType.API,
            url="https://router.huggingface.co/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv("HF_TOKEN")}"},
            params={
                "model": "datalab-to/surya-ocr-2:featherless-ai",
                "temperature": 0.0,
                "max_tokens": 4096
            },
            timeout=90
        )
    )

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_options,
        enable_remote_services=True
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(source)
    print(result.document.export_to_markdown())

if __name__ == "__main__": main()

# %%
"""
The problem with this specific document is that it's a scanned PDF — there's no embedded text layer. GraniteDocling detects the layout correctly (Text, List-Group, Section-Header, Page-Footer) but has no text to extract since OCR wasn't run.

RapidOCR will extract the actual text from the scanned pages. The VLM pipeline is the wrong tool for this document.
"""