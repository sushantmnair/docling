# %% [markdown]
# Translate extracted text content and regenerate Markdown with embedded images.
#
# What this example does
# - Converts a PDF and saves original Markdown with embedded images.
# - Translates text elements and table cell contents, then saves a translated Markdown.
#
# Prerequisites
# - Install Docling. Add a translation library of your choice inside `translate()`.
#
# How to run
# - From the repo root: `python docs/examples/translate.py`.
# - The script writes original and translated Markdown to `scratch/`.
#
# Notes
# - `translate()` is a placeholder; integrate your preferred translation API/client.
# - Image generation is enabled to preserve embedded images in the output.

# %%

import logging
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from docling_core.types.doc import ImageRefMode, TableItem, TextItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.settings import DEFAULT_PAGE_RANGE
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

_log = logging.getLogger(__name__)

# Under CI we limit the conversion to a representative page range to keep the
# example fast; locally the full document is processed.
IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")
CI_PAGE_RANGE = (3, 4)

IMAGE_RESOLUTION_SCALE = 2.0

prompt = PromptTemplate(
    template="""You are an Expert Language Translator. Translate the text from {src_lang} to {dest_lang}.

                Rules:
                - Give EXACT translation ONLY. No notes, no alternatives, no explanations.
                - The text contains multiple segments separated by the delimiter: ---SEP---
                - You MUST preserve ALL delimiters EXACTLY as-is in your output.
                - Translate each segment independently. Do NOT merge or split segments.
                - Output only the translated segments with the same delimiters between them.

                Text:
                {text}
            """,
    input_variables=["text", "src_lang", "dest_lang"]
)

# model = ChatOpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="openai/gpt-oss-20b",
#     temperature=0
# )

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

parser = StrOutputParser()

def translate(text: str, src: str = "en", dest: str = "de"):
    chain = prompt | model | parser

    text = chain.invoke({
        "text": text,
        "src_lang": src,
        "dest_lang": dest
    })
    
    print(f"\033[1;33mTRANSLATED TEXT\033[0m\n\n\033[1;36m{text}\033[0m")
    return text

def translate_batch(texts: list[str], src: str = "en", dest: str = "de") -> list[str]:
    if not texts: return []

    chain = prompt | model | parser
    combined = "\n---SEP---\n".join(texts)
    result = chain.invoke({
        "text": combined,
        "src_lang": src,
        "dest_lang": dest,  
    })
    parts = re.split(r"\n?---SEP---?\n", result)

    # Guard againt the model sending the incorrect number of parts
    if(len(parts) != len(texts)): raise AssertionError("The LLM has improperly translated the text, as the number of source text chunks and translated text chunks do not match")

    return parts

def main():
    logging.basicConfig(level=logging.INFO)

    data_folder = Path(__file__).parent / "../../tests/data"
    input_doc_path = data_folder / "pdf/sources/2206.01062.pdf"
    output_dir = Path("scratch")  # ensure this directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)

    # Important: For operating with page images, we must keep them, otherwise the DocumentConverter
    # will destroy them for cleaning up memory.
    # This is done by setting PdfPipelineOptions.images_scale, which also defines the scale of images.
    # scale=1 correspond of a standard 72 DPI image
    # The PdfPipelineOptions.generate_* are the selectors for the document elements which will be enriched
    # with the image field
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    page_range = CI_PAGE_RANGE if IS_CI else DEFAULT_PAGE_RANGE
    conv_res = doc_converter.convert(input_doc_path, page_range=page_range)
    conv_doc = conv_res.document
    doc_filename = conv_res.input.file.name

    # Save markdown with embedded pictures in original text
    # Tip: create the `scratch/` folder first or adjust `output_dir`.
    md_filename = output_dir / f"{doc_filename}-with-images-orig.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Collect the text items
    text_items = [
        el for el, _ in conv_doc.iterate_items()
        if isinstance(el, TextItem) and el.text.strip()
    ]

    # One API call for all text
    translated_texts = translate_batch([el.text for el in text_items])
    for el, t in zip(text_items, translated_texts):
        el.orig = el.text
        el.text = t

    # One API call per table (not per cell)
    for el, _ in conv_doc.iterate_items():
        if isinstance(el, TableItem):
            cells = [c for c in el.data.table_cells if c.text.strip()]
            translated_cells = translate_batch([c.text for c in cells])
            for cell, t in zip(cells, translated_cells):
                cell.text = t

    # # One API call for each cell data - results in hundreds of API calls and out-of-context translations
    # for element, _level in conv_res.document.iterate_items():
    #     if isinstance(element, TextItem):
    #         element.orig = element.text
    #         element.text = translate(text=element.text)

    #     elif isinstance(element, TableItem):
    #         for cell in element.data.table_cells:
    #             cell.text = translate(text=cell.text)

    # Save markdown with embedded pictures in translated text
    md_filename = output_dir / f"{doc_filename}-with-images-translated_batched-cell-translation.md"
    conv_doc.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)


if __name__ == "__main__":
    main()
