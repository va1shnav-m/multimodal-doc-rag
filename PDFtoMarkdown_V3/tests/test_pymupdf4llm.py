# import re
# from pathlib import Path

# import fitz
# import pymupdf4llm

# # -------------------------------------------------------------
# # Configuration
# # -------------------------------------------------------------

# PDF_PATH = "srs_example_2010_group2 (1)-1-36.pdf"

# OUTPUT_MD = "output_clean.md"

# ASSETS_DIR = Path("test_assets")
# ASSETS_DIR.mkdir(exist_ok=True)

# # -------------------------------------------------------------
# # Open PDF
# # -------------------------------------------------------------

# doc = fitz.open(PDF_PATH)

# print("=" * 80)
# print("Generating Markdown...")
# print("=" * 80)

# markdown = pymupdf4llm.to_markdown(
#     doc,
#     write_images=False
# )

# # -------------------------------------------------------------
# # Replace Picture Text Blocks with Image Placeholder
# # -------------------------------------------------------------

# pattern = (
#     r'<!-- Start of picture text -->'
#     r'.*?'
#     r'<!-- End of picture text -->'
# )

# matches = re.findall(pattern, markdown, flags=re.DOTALL)

# print(f"\nFound {len(matches)} picture-text block(s).")

# markdown = re.sub(
#     pattern,
#     "\n\n<!-- image -->\n\n",
#     markdown,
#     flags=re.DOTALL
# )

# # -------------------------------------------------------------
# # Save Markdown
# # -------------------------------------------------------------

# with open(OUTPUT_MD, "w", encoding="utf-8") as f:
#     f.write(markdown)

# print(f"\nMarkdown saved as: {OUTPUT_MD}")

# # -------------------------------------------------------------
# # Extract Original Embedded Images
# # -------------------------------------------------------------

# print("\n" + "=" * 80)
# print("Extracting Embedded Images")
# print("=" * 80)

# total_images = 0

# for page_no in range(len(doc)):

#     page = doc[page_no]

#     image_infos = page.get_image_info(xrefs=True)

#     if not image_infos:
#         continue

#     print(f"\nPage {page_no + 1}")

#     for image_index, info in enumerate(image_infos, start=1):

#         xref = info.get("xref")

#         if not xref:
#             continue

#         image = doc.extract_image(xref)

#         image_bytes = image["image"]
#         image_ext = image["ext"]

#         filename = (
#             ASSETS_DIR /
#             f"page_{page_no+1:04d}_image_{image_index:03d}.{image_ext}"
#         )

#         with open(filename, "wb") as img:
#             img.write(image_bytes)

#         bbox = info.get("bbox")

#         print(
#             f"Saved: {filename.name}"
#             f" | bbox={bbox}"
#         )

#         total_images += 1

# doc.close()

# print("\n" + "=" * 80)
# print("Finished")
# print("=" * 80)

# print(f"Total Images Extracted : {total_images}")
# print(f"Assets Folder          : {ASSETS_DIR.resolve()}")
# print(f"Markdown              : {OUTPUT_MD}")

import pymupdf4llm

pdf_path = r"C:\Users\Vaishnav M\Projects\PDFtoMarkdown_V3\srs_example_2010_group2 (1)-1-36.pdf"

md = pymupdf4llm.to_markdown(pdf_path)

print(type(md))
print(md[:3000])


