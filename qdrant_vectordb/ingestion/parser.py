# Move your existing parse_pdf() here.
import fitz
def parse_pdf(filepath):

    try:

        doc = fitz.open(
            stream=filepath.read(),
            filetype="pdf"
        )

        full_text = ""

        pages = []

        for page_number, page in enumerate(doc):

            text = page.get_text()

            full_text += text

            page_metadata = {

                "page_number": page_number + 1,

                "links": page.get_links(),

                "image_count": len(
                    page.get_images()
                ),

                "blocks": page.get_text(
                    "blocks"
                )

            }

            pages.append(

                {

                    "text": text,

                    "metadata": page_metadata

                }

            )

        document_metadata = {}

        for key, value in doc.metadata.items():

            document_metadata[key] = value

        document_metadata["page_count"] = len(
            doc
        )

        return {

            "text": full_text,

            "pages": pages,

            "document_metadata": document_metadata

        }

    except Exception as e:

        print(e)

        return None