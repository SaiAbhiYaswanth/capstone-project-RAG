from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


class PDFLoader:

    @staticmethod
    def normalize_tables(text):
        """
        Convert table-like rows into structured bullet lists
        so retrieval models understand them better.
        """

        lines = text.split("\n")
        processed_lines = []

        for line in lines:

            words = line.split()

            # Detect possible table rows (many short words)
            if len(words) >= 4 and all(len(w) < 15 for w in words):

                formatted = " | ".join(words)
                processed_lines.append(formatted)

            else:
                processed_lines.append(line)

        return "\n".join(processed_lines)

    @staticmethod
    def load_pdf(file):

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        # -------- Added Table Processing --------
        text = PDFLoader.normalize_tables(text)

        return text

    @staticmethod
    def chunk_text(text):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = splitter.split_text(text)

        return chunks