import os                          # for file handling and environment variables
import sys
import fitz                        # pymupdf - for PDF text extraction
from groq import Groq
from dotenv import load_dotenv     # to load keys from .env file

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# PDF extraction function

def extract_text_from_pdf(pdf_path: str) -> str:
    """Pull all text out of a PDF, page by page."""
    # Check if the file exists before attempting to open it
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"No PDF found at: {pdf_path}")

    # Open the PDF document using PyMuPDF
    doc = fitz.open(pdf_path)
    pages = []

    # Iterate through each page and extract its text content
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():                          # skip blank pages
            # Format each page's text with a page number header
            pages.append(f"[Page {page_num}]\n{text}")

    # Release the document resource
    doc.close()

    # Ensure at least some text was extracted; otherwise, it may be a scanned PDF
    if not pages:
        raise ValueError("Could not extract any text. PDF might be scanned/image-based.")

    # Join all pages with double newlines as separators
    return "\n\n".join(pages)


# Token safety check (rough, based on character count)

def is_too_long(text: str, max_chars: int = 60_000) -> bool:
    """
    Rough guard: 1 token ≈ 4 chars.
    llama-3.3-70b context = ~128k tokens → ~512k chars.
    60k chars ≈ 15k tokens, well within limits.
    Increase max_chars for larger docs, or use RAG for very large ones.
    """
    return len(text) > max_chars


#Build the system prompt with clear instructions and the document text

def build_system_prompt(document_text: str) -> str:
    return f"""You are a document assistant. Your job is to answer questions
strictly based on the document provided below.

Rules:
- Only use information from the document to answer
- If the answer isn't in the document, say "I couldn't find that in the document"
- When possible, mention which page the information came from
- Be concise and direct

DOCUMENT:
────────────────────────────────────────
{document_text}
────────────────────────────────────────
"""


# Conversation loop for Q&A with the document, maintaining history and showing token usage

def run_qa_session(system_prompt: str):
    """
    Interactive Q&A loop that maintains conversation history and displays token usage.

    Args:
        system_prompt: The system prompt containing document context and instructions
    """
    # Store the conversation history to maintain context across multiple questions
    conversation_history = []

    print("\nDocument loaded. Ask anything about it. Type 'quit' to exit.\n")

    while True:
        # Get user input and remove leading/trailing whitespace
        user_input = input("You: ").strip()

        # Skip empty inputs
        if not user_input:
            continue
        # Exit the session if user types 'quit'
        if user_input.lower() == "quit":
            print("Bye!")
            break

        # Add user's question to conversation history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Send request to Groq API with system prompt and full conversation history
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation_history  # Unpack conversation history
            ],
            temperature=0.2,        # lower = more factual, less creative
            max_tokens=1024,        # limit response length
        )

        # Extract the assistant's answer from the API response
        answer = response.choices[0].message.content

        # Add assistant's response to conversation history for context
        conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        # Display the answer along with token usage statistics
        usage = response.usage
        print(f"\nAssistant: {answer}")
        print(f"  [tokens — prompt: {usage.prompt_tokens} | "
              f"reply: {usage.completion_tokens} | "
              f"total: {usage.total_tokens}]\n")


# Entry point: handle PDF input, extract text, build prompt, and start Q&A session

def main():
    # Accept PDF path as argument or prompt for it
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("Path to PDF: ").strip()

    print(f"\nLoading: {pdf_path}")

    try:
        text = extract_text_from_pdf(pdf_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Extracted {len(text):,} characters across the document.")

    if is_too_long(text):
        print("\nWarning: document is large. Consider using a smaller PDF for now.")

    system_prompt = build_system_prompt(text)
    run_qa_session(system_prompt)


if __name__ == "__main__":
    main()