from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.summarizer import summarize, generate_title
from core.rag import build_rag_chain, load_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str) -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_items = extract_action_items(transcript)
    key_decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": key_decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def start_chat(rag_chain):
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")


if __name__ == "__main__":
    mode = input("New meeting or resume last one? (new/resume): ").strip().lower()

    if mode == "resume":
        print("Loading existing vector store...")
        rag_chain = load_rag_chain()
        start_chat(rag_chain)

    else:
        source = input("give video url or video file ").strip()
        result = run_pipeline(source)

        print(f"video summary {result['title']}")
        print(f"{result['summary']}")
        print("=" * 80)
        print(f"action items from the video file :{result['action_items']}")
        print(f"key decisions from the video file :{result['key_decisions']}")
        print(f"questions from the video file :{result['open_questions']}")
        print("=" * 60)

        rag_chain = result["rag_chain"]
        start_chat(rag_chain)