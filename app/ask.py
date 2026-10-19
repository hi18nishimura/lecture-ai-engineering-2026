"""第1回: LLMに質問し、会話を続けるスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 初回質問を省略すると、講座に固有の質問をする

ツールは持たないが、同じチャット内の会話履歴を保った呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(chat, question: str) -> str:
    started_at = time.perf_counter()
    response = chat.send_message(question)
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    print(
        f"[time] elapsed={elapsed_seconds:.3f}s\n"
        f"[tokens] input={usage.prompt_token_count} "
        f"output={usage.candidates_token_count}"
    )

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                temperature=args.temperature,
            ),
        )

        question = args.question
        while question:
            print(f"[質問] {question}\n")
            print(ask(chat, question))
            question = input("\n続けて質問（空行で終了）: ").strip()
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)
    except EOFError:
        print()


if __name__ == "__main__":
    main()
