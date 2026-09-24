from __future__ import annotations

import argparse

from silico import make_llm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="?", default="Say hello in three words.")
    parser.add_argument("--model", default="mock")
    args = parser.parse_args()
    result = make_llm(args.model).respond_with_metadata(args.message)
    print(f"model: {args.model}\nreply: {result.text!r}")
    print(f"metadata: {result.response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
