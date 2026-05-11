"""
main.py — CLI entry point for Smart Router.

Usage:
    python main.py --prompt "What is recursion?"
    python main.py --prompt "Write a binary search in Python"
    python main.py --prompt "Explain the CAP theorem" --model llama3.1:8b
    python main.py --models   # list available models
"""

import argparse
import sys

from router.engine import RouterEngine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart Router — directs prompts to the right Ollama model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --prompt "What is recursion?"
  python main.py --prompt "Write a binary search in Python"
  python main.py --prompt "Explain the CAP theorem in depth"
  python main.py --prompt "..." --model llama3.1:8b   (override routing)
  python main.py --models
        """,
    )
    parser.add_argument("--prompt", help="The prompt to route and run")
    parser.add_argument("--model", help="Override routing — always use this model")
    parser.add_argument("--models", action="store_true", help="List available models")
    return parser.parse_args()


def main():
    args = parse_args()
    engine = RouterEngine(model_override=args.model)

    if args.models:
        print("\nRegistered Models\n" + "-" * 50)
        for m in engine.list_models():
            print(f"  {m['name']:<25} [{m['speed']}]  {m['description']}")
        return

    if not args.prompt:
        print("Error: --prompt is required. Use --help for usage.")
        sys.exit(1)

    print(f"\nSmart Router")
    print(f"  Prompt: {args.prompt[:80]}{'...' if len(args.prompt) > 80 else ''}")
    print("-" * 50)

    result = engine.route(args.prompt)

    print(f"\n{result}")


if __name__ == "__main__":
    main()
