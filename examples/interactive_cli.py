import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tree_memory_engine import TreeMemory, build_demo_memory


PROMPT = "tree-memory> "


def ask_field(label, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def print_help():
    print(
        """
Commands:
  ask <question>       Answer from memory
  explain <question>   Show routing and retrieved facts
  add                  Add a new fact interactively
  update               Update a path+slot fact interactively
  alias                Add or update a path alias
  list                 List active facts
  routes <question>    Show candidate memory branches
  save <file>          Save memory to JSON
  load <file>          Load memory from JSON
  demo                 Reset to demo memory
  help                 Show this help
  quit                 Exit
""".strip()
    )


def print_answer(memory, question):
    result = memory.answer(question)
    if result["answer"] is None:
        print("No answer found.")
        return
    print(f"Answer: {result['answer']}")
    print(f"Path:   {result['path']}")
    print(f"Slot:   {result['slot']}")
    print(f"Conf:   {result['confidence']}")


def print_explanation(memory, question):
    explanation = memory.explain_retrieval(question)
    print(json.dumps(explanation, indent=2))


def add_fact(memory):
    print("Add fact")
    path = ask_field("path", "artifacts/vehicles/car_tires")
    slot = ask_field("slot", "car_tires.maker")
    answer = ask_field("answer", "Michelin")
    text = ask_field("text", f"{answer} is stored in {path}.")
    tags = ask_field("tags", path.replace("/", " "))
    fact = memory.add_fact(path=path, text=text, slot=slot, answer=answer, tags=tags)
    print(f"Added fact #{fact.id} at {fact.path}")


def update_fact(memory):
    print("Update fact")
    path = ask_field("path", "artifacts/vehicles/car_tires")
    slot = ask_field("slot", "car_tires.maker")
    answer = ask_field("new answer", "Bridgestone")
    text = ask_field("new text", f"{answer} is the updated value for {slot}.")
    tags = ask_field("tags", path.replace("/", " "))
    fact = memory.update_fact(path=path, slot=slot, text=text, answer=answer, tags=tags)
    print(f"Updated fact #{fact.id} at {fact.path}, version {fact.version}")


def add_alias(memory):
    print("Add alias")
    path = ask_field("path", "artifacts/vehicles/car_tires")
    alias = ask_field("alias", "car tires rubber road grip vehicle maker")
    memory.add_alias(path, alias)
    print(f"Alias updated for {path}")


def list_facts(memory):
    active = memory.active_facts()
    if not active:
        print("No active facts.")
        return
    for fact in active:
        print(f"#{fact.id} v{fact.version} {fact.path} [{fact.slot}] -> {fact.answer}")


def print_routes(memory, question):
    routes = memory.route(question)
    if not routes:
        print("No routes.")
        return
    for i, route in enumerate(routes, 1):
        terms = ", ".join(route.matched_terms)
        print(f"{i}. {route.path} score={route.score:.3f} terms=[{terms}]")


def main():
    memory = build_demo_memory()
    print("TreeMemory interactive CLI")
    print("Loaded demo memory. Type 'help' for commands.")

    while True:
        try:
            raw = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue

        cmd, _, arg = raw.partition(" ")
        cmd = cmd.lower()
        arg = arg.strip()

        if cmd in {"quit", "exit"}:
            break
        if cmd == "help":
            print_help()
        elif cmd == "ask":
            print_answer(memory, arg or ask_field("question"))
        elif cmd == "explain":
            print_explanation(memory, arg or ask_field("question"))
        elif cmd == "routes":
            print_routes(memory, arg or ask_field("question"))
        elif cmd == "add":
            add_fact(memory)
        elif cmd == "update":
            update_fact(memory)
        elif cmd == "alias":
            add_alias(memory)
        elif cmd == "list":
            list_facts(memory)
        elif cmd == "save":
            filename = arg or ask_field("file", "tree_memory_cli.json")
            memory.save(ROOT / filename)
            print(f"Saved {filename}")
        elif cmd == "load":
            filename = arg or ask_field("file", "tree_memory_cli.json")
            memory = TreeMemory.load(ROOT / filename)
            print(f"Loaded {filename}")
        elif cmd == "demo":
            memory = build_demo_memory()
            print("Reset to demo memory.")
        else:
            print(f"Unknown command: {cmd}. Type 'help'.")


if __name__ == "__main__":
    main()
