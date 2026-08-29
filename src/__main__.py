from src.service import RagService


def main() -> None:
    service = RagService()
    service.startup()
    service.run_interactive()


if __name__ == "__main__":
    main()
