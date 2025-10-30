from src.search import search_prompt


def main():
    print("=" * 60)
    print("Sistema RAG - Chat com Documentos")
    print("=" * 60)
    print("Digite 'sair', 'quit' ou 'exit' para encerrar o chat.\n")

    while True:
        try:
            question = input("Você: ").strip()

            if question.lower() in ["sair", "quit", "exit"]:
                print("\nEncerrando o chat. Até logo!")
                break

            if not question:
                print("Por favor, digite uma pergunta.\n")
                continue

            print("\nProcessando...\n")
            response = search_prompt(question)

            print(f"Assistente: {response}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nChat interrompido. Até logo!")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}")
            print("Tente novamente.\n")


if __name__ == "__main__":
    main()
