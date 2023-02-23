from src.networkText import networkText

def main():
    # Read the prompts and data from files
    with open("prompts/main_instructions.txt") as f:
        main_instructions = f.read().strip()

    with open("prompts/example.txt") as f:
        example = f.read().strip()

    with open("prompts/initial_state.txt") as f:
        initial_state = f.read().strip()

    with open("data/test_short.txt") as f:
        data = f.read().strip()

    # Initialize the network bot and run it
    bot = networkText(main_instructions, example, initial_state, data)
    bot.window_data(size=300, overlap=50)
    bot.run(save_path="results/states.txt")

if __name__ == "__main__":
    main()
