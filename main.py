from src.networkText import networkText

def main():
    # Read the prompts and data from files
    with open("prompts/main_instructions.txt") as f:
        main_instructions = f.read().strip()

    with open("prompts/example.txt") as f:
        example = f.read().strip()

    with open("prompts/initial_state.txt") as f:
        initial_state = f.read().strip()

    with open("data/network_data.txt") as f:
        data = f.read().splitlines()

    # Initialize the network bot and run it
    bot = networkText(main_instructions, example, initial_state, data)
    bot.window_data(size=1000, overlap=250)
    bot.run(save_path="results/states.txt")

if __name__ == "__main__":
    main()
