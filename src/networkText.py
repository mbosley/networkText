#!/usr/bin/env python3
import os
import openai
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_API_KEY')

class networkText:
    """
    A class for ingesting and processing text data using OpenAI.

    Attributes
    ----------
    main_instructions : str
        The main instructions for labeling the data.
    example : str
        An example of how to label the data.
    state : str
        The current state of the bot.
    data : list
        The list of network data to be ingested.
    verbose : bool, optional
        Whether to print detailed output to the console, by default True.

    Methods
    -------
    window_data(size, overlap)
        Window the data into smaller chunks for processing.
    assemble_full_prompt(new_info)
        Assemble a full prompt for the OpenAI API based on the current state and data window.
    get_new_state(new_prompt)
        Use the OpenAI API to generate a new state based on the current prompt.
    run()
        Run the network bot by windowing the data, generating prompts, and updating the state.
    """

    def __init__(self, main_instructions, example, initial_state, data, verbose=True):
        self.main_instructions = main_instructions # main instructions for labeling
        self.example = example # example instruction
        self.state = initial_state # starting state
        self.data = data # list of data to be ingested
        self.verbose = verbose

    def window_data(self, size=1000, overlap=250):
        if len(self.data.split()) < size:
            # If the data is smaller than the window size, just use the whole data
            self.windowed_data = [self.data]
            if self.verbose:
                print(f"Data is length ({len(self.data.split())}), less than window size ({size}), using full data.")
            return

        windows = []
        start = 0
        try:
            while start + size <= len(self.data.split()):
                window = " ".join(self.data.split()[start:start+size])
                windows.append(window)
                start += size - overlap
            if self.verbose:
                print(f"""
                \n\n------------------------------\n
                Created {len(windows)} windows of data.
                \n------------------------------\n\n"""
                )
            self.windowed_data = windows
        except Exception as e:
            print(f"Error occurred while windowing data: {e}")
            raise e

    def assemble_full_prompt(self, new_info):
        return  f"{self.main_instructions}\n\nexample:\n{self.example}\n\ncurrent_state:\n{self.state}\n\nprompt:\nupdate_graph({new_info})\n\nnew state:\n"

    def get_new_state(self, new_prompt):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=new_prompt,
            temperature=0,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].text.strip(), response.usage.total_tokens


    def run(self, save_path=None, append=True):
        token_count = 0
        counter = 0
        for window in self.windowed_data:
            prompt = self.assemble_full_prompt(window)
            if self.verbose:
                print(f"Full prompt is:\n\n{prompt}")
            self.state, tokens_used = self.get_new_state(prompt)
            if self.verbose:
                print(f"New state is:\n\n{self.state}")
            token_count += tokens_used
            print(f"Tokens used so far: {token_count}")
            if save_path:
                with open(save_path, "a" if append else "w") as f:
                    f.write(self.state + "\n")
            counter += 1
        if self.verbose:
            print(f"Processed {counter} windows of data.")
            print(f"Total number of tokens used: {token_count}")
