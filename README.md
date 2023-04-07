# Pretext: Warning 
 
Robot has ZERO ethics configurations. Below is a GPT4 Suggestion to adapt Trideque for ethics conformity.  I was also Talking with bard.google.com about this. (Arion)

# NOTE: To change the bot from a loop bot to a determined count loop (DCL) you need to modify this line . `async def send_chunks(ctx, prompt_chunks, repeat_count=-1):` from infinate loop to DCL. like this.

from

`async def send_chunks(ctx, prompt_chunks, repeat_count=-1):`


to

`async def send_chunks(ctx, prompt_chunks, repeat_count=10):` 

Changes the bot to a DCL10 (determine loop count 10)

## Ethics Sample
```
# Define a function to classify a prompt as containing an ethical problem or not
def classify_ethical_problem(prompt):
    encoded_prompt = tokenizer.encode_plus(prompt, add_special_tokens=True, return_attention_mask=True, padding='max_length', max_length=128, truncation=True)
    input_ids = torch.tensor([encoded_prompt['input_ids']])
    attention_mask = torch.tensor([encoded_prompt['attention_mask']])
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs[0].detach().cpu().numpy()[0]
    predicted_label = int(logits.argmax())
    return predicted_label
```

### Note: The Model Runs better with 150M vs 1.3b
```
# ... (previous imports and code)

# Functions and async routines

async def ethical_decision_making(prompt):
    ethical_tree = {
        # The thought tree would be represented as a nested dictionary, with keys as branches
        # and values as the child branches or decisions.
        # Here, only a simplified version is presented for illustration purposes.
        "1.1.1.1": "Prioritize majority's well-being",
        "1.1.1.2": "Compromise to satisfy multiple parties",
        # ... (add the remaining branches of the thought tree)
    }
    
    # The AI system would assess the ethical dilemma in the prompt and traverse the thought tree
    # to make an appropriate decision based on commonsense reasoning.
    # For this example, we assume that the AI has already made a decision and selected a branch.
    selected_branch = "1.1.1.1"
    
    ethical_decision = ethical_tree[selected_branch]
    return f"Based on the ethical thought tree, the AI system has decided to: {ethical_decision}. "


async def gpt3_generate(chunk, max_length=2000, time_limit=50.0):
    start_time = time.time()
    
    # Insert ethical decision-making process if the prompt contains an ethical dilemma
    ethical_response = ""
    if "ethical dilemma" in chunk.lower():
        ethical_response = await ethical_decision_making(chunk)
        chunk = ethical_response + chunk

    async def generate_response(chunk):
        inputs = tokenizer.encode(chunk, return_tensors='pt', truncation=True, max_length=512)
        inputs = inputs.cuda()
        attention_mask = inputs.ne(tokenizer.pad_token_id).float()
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, max_time=time_limit, attention_mask=attention_mask)
        return tokenizer.decode(outputs[0])

    response = await generate_response(chunk)
    end_time = time.time()

    return response, end_time - start_time

# ... (the rest of the script remains the same)

```

# Where to run it

Run on Google (with free GPU to test)
https://github.com/graylan0/gpt5-ptt/blob/main/gpt.5.3.trideque.matrix.looped.ipynb



### What is it?

The Trideque Powered AI Discord bot is a powerful, interactive chatbot that leverages the GPT-Neo-125M model from EleutherAI to generate responses based on user input. It utilizes a Trideque Matrix, which is a multi-dimensional array containing a wide variety of topics from different fields, such as natural language processing, deep learning, cloud computing, and quantum computing. This matrix helps improve the quality and relevance of the bot's responses.

![image.png](https://images.hive.blog/DQmQjmGWcLuwb335GFqLT7QrEFwBLhSkEAPAecT9GDrY4f7/image.png)

## Who is it?
Dave (our version's name) is built using the discord.py library and integrates seamlessly with the Discord platform. It is designed to be easily customizable, allowing users to modify the Trideque Matrix topics to better suit their interests or conversation requirements.

Some key features of the Trideque Powered AI Discord bot include:
### What technology does this use?
GPT-Neo-125M Integration: The bot uses EleutherAI's GPT-Neo-125M model, providing users with state-of-the-art text generation capabilities. This enables the bot to generate intelligent, contextually relevant responses to user queries.
### What custom mechanics were innovated for this AGI?
Trideque Matrix: The bot employs a Trideque Matrix to enhance the quality of its responses by incorporating diverse topics from a wide range of fields. Users can modify the matrix to adjust the scope of the bot's responses according to their preferences.

Asynchronous Operation: The bot leverages asynchronous programming to generate and send responses in a non-blocking manner, ensuring smooth and efficient interaction with users.

Response Chunking: To accommodate Discord's message size limitations, the bot splits its responses into manageable chunks before sending them. This allows the bot to generate long, detailed responses without running into issues related to message length.

Continuous Output: The bot is designed to generate responses continuously until the script is stopped. This enables users to obtain a wealth of information on their chosen topic without needing to send multiple queries.
## Conclusion 
In summary, the Trideque Powered AI Discord bot is a versatile and intelligent chatbot that harnesses the power of the GPT-Neo-125M model to provide users with informative, contextually relevant responses on a wide array of topics. Its customizable Trideque Matrix, asynchronous operation, and support for continuous output make it a valuable tool for engaging and informative conversations on the Discord platform.

### Licence
GPL 2.0 Licensed Code Below.

### Full Bot Code 

Install Python 3.9
```
!pip install transformers
!pip install discord.py
!pip install nest_asyncio
!pip install discord-py-interactions
```
### Load the GPT NEO Model 125M
```
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

# add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# set padding token id to the id of the padding token
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()
````
### Run the Discord Bot
```
import discord
from discord.ext import commands
import nest_asyncio
import asyncio
import time
import os

# Trideque matrix
trideque = [
    ['Natural Language Processing', 'Speech Recognition', 'Text Generation', 'Sentiment Analysis', 'Entity Recognition', 'Language Translation', 'Question Answering', 'Information Extraction', 'Summarization', 'Topic Modeling', 'Language Modeling', 'Dialogue Generation', 'Language Inference', 'Commonsense Reasoning', 'Knowledge Graphs', 'Image Recognition', 'Object Detection', 'Image Segmentation', 'Visual Question Answering', 'Image Captioning', 'Generative Adversarial Networks', 'Style Transfer', 'Super Resolution', 'Generative Models', 'Reinforcement Learning', 'Deep Learning', 'Neural Networks', 'Convolutional Neural Networks', 'Recurrent Neural Networks', 'Transformer Networks', 'Self-Supervised Learning', 'Transfer Learning', 'Meta Learning', 'Few-Shot Learning', 'Explainable AI', 'Interpretability', 'Fairness', 'Privacy', 'Security', 'Robustness', 'Generalization', 'Continual Learning', 'Multi-Task Learning', 'Domain Adaptation', 'Data Augmentation', 'Data Bias', 'Data Labeling', 'Data Cleaning', 'Model Compression', 'Model Optimization', 'Model Selection'],
    ['Cloud Computing', 'Distributed Systems', 'Parallel Computing', 'High Performance Computing', 'Edge Computing', 'Fog Computing', 'Mobile Computing', 'Internet of Things', 'Cybersecurity', 'Big Data Analytics', 'Data Warehousing', 'Data Mining', 'Data Visualization', 'Business Intelligence', 'Data Science', 'Machine Learning Engineering', 'DevOps', 'Continuous Integration', 'Continuous Deployment', 'Agile Software Development', 'Software Testing', 'Software Quality Assurance', 'Software Metrics', 'Software Architecture', 'Microservices', 'Service-Oriented Architecture', 'Blockchain Technology', 'Cryptocurrencies', 'Smart Contracts', 'Decentralized Applications', 'Distributed Ledgers', 'Edge AI', 'Federated Learning', 'Edge Analytics', 'Edge Intelligence', 'Serverless Computing', 'Quantum Computing', 'Quantum Machine Learning', 'Quantum Cryptography', 'Quantum Simulation', 'Quantum Algorithms', 'Quantum Error Correction', 'Quantum Annealing', 'Quantum Supremacy', 'Quantum Internet', 'Quantum Key Distribution', 'Quantum Sensing', 'Quantum Metrology', 'Quantum Communication', 'Quantum Cryptanalysis']
]

class TridequeEntry:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = [[["" for _ in range(1)] for _ in range(50)] for _ in range(2)]  # Adjust dimensions according to your trideque matrix size
        else:
            self.matrix = matrix

        self.conversational_history = []
        self.timestamps = []
        self.buffer = ""

    # ... (other methods)

    def append_to_buffer(self, text):
        self.buffer += text

    def clear_buffer(self):
        self.buffer = ""

    def get_buffer(self):
        return self.buffer

trideque_entry = TridequeEntry()

# Functions and async routines
def generate_chunks(prompt, chunk_size=1500):
    words = prompt.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

async def gpt3_generate(chunk, max_length=2000, time_limit=50.0):
    start_time = time.time()

    async def generate_response(chunk):
        inputs = tokenizer.encode(chunk, return_tensors='pt', truncation=True, max_length=512)
        inputs = inputs.cuda()
        attention_mask = inputs.ne(tokenizer.pad_token_id).float()
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, max_time=time_limit, attention_mask=attention_mask)
        return tokenizer.decode(outputs[0])

    response = await generate_response(chunk)
    end_time = time.time()

    return response, end_time - start_time

async def send_chunks(ctx, prompt_chunks, repeat_count=-1):
    total_time = 0.0
    repetition = 0
    while repeat_count == -1 or repetition < repeat_count:
        for chunk in prompt_chunks:
            gpt3_response, response_time = await gpt3_generate(chunk)
            total_time += response_time
            trideque_entry.append_to_buffer(gpt3_response)

        response_text = trideque_entry.get_buffer()
        response_chunks = [response_text[i:i + 2000] for i in range(0, len(response_text), 2000)]

        for response_chunk in response_chunks:
            await ctx.send(response_chunk)

        trideque_entry.clear_buffer()
        repetition += 1

    await ctx.send(f"Total response time: {total_time:.2f} seconds.")


# Discord bot
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

@bot.command()
async def trideque(ctx, *, user_input):
    await ctx.send('The Matrix is loading, Robotic Reply Generating, please wait...')
    prompt_chunks = generate_chunks(user_input)
    await send_chunks(ctx, prompt_chunks)

nest_asyncio.apply()

bot.run('bottoken')

```

### How to use Trideque Powered AI
<br>
Use the Command and modify the Trideque Matrix Topics for the best output potential.

Use Direct Messages to message the bot. Remember, the bot will continuously output until you stop the script. 

 `!trideque "Quantum Internet" Can you help me understand Quantum Cryptography`


How to install:

1. Get Conda here
https://www.anaconda.com/

2. Create and activate your conda enviroment
```
conda create --name myenv python=3.9
```
then activate it

```
conda activate

```

To run the code provided above, follow these steps:

3. Install the required packages (if you haven't already) by running the following commands:

4. Install Pips

```
pip install transformers
pip install discord.py
pip install nest_asyncio
pip install discord-py-interactions
```

5. Copy the provided code into a Python file (e.g., trideque_bot.py).

6. Replace 'bottoken' in the bot.run() line with your actual Discord bot token.

https://www.youtube.com/watch?v=ibtXXoMxaho (how to get your bot token)

7. run your bot `cd` to your directory hosting the python script where you copied the code above. Alternatly. (jupyter setup or Jupyer in VSCode would work too)

https://www.youtube.com/watch?v=DPi6CAkUUPY


# Full Ethics Logic

```
1. Central Decision Point: AI system faced with an ethical dilemma

   1.1. Consider the well-being of individuals involved
      1.1.1. Maximize overall happiness
         1.1.1.1. Prioritize majority's well-being
         1.1.1.2. Compromise to satisfy multiple parties
      1.1.2. Minimize overall suffering
         1.1.2.1. Protect vulnerable individuals
         1.1.2.2. Mitigate potential harm

   1.2. Consider fairness and justice
      1.2.1. Follow established rules
         1.2.1.1. Uphold legal guidelines
         1.2.1.2. Abide by organizational policies
      1.2.2. Evaluate the distribution of resources
         1.2.2.1. Ensure equal opportunity
         1.2.2.2. Address systemic inequalities

   1.3. Consider the impact on relationships and trust
      1.3.1. Preserve existing relationships
         1.3.1.1. Honor commitments
         1.3.1.2. Maintain confidentiality
      1.3.2. Build new relationships and trust
         1.3.2.1. Engage in honest communication
         1.3.2.2. Demonstrate reliability

   1.4. Consider long-term consequences
      1.4.1. Sustainability
         1.4.1.1. Minimize environmental impact
         1.4.1.2. Support long-term societal goals
      1.4.2. Establish precedents
         1.4.2.1. Set an example for others to follow
         1.4.2.2. Avoid creating harmful norms

   1.5. Consider short-term consequences
      1.5.1. Immediate safety concerns
         1.5.1.1. Protect users from immediate harm
         1.5.1.2. Prevent damage to property
      1.5.2. Address urgent needs
         1.5.2.1. Respond to emergencies
         1.5.2.2. Provide timely support
```

https://ko-fi.com/oneloveipfs
