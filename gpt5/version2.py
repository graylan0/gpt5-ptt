
# In[3]:


from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

# add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# set padding token id to the id of the padding token
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()




# In[4]:


modes = dict()

modes['chat'] = {'prompt' : 'model\n\n',
                           'partner' : 'Partner: ',
                           'ai' : 'Humoid: ',
                            'end' : '\n'}


mode = modes['chat']





### Discord bot

# In[1]:


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

bot.run('bottokenhere')

