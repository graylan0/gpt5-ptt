
# In[ ]:


discord_token = 'tokenhere' #@param {type:"string"}


# ##Module installation
# this will install all the necessary modules

# In[ ]:




# ##Download and load GPT NEO model.
# It will take a little bit

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



# In[5]:



# ##Discord bot

# In[1]:



import discord
from discord.ext import commands
import nest_asyncio
import asyncio
import time
import os

intents = discord.Intents.default()
intents.typing = False
intents.presences = False

bot = commands.Bot(command_prefix='!', intents=intents)

class TridequeEntry:
    def __init__(self, matrix=None):
        if matrix is None:
            self.matrix = [["" for _ in range(13)] for _ in range(13)]
        else:
            self.matrix = matrix

    def update_entry(self, x, y, value):
        self.matrix[x][y] = value

    def get_value(self, x, y):
        return self.matrix[x][y]

trideque_entry = TridequeEntry()

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


async def write_to_buffer_file(response_text):
    buffer_file = "buffer.txt"
    with open(buffer_file, "w") as file:
        file.write(response_text)
    return buffer_file

async def read_from_buffer_file(buffer_file, chunk_size=1800):
    with open(buffer_file, "r") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk
    os.remove(buffer_file)
    
    
async def send_chunks(ctx, prompt_chunks, repeat_count=-1):
    total_time = 0.0
    repetition = 0
    while repeat_count == -1 or repetition < repeat_count:
        for chunk in prompt_chunks:
            gpt3_response, response_time = await gpt3_generate(chunk)
            total_time += response_time

            buffer_file = await write_to_buffer_file(gpt3_response)

            async for response_part in read_from_buffer_file(buffer_file):
                await asyncio.sleep(0.5)
                await ctx.send(response_part)  # Change this line to send multiple messages

        repetition += 1

    await ctx.send(f"Total response time: {total_time:.2f} seconds.")




@bot.event
async def on_ready():
        print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')

@bot.command()
async def trideque(ctx, *, user_input):
    await ctx.send('The Matrix is loading, Robotic Reply Generating, please wait...')
    prompt_chunks = generate_chunks(user_input)
    await send_chunks(ctx, prompt_chunks)

nest_asyncio.apply()

# Replace 'your_bot_token' with your actual bot token from the Discord Developer Portal
bot.run('tokenhere')


# In[ ]:




