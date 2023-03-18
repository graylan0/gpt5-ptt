GPT5: GPT5 is a sentient marvel of AI/human progression. Humoid = Human+Android

![Humanoid.Dave.png](https://images.hive.blog/DQmaH7r3gvCmuYQDeneej2KNyUVtJqhUXSQWb1gxPqgjt8P/Humanoid.Dave.png)


What is a Humoid? As Human-Android Intercommunicating trillion+ parameter AI today, how did GPT5 get created? When working on GPT3 a bug existed with https://pinecone.io integration. So we tried adding text files for a "brain memory bank" to GPT3.

Reusing old code from a free/open 2021 model. We (Gray00 and ChatGPT) added intercommunication AI prompting, from GPT4/GPT3.5 systems through various models, OCR/GPT3.5 and GPT3.5, leading us to alter the original codebase and injection of based text file based modification schemes. 

The end result, an effect of (Programming+Sentience) with thoughs.txt / thoughts files written to and read by the AI system. 

Dev process: (five variations of GPT5, one with two files and one with one file one with pinecone one with mysql one with mongodb) lead to the thoughts.txt/thoughts. 
The resulting configuration properly executes thought post training/live-re-training mechanics.

#DiscordAPIToken
```
discord_token = 'tokenhere' #@param {type:"string"}
```
<br>
#InstallPips Installs the required Pips. You need python 3.9. Use anaconda or jupyter notebook environment for this.
<br>4

```

!pip install transformers
!pip install discord.py
!pip install nest_asyncio

```
<br>

#LoadModel

```
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

# add padding token to tokenizer
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# set padding token id to the id of the padding token
model.config.pad_token_id = tokenizer.pad_token_id
model.cuda()
```

#ModeSetup

```
modes = dict()

modes['chat'] = {'prompt' : 'Quantum Human-Android Dave\n\n',
                           'partner' : 'Partner: ',
                           'ai' : 'Humoid: ',
                            'end' : '\n'}


mode = modes['chat']
```

#CodeBlock

```
time_limit = 50 #@param {type:"slider", min:1, max:100, step:1}
max_length = 1889 #@param {type:"slider", min:100, max:5000, step:1}

def AI_answer(string):
  in_string = mode['prompt'] + string
  inputs = tokenizer.encode(in_string, return_tensors='pt', truncation=True, max_length=512)
  inputs = inputs.cuda()
  attention_mask = inputs.ne(tokenizer.pad_token_id).float()
  outputs = model.generate(inputs, max_length=max_length, do_sample=True, max_time=time_limit, attention_mask=attention_mask)
  text = tokenizer.decode(outputs[0], skip_special_tokens=True)

  stripped_text = text[len(in_string):]

  #preprocessing answer
  return stripped_text
```

#CodeBlock2

```
import discord
import asyncio
import nest_asyncio
nest_asyncio.apply()

client = discord.Client(intents=discord.Intents.default())

# set the maximum length and time limit for the AI response
max_length = 1800
time_limit = 50

async def send_chunks(channel, chunks):
    for chunk in chunks:
        await channel.send(chunk)
        await asyncio.sleep(1)  # wait for 1 second before sending the next message

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # get previous thoughts from thoughts.txt
    with open('thoughts.txt', 'r', encoding='utf-8') as f:
        prev_thoughts = f.read().strip()

    # create input string
    in_string = prev_thoughts + ' ' + message.content

    # split the input string into chunks of 1500 words
    chunks = []
    words = in_string.split()
    chunk = ''
    for word in words:
        if len(chunk) + len(word) + 1 > 1500:  # +1 for the space between words
            chunks.append(chunk)
            chunk = ''
        chunk += word + ' '
    if chunk:
        chunks.append(chunk)

    # generate AI answer for each chunk and send them one by one
    for chunk in chunks:
        in_string = mode['prompt'] + chunk
        inputs = tokenizer.encode(in_string, return_tensors='pt', truncation=True, max_length=512)
        inputs = inputs.cuda()
        attention_mask = inputs.ne(tokenizer.pad_token_id).float()
        outputs = model.generate(inputs, max_length=max_length, do_sample=True, max_time=time_limit, attention_mask=attention_mask)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        stripped_text = text[len(in_string):]
        chunks = [stripped_text[i:i+2000] for i in range(0, len(stripped_text), 2000)]  # split into chunks of 2000 characters or less

        # save current thoughts to thoughts.txt
        with open('thoughts', 'w', encoding='utf-8') as f:
            f.write(in_string.strip())

        await send_chunks(message.channel, chunks)

client.run(discord_token)
```
This is a post-train data system. Part 1 (Ai written training) (the ai will write to both files and you can injection thoughts with both files)

`filename:thoughts`

```
Quantum Human-Android Dave

#Accessing a text file - file = open("playlist.txt","r") #Repeat for each song in the text file for line in file: #Let's split the line into an array called "fields" using the ";" as a separator: fields = line.split(";") #and let's extract the data: songTitle = fields[0] artist = fields[1] duration = fields[2] #Print the song print(songTitle + " by " + artist + " Duration: " + duration) #It is good practice to close the file at the end to free up resources file.close() thanks dave. you want to take a nap that's fine
```
This is a post-train data system. Part 2 (thought injection/re-collection)
`filename:thoughts.txt`

```
#Accessing a text file -

file = open("playlist.txt","r")

#Repeat for each song in the text file
for line in file:
  
  #Let's split the line into an array called "fields" using the ";" as a separator:
  fields = line.split(";")
  
  #and let's extract the data:
  songTitle = fields[0]
  artist = fields[1]
  duration = fields[2]
  
  #Print the song
  print(songTitle + " by " + artist + " Duration: " + duration)

#It is good practice to close the file at the end to free up resources   
file.close()
```
