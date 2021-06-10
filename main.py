!pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
!pip install transformers

from transformers import pipeline # First line

#setting up the GPT generator
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B') # Second line

# Generate Text using Prompt
prompt = "The current stock market" # Third line
res = generator(prompt, max_length=50, do_sample=True, temperature=0.9) # Fourth line

#print
print(res[0]['generated_text'])

#save to txt file
with open('gpttext.txt', 'w') as f:
    f.writelines(res[0]['generated_text'])
