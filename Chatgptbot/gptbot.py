# -*- coding: utf-8 -*-
"""
Created on Mon May 22 17:43:15 2023

@author: Linusyao
"""

import openai
import IPython

openai.api_key  = 'rQHb2plDyxEwl1QISZpgT3BlbkFJ6BAjS8ZmgBaljNTG8u7p'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    content = response.choices[0].message["content"]
    return content

from IPython.core.magic import (Magics,magics_class,line_magic,cell_magic,line_cell_magic)

@magics_class
class Magicsgpt(Magics):
    @line_magic
    def chat(self,line):
        return get_completion(line)
    @cell_magic
    def gpt(self,line,cell):
        antwort = get_completion(cell)
        print(antwort)
    @line_cell_magic
    def chatgpt(self,line,cell=None):
        if cell is None:
            return get_completion(line)
        else:
            print(get_completion(cell))
            
def load_ipython_extension(ipython):
    ipython.register_magics(Magicsgpt)