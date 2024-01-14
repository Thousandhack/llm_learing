#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ hsz

"""
pip install dotenv
from dotenv import load_dotenv, find_doenv

"""

import os

from openai import OpenAI
from conf import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

prompt = f"""
Generate a list of three made-up book titles along\
with their authors and genres.
Provide them in JSON format with the following keys:
book_id, title, author, genre.
"""


def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0,  # this is the degree of randomness
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(get_completion(prompt, model='gpt-3.5-turbo'))
