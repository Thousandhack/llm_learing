#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ hsz
import os

from openai import OpenAI
# os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = "my_open_ai_key"
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
    """
        {
      "books": [
        {
          "book_id": 1,
          "title": "The Enigma of Elysium",
          "author": "Evelyn Sinclair",
          "genre": "Mystery"
        },
        {
          "book_id": 2,
          "title": "Whispers in the Wind",
          "author": "Nathaniel Blackwood",
          "genre": "Fantasy"
        },
        {
          "book_id": 3,
          "title": "Echoes of the Past",
          "author": "Amelia Rivers",
          "genre": "Romance"
        }
      ]
    }
    """
