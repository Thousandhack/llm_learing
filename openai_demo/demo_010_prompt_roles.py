#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ hsz

"""
chatgpt使用多种角色进行对话
"""

from openai import OpenAI
from conf import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant'},
        {'role': 'user', 'content': 'Who won the world series in 2020?'},
        {'role': 'assistant', 'content': 'The Los Angeles Dodgers won the World Series in 2020'},
        {'role': 'user', 'content': 'Where wa it played?'},
    ],
    max_tokens=300,
    temperature=0.5
)

print(response.choices[0].message.content)
"""
The 2020 World Series was played at the Globe Life Field in Arlington, Texas.
"""