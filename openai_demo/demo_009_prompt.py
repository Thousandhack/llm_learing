#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ hsz

from openai import OpenAI
from conf import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

response = client.chat.completions.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user',
         'content': '讲两个冷笑话'

         }
    ],
    max_tokens=300,
    temperature=0.5
)

print(response.choices[0].message.content)
