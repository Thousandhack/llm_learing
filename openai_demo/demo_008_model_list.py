#!/usr/bin/env python
# -*- coding:utf-8 -*-
# __author__ hsz

"""
获取目前所有open ai的所有模型信息

"""
from openai import OpenAI
from conf import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
model_list = client.models.list()
# print(model_list)
for model_info in model_list.data:
    info = {
        'id': model_info.id,
        'object': model_info.object,
        'crated': model_info.created,
        'owned_by': model_info.owned_by
    }
    print(info)
