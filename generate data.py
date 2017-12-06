# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:01:49 2017

@author: anu
"""
import json
import pandas as pd

data = []
with open('C:\\Users\\anu\\Downloads\\article.jsonl\\article.jsonl', 'r') as content_file:
    for line in content_file:
        data.append(json.loads(line))
        print(line)
df = pd.DataFrame(data)
df = df[df["media-type"] == "News"]
df = df.sample(15000)
df.to_csv("realnews.csv", encoding='utf-8')
    