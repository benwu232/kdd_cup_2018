# coding: utf-8

import requests

file_name = '../submit_attn_pos/submission.csv'
files = {'files': open(file_name, 'rb')}
print('submit ', file_name)
data = {
    "user_id": "benwu232",
    # user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "7e44bd9c96630d030eb5d15838d71da878d1ac0da3607f822c98ed8f69cd4a94",  # your team_token.
    "description": 'seq2seq, attn arbitrary',  # no more than 40 chars.
    "filename": file_name,  # your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)
