import requests
url = 'https://biendata.com/competition/meteorology/bj/2018-04-01-0/2018-04-01-23/2k0d1d8'
respones= requests.get(url)
with open ("bj_meteorology_2018-04-01-0-2018-04-01-23.csv",'w') as f:
    f.write(respones.text)
