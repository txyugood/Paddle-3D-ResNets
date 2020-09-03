import json

with open('ucf101_json/ucf101_01.json','r') as f:
    data = json.load(f)
database = data['database']
count = 0
for k,v in database.items():
    count += v['annotations']['segment'][1] -1
print(count)