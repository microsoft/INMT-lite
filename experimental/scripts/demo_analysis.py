import json 
import numpy as np
analysis_path = '/home/t-hdiddee/ACL/user-study/quiz.dsv'
with open(analysis_path,'r') as file: 
    records = file.read().split('\n')
print(f'{len(records)} are the number of records being analysed.')
hours = [] 
for record in records:
    try: 
        wid, q, r = record.split('$')
        if q == 'आप कितने समय से हिंदी कीबोर्ड का उपयोग कर रहे हैं?':
            r = r.replace('""','"')
            r = r.strip('"')
            hours.append(json.loads(r))
    except: 
        print(q)

# print(hours)
never, lessthan15, lessthan2, morethan15 = 0,0,0,0
for hour in hours:
    if hour['keyboard'][0] == 'कभी नहीं':
        never += 1 
    if hour['keyboard'][0] == '15 घंटे से कम':
        lessthan15 += 1 
    if hour['keyboard'][0] == '2 घंटे से कम':
        lessthan2 += 1 
    if hour['keyboard'][0] == '24 घंटे':
        morethan15 += 1 

print(never, lessthan2, lessthan15, morethan15)


        