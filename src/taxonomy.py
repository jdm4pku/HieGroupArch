import csv
import json
import os
import time
from openai import OpenAI
from logger import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

def load_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

# def write_json(data,path):
#     with open(path,'w',encoding='utf-8') as f:
#         json.dump(data,f,indent=4)

def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        f.write(json_data)

def read_csv(csv_path):
    json_data = []
    with open(csv_path,'r',encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            json_data.append(row)
    return json_data

def preprocess():
    os_name = "openEuler2309"
    csv_path = f"/home/jindm/project/HieGroupArch/data/csv/{os_name}.csv"
    json_data = read_csv(csv_path)
    name_description = []
    for item in json_data:
        name = item["name"].split(",")[0]
        description = item["description"].split(",")[0]
        name_description.append({
            "name":name,
            "descroption":description
        }
        )
    json_path = f"/home/jindm/project/HieGroupArch/data/json/{os_name}.json"
    write_json(name_description,json_path)

def get_completion(prompt):
    client = OpenAI(
        api_key = "", 
        base_url="http://15.204.101.64:4000/v1"
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-3.5-turbo",
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def gpt_inference(base_tax,item):
    prompt = f"术语表T如下:\n {base_tax}. 术语N: \n {item}. 请判断当前的术语表T是否包含了术语N。你回复的候选项: A. 不包含。 B.包含。 如果包含，请解释和当前术语表中的哪个是重复的。"
    answer = get_completion(prompt)
    return answer

def unify_taxonomy():
    base_tax = load_file("/home/jindm/project/HieGroupArch/data/json/final_taxonomy.json")
    os_name = "openEuler2309" # "fedora40","opencloudos92","openEuler2309"
    merge_tax_dir = "/home/jindm/project/HieGroupArch/data/json"
    merge_tax_path = os.path.join(merge_tax_dir,f"{os_name}.json")
    merge_tax = load_file(merge_tax_path)
    judge = []
    for item in tqdm(merge_tax):
        response = gpt_inference(base_tax,item)
        judge.append(
            {
                "name": item["name"],
                "judge":response
            }
        )
    output_path = f"/home/jindm/project/HieGroupArch/data/action/{os_name}.json"
    write_json(judge,output_path)

def main():
    preprocess()
    # unify_taxonomy()

if __name__=="__main__":
    main()