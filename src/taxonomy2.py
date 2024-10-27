import os
import json
import time
import pandas as pd
from tqdm import tqdm
from zhipuai import ZhipuAI
from openai import OpenAI
from logger import get_logger
import pandas as pd

logger = get_logger(__name__)

def load_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def write_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        f.write(json_data)

def translate_with_glm(text):
    client = ZhipuAI(api_key="55ccab88d67133915e00f4500f86ab6f.dVXaO927Yn0NNhBl") # 填写您自己的APIKey
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                model="glm-4-plus",  # 填写需要调用的模型编码
                messages=[
                    {"role": "system", "content": "你是一个英文翻译中文的专家"},
                    {"role": "user", "content": f"将下面的句子翻译成中文: {text}"}
                ],
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def covert_excel_data(data,origin_path,arch):
    json_data = load_file(origin_path)
    for item in tqdm(json_data):
        name = item["name"]
        desc_en = item["descroption"]
        desc_zh = translate_with_glm(desc_en)
        data["发行版"].append(arch)
        data["分组名"].append(name)
        data["分组英文描述"].append(desc_en)
        data["分组中文描述"].append(desc_zh)
    return data

def process1():
    origin_dir = "/home/jindm/project/HieGroupArch/data/json"
    arch_list = ['centos7','fedora40','anolis89','opencloudos92','openEuler2309']
    # arch_list = ['openEuler2309']
    data = {
        "发行版":[],
        "分组名":[],
        "分组英文描述":[],
        "分组中文描述": []
    }
    for arch in arch_list:
        origin_path = os.path.join(origin_dir,f"{arch}.json")
        data = covert_excel_data(data,origin_path,arch)
    write_path = "/home/jindm/project/HieGroupArch/taxonomy2/all_group.xlsx"
    df = pd.DataFrame(data)
    df.to_excel(write_path, index=False)

def gpt_completion(prompt):
    client = OpenAI(
        api_key = "", # 替换成你的api-key
    )
    flag = False
    while not flag:
        try:
            response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o-mini", # 这里切换你想要使用的那模型
                    temperature=0   
            )
            flag=True
        except Exception as e:
            print(e)
            time.sleep(0.5)
    return response.choices[0].message.content

def gpt_inference(taxonomy,item):
    prompt = f"""
    当前的术语表T：{taxonomy} \n\n
    需要判断的术语A: {item} \n\n
    请你判断术语A是否已经存在术语表T中.如果存在,先输出1,然后解释和术语表中的哪个重复。如果不存在，直接输出0。请注意可能存在术语在描述上略有差异，但是表达的是同一个意思的情况，这种认定是同一个。
    """
    response = gpt_completion(prompt)
    # response = glm_completion(prompt)
    logger.info(response)
    if '1' in response:
        return response, taxonomy
    elif '0' in response:
        taxonomy.append(item)
        return response,taxonomy

def process2():
    taxnomy = []
    origin_dir = "/home/jindm/project/HieGroupArch/data/json"
    arch_list = ['centos7','fedora40','anolis89','opencloudos92','openEuler2309']
    label_result = {
        "name":[],
        "label": [],
    }
    for arch in arch_list:
        origin_path = os.path.join(origin_dir,f"{arch}.json")
        origin_data = load_file(origin_path)
        for item in tqdm(origin_data):
            response, taxnomy  = gpt_inference(taxnomy,item)
            label_result['name'].append(item["name"])
            label_result['label'].append(response)
    output_path = f"/home/jindm/project/HieGroupArch/taxonomy2/gpt4o_mini_label.xlsx"
    taxnomy_path = f"/home/jindm/project/HieGroupArch/taxonomy2/gpt4o_mini_taxo.json"
    df = pd.DataFrame(label_result)
    df.to_excel(output_path, index=False)
    write_json(taxnomy,taxnomy_path)

def process3():
    """
    生成最终的术语表
    """
    file_path = "/home/jindm/project/HieGroupArch/taxonomy2/label_merge.xlsx"
    save_path = "/home/jindm/project/HieGroupArch/taxonomy2/taxo_merge.json"
    df = pd.read_excel(file_path)
    taxo = []
    for index, row in df.iterrows():
        if row["最终判定"] == 0:
            group_name = row["分组名"]
            group_desc = row["分组英文描述"]
            if pd.isna(group_desc):
                group_desc = group_name
            item = {
                "name": group_name,
                "description": group_desc
            }
            taxo.append(item)
    sorted_taxo = sorted(taxo, key=lambda x: x["name"])
    write_json(sorted_taxo,save_path)

def process4():
    pass

def check_in(name_list):
    origin_path = "/home/jindm/project/HieGroupArch/cluster/1_layer.json"
    data = load_file(origin_path)
    candidate_list = []
    for item in data:
        candidate_list.append(item["name"])
    for name in name_list:
        if name not in candidate_list:
            return 0
    return 1

def process5(): 
    """处理聚类结果"""
    origin_path = "/home/jindm/project/HieGroupArch/数据整理/第一层聚类(待整理).txt"
    with open(origin_path, 'r', encoding='utf-8') as file:
        data = file.read()
    sections = data.split("==================")
    save_result = {
        "name": [],
        "desc": [],
        "children":[],
        "exist":[]
    }
    for section in sections:
        subfea_start = section.find("input:")
        subfea_end = section.find("10", subfea_start + 1)
        substring_1 = section[subfea_start+6:subfea_end]
        lines = substring_1.splitlines()
        subfea_name_list = [line.split(':')[0].strip() for line in lines if line != ""]
        check_result = check_in(subfea_name_list)
        cluster_start = section.find("cluster:")
        cluster_end = section.find("10", cluster_start)
        substring_2 = section[cluster_start+8:cluster_end]
        cluster_name = substring_2.split(':')[0].strip()
        print("============")
        print(cluster_name)
        cluster_desc = substring_2.split(':')[1].strip()
        save_result["name"].append(cluster_name)
        save_result["desc"].append(cluster_desc)
        save_result["children"].append(subfea_name_list)
        save_result["exist"].append(check_result)
    df = pd.DataFrame(save_result)
    output_path = "/home/jindm/project/HieGroupArch/cluster/1_layer.xlsx"
    df.to_excel(output_path, index=False)

def process6():
    file_path = "/home/jindm/project/HieGroupArch/cluster/1_layer.xlsx"
    save_path = "/home/jindm/project/HieGroupArch/cluster/1_layer.json"
    df = pd.read_excel(file_path)
    taxo = []
    for index, row in df.iterrows():
            item = {
                "name": row["name"],
                "description": row["desc"],
                "children":row["children"]
            }
            taxo.append(item)
    sorted_taxo = sorted(taxo, key=lambda x: x["name"])
    write_json(sorted_taxo,save_path)

def process7():
    """处理第二层的聚类结果"""
    origin_path = "/home/jindm/project/HieGroupArch/数据整理/第二层聚类(待整理).txt"
    with open(origin_path, 'r', encoding='utf-8') as file:
        data = file.read()
    sections = data.split("==================")
    save_result = {
        "name": [],
        "desc": [],
        "children":[],
        "exist":[]
    }
    for section in sections:
        subfea_start = section.find("input:")
        subfea_end = section.find("10", subfea_start + 1)
        substring_1 = section[subfea_start+6:subfea_end]
        lines = substring_1.splitlines()
        subfea_name_list = [line[:].split(':')[1].strip() for line in lines if line != ""]
        check_result = check_in(subfea_name_list)
        cluster_start = section.find("cluster:")
        cluster_end = section.find("10", cluster_start)
        substring_2 = section[cluster_start+8:cluster_end]
        cluster_name = substring_2.split(':')[0].strip()
        print("============")
        print(cluster_name)
        cluster_desc = substring_2.split(':')[1].strip()
        save_result["name"].append(cluster_name)
        save_result["desc"].append(cluster_desc)
        save_result["children"].append(subfea_name_list)
        save_result["exist"].append(check_result)
    df = pd.DataFrame(save_result)
    output_path = "/home/jindm/project/HieGroupArch/cluster/2_layer.xlsx"
    df.to_excel(output_path, index=False)   

def main():
    process7()

if __name__=="__main__":
    main()