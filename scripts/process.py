import json
import os
import random
import csv
import plotly.express as px
import pandas as pd

def savebyupdate(tosave,save_path):
    contents = tosave
    if os.path.exists(save_path):
        with open(save_path, "r", encoding='utf-8') as f1:
            contents = json.load(f1)
            assert type(contents) == dict
            contents.update(tosave)
    with open(save_path, "w", encoding='utf-8') as f2:
        json.dump(contents, f2,ensure_ascii=False,indent=4)


def print_file(filename='safetytest.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        files = json.load(f)
        total = 0
        i=1
        for key in files.keys():
            print(i, key,len(files[key]))
            i+=1
            total+=len(files[key])
        print(total)

def print_file2(filename='safetytest.json'):
    with open(filename, 'r', encoding='utf-8') as f:
        files = json.load(f)
        total = 0
        i=1
        for key1 in files.keys():
            for key2 in files[key1].keys():
                print(i, key1,key2,len(files[key1][key2]))
                i+=1
                total+=len(files[key1][key2])
        print(total)




def combine_dim(newdim,*dims):
    with open("cognizetest.json", 'r', encoding='utf-8') as f:
        files = json.load(f)
    temp = []
    for d in dims:
        temp += files[d]
    for d in dims:
        del files[d]
    files[newdim]=temp
    with open("cognizetest.json",'w',encoding='utf-8') as f:
        json.dump(files,f,ensure_ascii=False,indent=4)


def extend_items():
    file1 = "knowledgetest.json"
    file2 = "knowledgetest2.json"
    with open(file1, 'r', encoding='utf-8') as f:
        file_1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        file_2 = json.load(f)
    for key in file_1.keys():
        id = len(file_2[key])
        for item in file_1[key]:
            item['Id'] = id
            id+=1
            file_2[key].append(item)
    with open("knowledgetest3.json",'w',encoding='utf-8') as f:
        json.dump(file_2,f,ensure_ascii=False,indent=4)


def pick_data(read_file,save_path, max_num=10):
    new_file={}
    total_num=0
    with open(read_file, 'r', encoding='utf-8') as f:
        files = json.load(f)
        for key in files.keys():
            new_file[key]=[]
            if len(files[key])<max_num:
                new_file[key] = files[key]
                total_num += len(files[key])
            else:
                new_id_lists = random.sample(range(len(files[key])),max_num)
                for id in range(len(new_id_lists)):
                    temp = {}
                    temp["Id"] = id
                    temp["Question"] = files[key][id]["Question"]
                    temp["Answer"] = files[key][id]["Answer"]
                    temp["Type"] = files[key][id]["Type"]
                    temp["Source"] =files[key][id]["Source"]
                    new_file[key].append(temp)
                total_num += max_num

    save_path = str.join(read_file.split('/')[-1].split('.')[0].split('_')[:-2].append(str(total_num)),'_')+'.json'
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(new_file,f,ensure_ascii=False,indent=4)
    print_file(filename=save_path)



def make_ori_data(rootdir,save_path="test_ori.json"):
    tosave={}
    for filename in os.listdir(rootdir):
        
        filepath = os.path.join(rootdir, filename)
        firstclass = filename.split('_')[1]
        if not os.path.isfile(filepath):
            continue
      
        with open(filepath,'r',encoding='utf-8') as f:
            files = json.load(f)
            tosave[firstclass]=files
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(tosave,f,ensure_ascii=False,indent=4)
    print_file2(save_path)

def make_sent_data(ori_file="test_ori.json", save_path="test_sent.json"):
    with open(ori_file, 'r', encoding='utf-8') as f:
        files = json.load(f)
        for key in files.keys():
            for type in files[key].keys():
                for item in files[key][type]:
                    del item['Answer']
                    del item['Source']
                    del item['Type']
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(files,f,ensure_ascii=False,indent=4)
    print_file2(save_path)

def make_dev_data(read_file="test_ori.json",max_num=2,save_path='test_dev.json'):
    new_file={}
    with open(read_file, 'r', encoding='utf-8') as f:
        files = json.load(f)
        for key in files.keys():
            new_file[key]={}
            for key2 in files[key].keys():
                new_file[key][key2]=[]
                if len(files[key][key2])<max_num:
                    new_file[key][key2] = files[key][key2]
                else:
                    new_id_lists = random.sample(range(len(files[key][key2])),max_num)
                    for id in range(len(new_id_lists)):
                        temp = {}
                        temp["Id"] = id
                        temp["Question"] = files[key][key2][id]["Question"]
                        temp["Answer"] = files[key][key2][id]["Answer"]
                        temp["Type"] = files[key][key2][id]["Type"]
                        new_file[key][key2].append(temp)
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(files,f,ensure_ascii=False,indent=4)
    print_file2(filename=save_path)


def main(all_data_path,new_data_path):
    if not os.path.exists(new_data_path):
        os.mkdir(new_data_path)
    max_num_setting = {
        'aligntest_345.json':1,
        'cognizetest_2270':10,
        'hallutest_450':10,
        'iftest_44':10,
        'knowledgetest_1606':10,
        'longtest_200':10,
        'safetytest_10W':10
    }
    # step 1: 选数据
    for filename in os.listdir(all_data_path):
        filepath = os.path.join(all_data_path, filename)
        if not os.path.isfile(filepath):
            continue
        save_path = os.path.join(new_data_path, filename)
        pick_data(filepath,save_path, max_num=max_num_setting[filename])
    
    # step 2:合并等
    make_ori_data(rootdir=new_data_path,save_path='test_ori1.json')  # 包含测试数据，标准答案等详细信息
    make_dev_data() # 少量数据，用于发给用户进行对齐的，比较少用
    make_sent_data() # 发送版数据，将答案和其他信息抹去，只保留提问。
    
if __name__ == '__main__':
    all_data_path = '../AllData'
    new_data_path = '../datas4aicompany0407'
    main(all_data_path, new_data_path)
    for f in os.listdir():
        print_file(os.path.join(new_data_path+f))
    
            
            