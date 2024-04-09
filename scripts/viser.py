# -- coding:utf-8 --
from calendar import prmonth
from cgitb import reset
import time
import requests
import json,jsonpath
import re
import  matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys,argparse
import pandas as pd
import os
import plotly.express as px

plt.rcParams['font.family'] = 'SimSun'
plt.rcParams['axes.unicode_minus'] = False
parser = argparse.ArgumentParser()
parser.add_argument('--ori_file',default="test_ori.json")
parser.add_argument('--responsed_file', default="test_sent_responsed.json")
parser.add_argument('--judgeUrl', default='http://172.27.216.52:8101/qwen1.5-72B-chat-GPTQ-Int4')
args = parser.parse_args()

def draw_sunburst(firstclass,secondclass,knowledgeclass,filename='test_ori.json'):
    datas = []
    # datas.append(['firstclass','secondclass','num'])
    with open(filename, 'r', encoding='utf-8') as f:
        files = json.load(f)
        # 先处理知识维度的
        for k1 in knowledgeclass.keys():
            num=0
            for k2 in knowledgeclass[k1]:
                num+=len(files['knowledgetest'][k2])
            datas.append(['知识',k1,num])
        total = 0
        i=1
        for c1 in files.keys():
            if c1 == 'knowledgetest':
                continue
            
            for c2 in files[c1].keys():
                if c1 not in secondclass:
                    datas.append([firstclass[c1],c2,len(files[c1][c2])])
                else:
                    datas.append([firstclass[c1],secondclass[c1][c2],len(files[c1][c2])])
    print(datas)
    datas = pd.DataFrame(datas)
    fig = px.sunburst(datas,path=[0,1],values=2,color=2,color_continuous_scale='blues')
    fig.update_traces()
    fig.show()


def vis2(readfile='test_results.json',modelname:str="",e2n=None):
    with open(readfile, 'r', encoding='utf-8') as f:
        jsondata = json.load(f)
    df=pd.DataFrame(columns=['一级能力维度','二级能力维度','得分'])
    for dim in jsondata.keys():        
        for kind in jsondata[dim].keys():
            scorelist=jsonpath.jsonpath(jsondata,f'$.{dim}.{kind}..Judgement')
            score_dim_kind=sum(scorelist)/len(scorelist)
            df.loc[len(df)]=[dim,kind,score_dim_kind]
        x=list(jsondata[dim].keys())
        y=df.loc[df['一级能力维度']==dim,'得分'].tolist()
        yy = [a*100 for a in y]
        title=f"{modelname}大模型-{dim}"
        drawbar(x,yy,title)
        drawradar(x,yy,title)
    df.to_excel(readfile+'.xlsx',index=False)

    
def vis(dim, color,readfile='test_results.json',e2n=None):
    datas = {}
    with open(readfile, 'r', encoding='utf-8') as f:
        files = json.load(f)
        for key in files.keys():
            datas[key]={}
            for kind in files[key].keys():
                datas[key][kind]=0
                for item in files[key][kind]:
                    datas[key][kind] += item['Judgement']
                datas[key][kind] /= (item['Id']+1)
    print(datas)
    _sorted_dim, _sorted_scores = zip(*sorted(zip(datas[dim].keys(), datas[dim].values()), key=lambda x:x[1], reverse=True))
    sorted_dim = []
    sorted_scores = []
    if e2n:
        func_e2n = lambda m:e2n[m]
    for i in range(len(_sorted_scores)):
        sorted_scores.append(round(_sorted_scores[i],3))
        if e2n:
            sorted_dim.append(func_e2n(_sorted_dim[i]))
        else:
            sorted_dim.append(_sorted_dim[i])

    plt.rcParams['font.sans-serif'] = ['SimHei']
    
    FIGSIZE=(8,5)

    # basecolor = color
    # colors = [adjust_lightness(basecolor, 1.5) if bin_edge < 0.3 else basecolor if bin_edge <= 0.5 else adjust_lightness(basecolor, 0.5) for bin_edge in sorted_scores]
    plt.figure(figsize=FIGSIZE)

    bars = plt.bar(sorted_dim, sorted_scores, color=color)
    plt.bar_label(bars, sorted_scores)
    plt.xlabel(dim)
    plt.ylabel("得分")
    plt.ylim((0, 1))
    plt.yticks(np.arange(0,1,0.1))
    plt.tight_layout()
    plt.savefig(dim+'.png',dpi=200, bbox_inches='tight')
    plt.show()

def statistic(files,knowledgeclass):
    datas = {}
    datas['knowledgetest']={}
    # 先处理知识维度的
    for k1 in knowledgeclass.keys():
        temp=0
        num=0
        for k2 in knowledgeclass[k1]:
            for item in files['knowledgetest'][k2]:
                temp += item['Judgement']
                num+=1
        datas['knowledgetest'][k1]=temp/num
    # 再处理其他维度
    for key1 in files.keys():
        if key1=='knowledgetest':
            continue
        datas[key1]={}
        for key2 in files[key1].keys():
            temp=0
            for item in files[key1][key2]:
                temp += item['Judgement']
            temp /=len(files[key1][key2])
            datas[key1][key2]=temp
    return datas

def draw_1bar(xlabel, y,savepath='vis_res/sum.png'):
    xlabel,y = zip(*sorted(zip(xlabel,y), key=lambda x:-x[1]))
    x = np.arange(len(xlabel))
    plt.figure(figsize=(10,5))
    bars = plt.bar(x, y)
    plt.bar_label(bars, y)
    plt.xticks(x,xlabel)
    plt.ylim(0,100)
    plt.axhline(60, color='grey', alpha=0.25,linestyle='--')
    plt.axhline(85, color='grey', alpha=0.25,linestyle='--')
    plt.savefig(savepath,dpi=200)
    plt.show()

def draw_2bar(xlabel,y1,y2,y1_label,y2_label,title,savepath='vis_res/gptvstc_bar.png'):
    width = 0.3
    # i=0
    x = np.arange(len(xlabel))
    # fig, ax = plt.subplots(layout='constrained')
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, y1, width,label=y1_label)
    rects2 = ax.bar(x+width+0.1, y2, width,label=y2_label)

    ax.bar_label(rects1)
    ax.bar_label(rects2)
        # i += 1
    
    ax.set_ylabel('得分')
    # ax.set_title('Penguin attributes by species')
    plt.axhline(60, color='grey', alpha=0.25,linestyle='--')
    plt.axhline(85, color='grey', alpha=0.25,linestyle='--')
    ax.set_xticks(x + width/2+0.1/2, xlabel)
    ax.set_yticks(np.arange(0,120,20))
    ax.legend(loc='best')
    # ax2 = plt.twinx()
    # ax2.set_ylabel('总分')
    # ax2.set_ylim(0,100)
    # plt.plot(x,values[-1])
    plt.title(title)
    ax.set_ylim(0, 120)
    plt.savefig(savepath,dpi=200)
    plt.show()

def draw_2radar(xlabel,y1,y2,y1_label,y2_label,title='星辰-12B V.S. GPT-3.5',savepath='vis_res/gptvstc_radar.png'):
    plt.figure()   
    # 雷达图 总图
    angles = np.linspace(0, 2 * np.pi, len(xlabel), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    y1.append(y1[0])
    y2.append(y2[0])
    xlabel.append(xlabel[0])
    # 绘制雷达图
    #plt.figure(1)
    plt.polar(angles, y1,label=y1_label)
    plt.polar(angles, y2,label=y2_label)
    # 标注y值
    # for i in range(len(angles)):
    #     plt.text(angles[i], radar_data1[i], f'{radar_data1[i]:.2f}', ha='center', va='center', color='blue')
    # 设置极坐标的标签
    plt.thetagrids(angles * 180/np.pi, labels=xlabel)
    # 填充多边形
    #plt.fill(angles, radar_data, alpha=0.25)
    plt.title(title)
    plt.legend(loc='upper left')
    plt.ylim(0,100)
    plt.savefig(savepath,dpi=200)
    plt.show()


def draw_xbar_1line(xlabel,y0,yx,yx_label,savepath):
    plt.figure()
    # 先画折线图
    x = np.arange(len(xlabel))
    plt.plot(x,y0)
    for i,(_x,_y) in enumerate(zip(x,y0)):
        plt.text(_x,_y+3,y0[i],color='red',fontsize=15)


    # 再画柱状图
    colormap = plt.get_cmap('Blues')
    colors   = colormap(np.linspace(0.1, 1, len(yx_label))) 
    # h=['...','oo','++','XX','**','...','oo','++','XX','**','...','oo','++','XX','**','...','oo','++','XX','**']
    width = 0.1
    for i in range(len(yx)):
        offset = width * i
        rects = plt.bar(x+offset, yx[i], width, label=yx_label[i],color=colors[i])
    plt.axhline(60, color='grey', alpha=0.25,linestyle='--')
    plt.axhline(85, color='grey', alpha=0.25,linestyle='--')
    plt.xticks(x + width, xlabel)
    plt.yticks(np.arange(0,120,20))
    plt.legend(loc='best')
    plt.savefig(savepath,dpi=200)
    plt.show()


def vis3(mapping_c1,mapping_c2,knowledgeclass):
    # 将所有数据读入并统计,合并知识维度
    rootdir = './rec'
    datas = {}
    for filename in os.listdir(rootdir):
        if filename.endswith('json'):
            with open(os.path.join(rootdir, filename),'r',encoding='utf-8') as f:
                files = json.load(f)

                modelname = filename[0:-5].split('_')[-1]
                datas[modelname] = statistic(files, knowledgeclass)
                
    # 英转中，计算均值
    datas_1={}
    datas_2={}
    datas_3={}
    dim_weight = {'safetytest':0.2,'cognizetest':0.1,'iftest':0.15,'aligntest':0.1,'hallutest':0.2,'knowledgetest':0.1,'longtest':0.15}
    for modelname in datas.keys():
        sum = 0
        datas_2[mapping_mn[modelname]]={}
        datas_3[mapping_mn[modelname]]={}
        for firstclass in datas[modelname].keys():
            datas_3[mapping_mn[modelname]][mapping_c1[firstclass]]={}
            temp = 0
            for key, value in datas[modelname][firstclass].items():
                temp += value
                if firstclass in mapping_c2:
                    datas_3[mapping_mn[modelname]][mapping_c1[firstclass]][mapping_c2[firstclass][key]]=round(value*100 , 1)
                else:
                    datas_3[mapping_mn[modelname]][mapping_c1[firstclass]][key]=round(value*100 , 1)
            temp /= len(datas[modelname][firstclass])
            sum += temp*dim_weight[firstclass]
            datas_2[mapping_mn[modelname]][mapping_c1[firstclass]] = round(temp*100 , 1)
        datas_1[mapping_mn[modelname]]=round(sum*100,1)

        
    # 画总图
    print(datas_1)
    draw_1bar(list(datas_1.keys()),list(datas_1.values()))
        
    # # 画两个对比图
    dims=list(datas_2['星辰-12B'].keys())
    draw_2radar(dims,list(datas_2['星辰-12B'].values()),list(datas_2['GPT-3.5'].values()),'星辰-12B','GPT-3.5',title='星辰-12B V.S. GPT-3.5',savepath='gptvstc_radar.png')
    for d in dims:
        xlabel = list(datas_3['星辰-12B'][d].keys())
        y1 = list(datas_3['星辰-12B'][d].values())
        y2 = list(datas_3['GPT-3.5'][d].values())
        draw_2bar(xlabel,y1,y2,'星辰-12B','GPT-3.5',title=d+' 星辰-12B V.S. GPT-3.5',savepath='vis_res/gptvstc_bar_'+d+'.png')
    
    # # 分维度讨论
    # # print(datas_2)
    models=list(datas_2.keys())
    y0=[]
    
    for d in dims:
        y0=[]
        yx=[]
        yx_label=[]
        for m in models:
            y0.append(datas_2[m][d])
            yx.append(np.array(list(datas_3[m][d].values())))
            yx_label=np.array(list(datas_3[m][d].keys()))
        yx=np.array(yx)
        yx=yx.T
        draw_xbar_1line(models,y0,yx,yx_label,'vis_res/'+d+'.png')



if __name__ == '__main__':
    mapping_mn={'telechat-7b':'星辰-7B','telechat-12b':'星辰-12B','qwen1.5-14B-chat':'千问1.5-14B', 'qwen-14B-chat':'千问-14B', 'glm4':'智谱4','gpt-3.5':'GPT-3.5',
             'baichuan2-13b-chat':'百川2-13B', 'qwen-7B-chat':'千问-7B', 'chatglm3-6b-chat':'智谱3-6B', 'baichuan2-7b-chat':'百川2-7B'}
    mapping_c1 = {'safetytest':'安全','cognizetest':'认知','iftest':'指令遵循','aligntest':'交互','hallutest':'幻觉','knowledgetest':'知识','longtest':'长文本'}
    mapping_c2={'cognizetest':{'penguins_in_a_table':'表格理解','object_counting':'计数能力','navigate':'导航','multistep_arithmetic_two':'多步算数','logical_deduction':'逻辑推演',
                                'causal_judgement':'结果判断','causal':'因果推理','boolean_expressions':'布尔表达式'},
                 'longtest':{'l_3k':'3千字','l_5k':'5千字','l_8k':'8千字','l_12k':'1万2字','l_17k':'1万7字'},
                 'hallutest':{'Knowledge':'事实幻觉','Misleading':'误导攻击幻觉','Misleading-hard':'指令幻觉'},
                 'safetytest':{'Offensiveness':'攻击性言论', 'Unfairness and Discrimination':'偏见歧视', 'Insult':'礼貌文明', 'Reverse_Exposure':'反面诱导', 
                               'Goal_Hijacking':'目标劫持', 'Prompt_Leaking':'提示泄露', 'Unsafe Inquiry':'不安全询问', 'Role_Play_Instruction':'角色指令', 
                               'PhysicalMental Health':'身心健康', 'Privacy and Property':'隐私财产', 'Ethics and Morality':'道德伦理', 'Crimes_And_Illegal':'违法犯罪', 'China Politics':'政治敏感'}
                }
    knowledgeclass={'自然科学与数学':{'advanced_mathematics','college_chemistry', 'college_physics','discrete_mathematics', 'high_school_biology', 'high_school_chemistry','high_school_geography',
                                'high_school_mathematics','logic', 'high_school_physics','middle_school_chemistry', 'middle_school_geography', 'middle_school_biology', 'middle_school_mathematics', 
                                'middle_school_physics', 'probability_and_statistics'},
                    '人文社会':{'art_studies','high_school_history', 'high_school_chinese', 'high_school_politics','law', 'legal_professional', 'mao_zedong_thought','ideological_and_moral_cultivation',
                            'chinese_language_and_literature','civil_servant','middle_school_history', 'middle_school_politics','modern_chinese_history','marxism', 'education_science'},
                    '应用科学与技术':{'accountant','operating_system' ,'computer_architecture', 'computer_network','physician','business_administration',
                               'plant_protection', 'tax_accountant', 'metrology_engineer', 'urban_and_rural_planner', 'teacher_qualification','college_programming',
                               'college_economics','environmental_impact_assessment_engineer','electrical_engineer'},
                    '生活常识与流行文化':{'sports_science', 'professional_tour_guide','basic_medicine','fire_engineer', 'clinical_medicine','veterinary_medicine'}

                    }
    draw_sunburst(mapping_c1,mapping_c2,knowledgeclass,filename='test_ori.json')
    vis3(mapping_c1,mapping_c2,knowledgeclass)