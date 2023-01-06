import numpy as np
import os
import json

def give_word(d, i):
    if i == 0:
        return -1
    else:
        return d[i].split(' ')

dict_path = '/home/yangxu/project/self-critical.pytorch/data/spice_sg_dict2.npz'
ssg_path = '/home/yangxu/project/self-critical.pytorch/data/coco_spice_sg2'
kg_dict_path = 'data/cocobu.json'

dict_data = np.load(dict_path)['spice_dict'][()]
i2w = dict_data['ix_to_word']
w2i = dict_data['word_to_ix']

kg_dict = json.load(open(kg_dict_path))
kg_i2w = kg_dict['ix_to_word']
# kg_w2i = kg_dict['word_to_ix']

kg = {}
t = 0
for file in os.listdir(ssg_path):
    if file.endswith(".npy"):
        if t % 1000 == 0:
            print(t)
        t = t+1
        ssg_data=np.load(ssg_path +'/'+file)
        rela = ssg_data[()]['rela_info']
        attr_info = ssg_data[()]['obj_info']

        N_attr = len(attr_info)
        N_rela = len(rela)
        for rela_id in range(N_rela):
            s = give_word(i2w, int(attr_info[int(rela[rela_id][0])][0]))
            if s == -1:
                continue
            o = give_word(i2w, int(attr_info[int(rela[rela_id][1])][0]))
            if o == -1:
                continue
            r = give_word(i2w, int(rela[rela_id][2]))
            if r == -1:
                continue

            kg_list = s+ r + o
            if len(kg_list) >= 10:
                continue
            kg_temp = ''
            for w in kg_list:
                kg_temp = kg_temp + ' ' + w
            if kg_temp not in kg.keys():
                kg[kg_temp] = 1
            else:
                kg[kg_temp] = kg[kg_temp] + 1


        for attr_id in range(N_attr):
            attr = attr_info[attr_id]
            if len(attr)<2:
                continue
            obj = give_word(i2w, int(attr[0]))
            if obj == -1:
                continue
            for j in range(len(attr)-1):
                a = give_word(i2w, int(attr[j+1]))
                if a == -1:
                    continue
                kg_list = a + obj
                if len(kg_list) >= 10:
                    continue

                kg_temp = ''
                for w in kg_list:
                    kg_temp = kg_temp + ' ' + w
                if kg_temp not in kg.keys():
                    kg[kg_temp] = 1
                else:
                    kg[kg_temp] = kg[kg_temp] + 1

kg_sorted_keys = sorted(kg, key=kg.get, reverse=True)
kg_matrix = kg_sorted_keys[0:20000]
kg_count = {}
for kg_temp in kg_matrix:
    kg_count[kg_temp] = kg[kg_temp]
kg_info = {}
kg_info['kg_matrix'] = kg_matrix
kg_info['kg_count'] = kg_count
json.dump(kg_info, open('dict/kg_ssg.json', 'w'))



