import numpy as np
import json

def get_word(index, itow):
    word = []
    for ii in index:
        word.append(itow[str(ii)])
    return word

kg_info = json.load(open('/home/yangxu/project/Transformer_caption/dict/kg_ssg.json'))
kg_dict_path = 'data/cocobu.json'

kg = kg_info['kg_matrix']
N_kg = len(kg)
kg_index = np.zeros([N_kg,20],dtype=np.int32)
kg_mask = np.zeros([N_kg,20],dtype=np.int32)
N_max = 0

kg_dict = json.load(open(kg_dict_path))
kg_i2w = kg_dict['ix_to_word']

kg_w2i = {}
for i in kg_i2w.keys():
    kg_w2i[kg_i2w[i]] = int(i)

t=0

for i in range(N_kg):
    kg_temp = kg[i][1:].split(' ')
    N_temp = len(kg_temp)
    ii = 1
    for j in range(N_temp):
        if kg_temp[j] not in kg_w2i.keys():
            ii = 0

    if ii == 1:
        N_max = np.max([N_max, N_temp])
        for j in range(N_temp):
            kg_index[t,j] = kg_w2i[kg_temp[j]]
            kg_mask[t,j] = 1
        t = t+1

kg_info = {}
N_wanted=10000
kg_info['kg_mask'] = kg_mask[:N_wanted,0:N_max]
kg_info['kg_index'] = kg_index[:N_wanted,0:N_max]
# json.dump(kg_info, open('/home/yangxu/project/Transformer_caption/dict/ssg.json', 'w'))
np.save('data/kg/ssg_py3.npy',kg_info)


# vocab_path = '/home/yangxu/project/causal_caption/data/cocobu_con.json'
# vocab_data = json.load(open(vocab_path))
# vocab_count = vocab_data['counts']
# vocab_tag = vocab_data['tags']
# wtoi = vocab_data['word_to_ix']
# itow = vocab_data['ix_to_word']
# for i in range(N_kg):
#     N_word = np.sum(kg_info['kg_mask'][i,:])
#     word = get_word(kg_info['kg_index'][i,:N_word],itow)
#     kg_temp = kg[i][1:]
#     print("++++++++++++{0}+++++++++++++".format(i))
#     print("word version: {0}".format(kg_temp))
#     print("index version: {0}".format(word))