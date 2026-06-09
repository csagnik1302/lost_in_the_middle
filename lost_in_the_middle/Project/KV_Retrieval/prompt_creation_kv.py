import json

############################################### IMPORTS ############################################### 

# with open(r"/home/irlab/sagnik/lost_in_the_middle/Project/KV_Retrieval/kv-retrieval-75_keys.jsonl","r") as f:
#   data=[]
#   for i in f:
#     data.append(json.loads(i))



def prompt(input_data,dataset_index,gold_index):

######## Starting #############################

    start="Extract the value corresponding to the specified key in the JSON object below. No need to generate code or anything else"
   
######## JSON Part #############################   

    # input1=input[k]['ordered_kv_records']

    # middle1=dict()

    # for i in input1:
    #     middle1[i[0]]=i[1]


######### KEY (END PART) ###############################

    gold_key=input_data[dataset_index]['key']


########## GOLD STANDARD PREPROECSSING #############

    json_data=input_data[dataset_index]['ordered_kv_records'].copy()

    gold_value=input_data[dataset_index]['value']
    gold_kv=[gold_key,gold_value]

    json_data.remove(gold_kv)
    json_data=json_data[:]
    json_data.insert(gold_index,gold_kv)


######## JSON (MIDDLE PART) #############################   

    middle1=dict()

    for i in json_data:
        middle1[i[0]]=i[1]
    
    middle=json.dumps(middle1)


######### FINAL PROMPT #######################

    prompt=f"{start}\n\nJSON data:\n{middle}\n\nKey:{gold_key}\nCorresponding value:"


    return prompt,gold_value



if __name__=="__main__":
   
   prompt1,gold_value=prompt(data,0,21)

   print(prompt1)
  