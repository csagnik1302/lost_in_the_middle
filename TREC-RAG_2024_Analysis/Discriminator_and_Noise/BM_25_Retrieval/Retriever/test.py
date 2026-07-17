# import json
# import gzip

# list1=['0'+str(i) for i in range(10)]
# list2=[str(i) for i in range(10,60)]
# list1.extend(list2)


# for i in list1:
#     PATH=f'/home/irlab/sagnik/TREC-RAG_2024_Analysis/Discriminator_and_Noise/Discriminator/Data/msmarco_v2.1_doc_segmented/msmarco_v2.1_doc_segmented_{i}.json.gz'
    
#     with gzip.open(PATH,'r') as f:
#         for i in f:

#             ms_marco_input=json.loads(i)
#             print(ms_marco_input)
#             break
#     break

a=2

print(f'{{{a}}}')