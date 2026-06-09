import json


########################## STARTING #####################

def prompt_qa(PATH,position):

    with open(PATH,"r") as f:
        out=[]
        for i in f:
            out.append(json.loads(i))

    start="Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n\n"

    middle=''
    count=1

    middle1=out[position]['ctxs']

    for j in middle1:
        title=j['title']
        text=j['text'].replace('\n', ' ').strip()
        middle+=f"Document [{count}] (Title: {title}) {text}\n"
        count+=1
    
    end=f"\nQuestion: {out[position]['question']}\nAnswer:"

    prompt=start+middle+end

    answers=out[position]['answers']

    return prompt,answers

    


if __name__=="__main__":

    prompt,answers=prompt_qa(r"/home/irlab/sagnik/lost_in_the_middle/Project/QA/nq-open-10_total_documents_gold_at_0.jsonl",1)


    print(prompt)