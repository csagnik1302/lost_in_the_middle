def all_score(judgement_list):

    total=0

    for i in judgement_list:
        if i=='support':
            total+=1
        elif i=='partial_support':
            total+=0.5
    
    score=total/len(judgement_list)

    return score

######

def all_strict_score(judgement_list):

    total=0

    for i in judgement_list:
        if i=='support':
            total+=1
    
    score=total/len(judgement_list)

    return score

######

def vital_score(judgement_list, relevance_list):

    total=0
    length_vital=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="vital":
            length_vital+=1
            if judgement_list[i]=='support':
                total+=1
            elif judgement_list[i]=='partial_support':
                total+=0.5

    score=total/length_vital

    return score


######


def vital_strict_score(judgement_list, relevance_list):

    total=0
    length_vital=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="vital":
            length_vital+=1
            if judgement_list[i]=='support':
                total+=1

    score=total/length_vital

    return score


#######

def weighted_score(judgement_list,relevance_list):

    total_1=0
    length_vital_1=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="vital":
            length_vital_1+=1
            if judgement_list[i]=='support':
                total_1+=1
            elif judgement_list[i]=='partial_support':
                total_1+=0.5



    total_2=0
    length_vital_2=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="okay":
            length_vital_2+=1
            if judgement_list[i]=='support':
                total_2+=1
            elif judgement_list[i]=='partial_support':
                total_2+=0.5

    
    score=(total_1+0.5+total_2)/(length_vital_1+0.5+length_vital_2)

    return score


###########


def weighted_score_strict(judgement_list,relevance_list):

    total_1=0
    length_vital_1=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="vital":
            length_vital_1+=1
            if judgement_list[i]=='support':
                total_1+=1


    total_2=0
    length_vital_2=0

    for i in range(len(relevance_list)):
        if relevance_list[i]=="okay":
            length_vital_2+=1
            if judgement_list[i]=='support':
                total_2+=1

    
    score=(total_1+0.5+total_2)/(length_vital_1+0.5+length_vital_2)

    return score