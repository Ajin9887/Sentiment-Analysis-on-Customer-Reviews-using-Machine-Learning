def findLargest(st,k):
    num_list=list(st)
    for i in range(len(num_list)):
        if k==0:
            break
        if num_list[i]!='9':
            num_list[i]='9'
            k=k-1
    return''.join(num_list)
st="569431"
k=3
print("Orignal no",st)
print("Largest no",findLargest(st,k))

            
