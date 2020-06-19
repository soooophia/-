def direction(n,str):
    start=0
    str_l=list(str)
    dir={'L':-1, 'R':1}
    for i in str_l:
        start=start+dir[i]
    start=start%4
    sol={0:'N',1:'E',2:'S',3:'W'}
    print(sol[start])

direction(3,'LRR')
