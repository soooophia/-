# def divBy(l,h):
#     result=0
#     tmp=0
#     count=0
#     for i in range(l,h+1):
#         if i==l:
#             for j in range(1,i+1):
#                 tmp=j
#                 result +=tmp%3
#         else:
#              tmp=i
#              result+=tmp%3
#         if result%3==0:
#              count+=1
#
#     return count
#
# print(divBy(2,5))

a=['1','2','<3']
for i,item in enumerate(a):
    if item=='<3':
        a[i]='3'
print(a)