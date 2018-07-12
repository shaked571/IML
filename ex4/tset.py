from random import *
for i in range(5600):
    print('.', end='')

newlist=[2,3,1,5,6,123,436,124,223,3]
# total 5600 values seprated by commas in the above list

emptylist = []
for values in newlist:
          convstr = str(values)
          convstr = convstr.split(",")
          emptylist.extend(convstr)

k=0
for i in range(5600):
    for j in range(0,4):
        print(i,j,emptylist[k])
        k=k+1

