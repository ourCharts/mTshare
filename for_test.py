
def cosVector(vec1,vec2):
    x = [vec1[2]-vec1[0],vec1[3]-vec1[1]]
    y = [vec2[2]-vec2[0],vec2[3]-vec2[1]] 
    result1=0.0
    result2=0.0
    result3=0.0
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2     #sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    return result1/((result2*result3)**0.5) #结果显示

arr1 = [2,1,0,1]
arr2 = [4,1,3,0]
print(cosVector(arr1,arr2))