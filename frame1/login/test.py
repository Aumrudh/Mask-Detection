f=open("test.txt","r")
s=f.read()
f.close()
a=list(map(int,s.split()))
a[0]+=1
a[1]+=1
a[2]+=1
a[3]+=1
for i in range(0,len(a)):
	a[i]=str(a[i])
a=' '.join(a)
f=open("test.txt","w")
f.write(a)