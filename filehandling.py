f=open("abc.txt","w")
f.write("""welcome to file handling.
my name is Ian.
am a programmer,
mathematician
and the best existing man.""")
f.close()

f=open("abc.txt")
print(f.read())

f=open("abc.txt")
print(f.readline())
print(f.readline())

#instead of repeating all the lines to execute the whole text we can use for in function
f=open("abc.txt","r")
for x in f:
    print(x)
f.close()  

#appending the file
f=open("abc.txt","a")
f.write("just discovered am also brilliant")
f.close()

f=open("abc.txt")
print(f.read())

#new file
f=open("comedy.csv","w") 
f.write("""Age	Experience	Rank	Nationality	Go
36	10	9	UK	NO
42	12	4	USA	NO
23	4	6	N	NO
52	4	4	USA	NO
43	21	8	USA	YES
44	14	5	UK	NO
66	3	7	N	YES
35	14	9	UK	YES
52	13	7	N	YES
35	5	9	N	YES
24	3	5	USA	NO
18	3	7	UK	YES
45	9	9	UK	YES

        """)  
f.close()
import pandas as pd
data={
   'Age':[36,42,23,52,43,44,66,35,52,35,24,18,45],
   'Experience':[10,12,4,4,21,14,3,14,13,5,3,3,9],
   'Rank':[9,4,6,4,8,5,7,9,7,9,5,7,9],
   'Nationality':["UK","USA","N","USA","USA","UK","N","UK","N","N","USA","UK","UK"],
   'Go':["NO","NO","NO","NO","YES","NO","YES","YES","YES","YES","NO","YES","YES"]
}
df=pd.DataFrame(data)
print(df)