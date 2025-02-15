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

   