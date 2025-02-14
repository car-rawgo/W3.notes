"""try:
    print(x)
except:
    print("An exception occured")
"""

#nameError
"""try:
    print(x)
except NameError:
    print("variable x is not defined")
except:
    print("something else went wrong")
"""
#Else
"""try:
    print("Hello")
except:
    print("Something went wrong")
else:
    print("Nothing went wrong")  
"""
#finally
"""try:
    print(x)
except:
    print("Something went wrong")
finally:
    print("The 'try except' is finished")
"""
#cleaning up resources
try:
    f=open("demofile.txt")
    try:
        f.write("Lorum Ipsum")
    except:
        print("something went wrong when writting to the file")
    finally:
        f.close()
except FileNotFoundError:
    print("something went wrong when opening the file")                                                     

#raise an exception
#x=-1
#if x<0:
 #   raise Exception("Sorry, no numbers below zero")

#cghgj
#y="hello"
#if not type(y) is int:
 #   raise TypeError("Only integers are alowed")


#inputs
#username=input("Enter username:")
#print("username is: "+username)

#python string formatting
#f-string
txt=f"the price is 49 dollars"
print(txt)

#placeholders and modifiers
price=59
text=f"The price is {price} dollars."
print(txt)

price=59
txt=f"the price is {price:.2f} dollars"
print(txt)

#perform operations
txt=f"the price is {20*59} dollars"
print(txt)

price=59
tax=0.25
txt=f"The price is{price+(price*tax)} dollars"
print(txt)

#if...else
price=49
txt=f"It is very {'Expensive' if price>50 else 'cheap'}"
print(txt)

fruit="apples"
txt=f"I love {fruit.upper()}"
print(txt)

def myconverter(x):
    return x*0.3048
txt=f"The plane is flying at a {myconverter(30000)} meter altitude"
print(txt) 

quantity=3
itemno=567
price=49
myorder="I want {} pieces of item number {} for {:.2f} dollars."
print(myorder.format(quantity,itemno,price))

age=21
name="Ian"
txt="His name is {1}. {1} is {0} years old"
print(txt.format(age,name))