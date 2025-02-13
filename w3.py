
import platform
x=platform.system()
print(x)

#list all names in platform module
import platform
x=dir(platform)
print(x)


#python dates
import datetime
x=datetime.datetime.now()
print(x)

#create date objects
import datetime
x=datetime.datetime(2025,2,11)
print(x)

#strtime method
import datetime
x=datetime.datetime(2025,2,11)
print(x.strftime("%d"))

#python math
x=min(5,10,25)
y=max(5,10,25)
print(x)
print(y)

#absolute numbers
x=abs(-7.25)
print(x)

#power values 
x=pow(4,3)
print(x)

#the math module
import math
x=math.sqrt(81)
print(x)

import math
x=math.pi
print(x)

#json in python
import json
x='{"name":"john","age":30,"city":"newyork"}'
y=json.loads(x)
print(y["age"])

#json.dumps()
import json
x={
    "name":"john",
    "age":36,
    "country":"Kenya"
}
y=json.dumps(x)
print(y)

#convert python objects in json strings
import json
print(json.dumps({"name":"john","age":30}))
print(json.dumps(["apple","bananas"]))
print(json.dumps(("apple","bananas")))
print(json.dumps("hello"))
print(json.dumps(43))
print(json.dumps(39.42))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None)) 

import json
x={
    "name":"john",
    "age":30,
    "married":True,
    "divorced":False,
    "children":("Ann","Billy"),
    "pets":None,
    "cars":[
        {"model":"BMW230","mpg":27.5},
        {"model":"ford edge","mpg":24.1}
    ]
}
print(json.dumps(x,indent=4,separators=(".","=")))

#format the results using indents
import json
x={
    "name":"john",
    "age":30,
    "married":True,
    "divorced":False,
    "children":("Ann","Billy"),
    "pets":None,
    "cars":[
        {"model":"BMW230","mpg":27.5},
        {"model":"ford edge","mpg":24.1}
    ]
}
print(json.dumps(x,indent=4,sort_keys=True))

#python Regular Expression
#RegEx module
import re
txt="The rain in Spain"
x=re.search("^The.*Spain$",txt)
if x:
    print("Yes, we have a match")
else:
    print("No match")   

#findall() functions
import re
txt="The rain in spain"
x=re.findall("ai",txt)
print(x)

#split() function
import re
txt="The rain in Spain"
x=re.split("\s",txt)
print(x)

#sub() function
import re
txt="the rain in spain"
x=re.sub("\s","9",txt)
print(x)

#span() funtion
import re
txt="The rain in Spain"
x=re.search(r"\bS\w+",txt)
print(x.span())

#string function
import re
txt="The rain in Spain"
x=re.search(r"\bS\w+",txt)
print(x.string)

#group()
import re
txt="The rain in Spain"
x=re.search(r"\bS\w+",txt)
print(x.group())