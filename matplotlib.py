#import matplotlib
#print(matplotlib.__version__)

#matplotlib pyplot
import matplotlib.pyplot as plt
import numpy as np
xpoints=np.array([0,6])
ypoints=np.array([0,250])
plt.plot(xpoints,ypoints)
plt.show()

#plotting without line
import matplotlib.pyplot as plt
import numpy as np
xpoints=np.array([1,8])
ypoints=np.array([3,10])
plt.plot(xpoints,ypoints,'o')
plt.show() 

#multiple lines
import matplotlib.pyplot as plt 
import numpy as np
xpoints=np.array([1,2,6,8])
ypoints=np.array([3,8,1,10])
plt.plot(xpoints,ypoints)
plt.show()

#default x-points
import matplotlib.pyplot as plt
import numpy as np
ypoints=np.array([3,8,1,10,5,7])
plt.plot(ypoints)
plt.show()

#markers
import matplotlib.pyplot as plt
import numpy as np
ypoints=np.array([3,8,1,10])
plt.plot(ypoints,marker='o')
plt.show()

#formart string marker/line/color
import matplotlib.pyplot as plt
import numpy as np
ypoints=np.array([3,8,1,10])
plt.plot(ypoints,'*-b')
plt.show

#markersize (ms), markeredgecolor(mec),markerfacecolor(mfc)
import matplotlib.pyplot as plt
import numpy as np
ypoints=np.array([3,8,1,10])
plt.plot(ypoints,marker='o',ms=10,mec='b',mfc='r')
plt.show()

#multiple lines
import matplotlib.pyplot as plt
import numpy as np
y1=np.array([3,8,1,10])
y2=np.array([6,2,7,11])
plt.plot(y1)
plt.plot(y2)
plt.show()

import matplotlib.pyplot as plt
import numpy  as np
x1=np.array([0,1,2,3])
y1=np.array([3,8,1,10])
x2=np.array([0,1,2,3])
y2=np.array([6,2,7,11])
plt.plot(x1,y1,x2,y2)
plt.show()

#creating labels for a plot
import numpy as np
import matplotlib.pyplot as plt
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x,y)
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.show()

#create a title
import numpy as np
import matplotlib.pyplot as plt
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x,y)
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.show()

#set fonts for title and labels
import numpy as np
import matplotlib.pyplot as plt
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x,y)
font1={'family':'serif','color':'blue','size':20}
font2={'family':'serif','color':'darkred','size':15}
plt.title("Sports Watch Data",fontdict=font1)
plt.xlabel("Average Pulse",fontdict=font2)
plt.ylabel("Calorie Burnage",fontdict=font2)
plt.show()

#add grid lines
import numpy as np
import matplotlib.pyplot as plt
x=np.array([80,85,90,95,100,105,110,115,120,125])
y=np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x,y)
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")
plt.grid()
plt.show()

#matplotlib subplot
#the figure has 1 row and 2 columns
import matplotlib.pyplot as plt
import numpy as np 
x=np.array([0,1,2,3])
y=np.array([3,8,1,10])
plt.subplot(1,2,1)
plt.plot(x,y)
x=np.array([0,1,2,3])
y=np.array([10,20,30,40])
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()

#creating scatter plots
import matplotlib.pyplot as plt
import numpy as np 
x=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y=np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x,y)
plt.show()

#compare plots
import matplotlib.pyplot as plt
import numpy as np 
x=np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y=np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x,y)
x=np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y=np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x,y)
plt.show()

#colormap
import matplotlib.pyplot as plt
import numpy as np

x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])

plt.scatter(x, y, c=colors, cmap='viridis')

plt.show()

#matplotlib bars
import matplotlib.pyplot as plt
import numpy as np
x=np.array(["A","B","C","D"])
y=np.array([3,8,1,10])
plt.bar(x,y)
plt.show()

#horizontal bars
import matplotlib.pyplot as plt
import numpy as np
x=np.array(["A","B","C","D"])
y=np.array([3,8,1,10])
plt.barh(x,y)
plt.show()

#matplotlib histogram hist()
import numpy as np
x=np.random.normal(170,10,250)
print(x)
