import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from geolife_data_utilities import get_data
import random
from agnostic_prediction import agnostic_lite, eval_aglite
from implementation_cost import keep_time
import matplotlib.patches as mpatches
np.set_printoptions(suppress=True)

def make_vector(start, disbear):    
    x = start[0] + disbear[0]*cos(0.017453292519943295*(90-(disbear[1]*360)))
    y = start[1] + disbear[0]*sin(0.017453292519943295*(90-(disbear[1]*360)))
    return start[0], start[1], x, y

def plot_course(input, output, prediction, dataset, num, transports, transport, error):
    if input.shape==(1,15,2):
        input = input.reshape((15,2))
    if output.shape==(1,2):
        output = output.reshape((2,))
    if prediction.shape==(1,2):
        prediction = prediction.reshape((2,))
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    start = [0,0]
    minx=0
    maxx=-0
    miny=0
    maxy=-0
    for i in range(input.shape[0]):
        r1, r2, r3, r4 = make_vector(start, input[i])
        x1.append(1000*r1)
        y1.append(1000*r2)
        x2.append(1000*r3)
        y2.append(1000*r4)
        start=[r3, r4]
        if r3>maxx:
            maxx=r3
        elif r3<minx:
            minx=r3
        if r4>maxy:
            maxy=r4
        elif r4<miny:
            miny=r4
    r1, r2, r3, r4 = make_vector(start, output)
    x1.append(1000*r1)
    y1.append(1000*r2)
    x2.append(1000*r3)
    y2.append(1000*r4)
    if r3>maxx:
        maxx=r3
    elif r3<minx:
        minx=r3
    if r4>maxy:
        maxy=r4
    elif r4<miny:
        miny=r4

    r1, r2, r3, r4 = make_vector(start, prediction)
    x1.append(1000*r1)
    y1.append(1000*r2)
    x2.append(1000*r3)
    y2.append(1000*r4)     
    if r3>maxx:
        maxx=r3
    elif r3<minx:
        minx=r3
    if r4>maxy:
        maxy=r4
    elif r4<miny:
        miny=r4    

    np.x1=np.array(x1)
    np.y1=np.array(y1)
    np.x2=np.array(x2)
    np.y2=np.array(y2)
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    ax.quiver(np.x1, np.y1, np.x2-np.x1, np.y2-np.y1, angles='xy', scale_units='xy', scale=1, color=['black','black','black','black','black','black','black','black','black','black','black','black','black','black','black','blue','green'])    
    minx*=1000
    maxx*=1000
    miny*=1000
    maxy*=1000    
    width=maxx-minx
    height=maxy-miny    
    zoomout_percent = 0.02
    if width>height:                
        limits = [minx-width*zoomout_percent, maxx+width*zoomout_percent, (maxy+miny-width)/2-width*zoomout_percent, (maxy+miny+width)/2+width*zoomout_percent]
    else:        
        limits = [(maxx+minx-height)/2-height*zoomout_percent, (maxx+minx+height)/2+height*zoomout_percent, miny-height*zoomout_percent, maxy+height*zoomout_percent]    
    plt.axis(limits)    
    plt.title(dataset+'_test_'+str(num)+'\npredicted method:' + transports[transport] +'\n error: '+str(error)+'m')    
    
    handles, _ = ax.get_legend_handles_labels()    
    patch = mpatches.Patch(color='black', label='Input Movement')
    handles.append(patch)
    patch = mpatches.Patch(color='blue', label='Output Movement')
    handles.append(patch)
    patch = mpatches.Patch(color='green', label='Predicted Output')
    handles.append(patch)
    plt.legend(handles=handles, loc='upper right')

    plt.grid()
    plt.draw()
    plt.show()

def plot_random_course():
    area = random.choice(['NW', 'NE', 'SW', 'SE'])
    vehicle = random.choice(['bus', 'bike', 'car', 'subway', 'taxi', 'train', 'walk'])
    vehicle = random.choice(['walk'])
    dataset = area+'_'+vehicle
    _, X, _, Y = get_data(dataset)
    num = random.randint(0, X.shape[0])
    myinput = X[num]
    myoutput = Y[num]
    time_spent = keep_time(agnostic_lite, myinput)
    transport, myprediction = agnostic_lite(myinput)
    transport, error = eval_aglite(myinput, myoutput)
    transports = ['bike', 'bus', 'car', 'subway', 'taxi', 'train', 'walk']
    print('method predicted:', transports[transport])
    print('time spent predicting:', round(time_spent,3), 'seconds')
    print('error:', error, 'meters')
    plot_course(myinput, myoutput, myprediction, dataset, num, transports, transport, error)
    return

plot_random_course()