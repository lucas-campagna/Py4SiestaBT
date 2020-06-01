from numpy import arange, linspace, loadtxt, interp
from numpy.linalg import norm
from scipy.interpolate import interp1d

from siesta2bandos import BandFileReader
import matplotlib.pyplot as plt
import pandas as pd

from cycler import cycler

# global
bbox = dict(boxstyle='square', fc="#FFF2CC",ec='#D6B656')
colors=plt.cm.tab20.colors
plt.rc('axes', prop_cycle=cycler(color=colors))
title = lambda n: {'phos':'P','germ':'Ge','gep3':'$GeP_3$'}[n]

#functions
def get_cell_volume(SIESTA_output_path):
    with open(search('.out',SIESTA_output_path),'r') as fp:
        for line in fp.readlines():
            if 'volume = ' in line:
                return float(line.split()[4])

def get_fermi_energy(bt_path):
    with open(search('.intrans',bt_path),'r') as fp:
        return float(fp.readlines()[2].split()[0])
            
def read_trace(param):
    dirEstrutura = param['dirBoltzTraP']
    ef = get_fermi_energy(param['dirBoltzTraP'])
    data = pd.read_csv(search('.trace',dirEstrutura),
                                 usecols=[0,1,2,3,4,5,6,7,8,9],
                                 sep='\s+',
                                 skiprows=1,
                                 names=['E','T','N','DOS','S','s/t','R_H','ke','c','chi'])
    data['ZT'] = data['S']**2*data['s/t']*data['T']/data['ke']
    data['E'] = (data['E'] - ef)*13.605698066 # Ry -> eV
    v,x,z,i = read_xv(search('.XV',param['dirSIESTA']))
    cell_volume = get_cell_volume(param['dirSIESTA'])
    cell_volume *= 0.529177 # borh -> Ang
    for i,d in enumerate(['x','y','z']):
        if not d in param['extensao']:
            cell_volume /= (norm(v[i])) # Ang^3 -> Ang^2 | Ang^1
    data['N']/=cell_volume
    return data


import os
def search(pattern,root='.'):
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith(pattern):
                 return os.path.join(root, file)
                
def plotBand(axis,param,**kwargs):
    folder=param['dirSIESTA'] #folder or '../pristine/'+n

    bands = BandFileReader(search('.bands',folder)).read().parser()
    
    E = read_trace(param)['E']
    Emin = E.min()
    Emax = E.max()
    
    axis.plot(bands['x'],bands['y'],'k')
    axis.set_ylim(Emin,Emax)
    axis.set_xlim(bands['x'].min(),bands['x'].max())
    axis.grid(axis='y',linestyle='--')
    axis.set_ylabel('$E-E_f\ \ (eV)$')
    axis.set_xlabel(r'$\vec{k}$')
    axis.grid(axis='x',color='#ee4422',linestyle='--',alpha=0.5)
    axis.set_xticks(bands['sym_points'])
    axis.set_xticklabels(bands['sym_points_name'])
    
    retVal = {}
    
    if 'gap' in kwargs.keys() and kwargs['gap'] == True:
        homo = bands['y'][bands['y'] < 0].max()
        lumo = bands['y'][bands['y'] > 0].min()
        gap = lumo - homo
        
        ik = (bands['sym_points'][1:] - bands['sym_points'][:-1]).argmax()
        k = bands['sym_points'][ik]
        k += ((bands['sym_points'][1:] - bands['sym_points'][:-1])/2)[ik]
        
        axis.annotate(xy=(k,lumo),
                      xytext=(k,homo),
                      s='',
                      arrowprops=dict(arrowstyle='<->'),
                     )
        axis.annotate(xy=(k,(lumo+homo)/2),
                      s='${gap}\ eV$'.format(gap=gap),
                      bbox=bbox,
                      va='center',
                      ha='center'
                     )
        axis.hlines(homo,Emin-1,Emax+1,colors='#ee4422',linestyles='--',alpha=.6)
        axis.hlines(lumo,Emin-1,Emax+1,colors='#ee4422',linestyles='--',alpha=.6)
        retVal['gap'] = gap
    
    # Nao implementado ainda
    #if 'transition' in kwargs.keys() and kwargs['transition'] == True:
    #    dfbands = pd.DataFrame(index=bands['x'],data=bands['y'])
    #    homo = dfbands[dfbands < 0].max().max()
    #    k_homo = dfbands.index[dfbands[dfbands < 0].max().idxmax()]
    #    lumo = dfbands[dfbands > 0].min().min()
    #    k_lumo = dfbands.index[dfbands[dfbands > 0].min().idxmin()]
    #    axis.annotate(xy=(k_lumo,lumo),
    #                  xytext=(k_homo,homo),
    #                  s='',
    #                  arrowprops=dict(arrowstyle='->'),
    #                 )
    #    
    #    retVal['homo'] = {'k':k_homo,'val':homo}
    #    retVal['lumo'] = {'k':k_lumo,'val':lumo}
    return retVal
                

def read_xv(file):
    from numpy import zeros
    v = zeros((3,3))
    n = 0
    it = iter(open(file,'r').read().split())
    for k in range(3):
        for j in range(3):
            v[k,j] = float(next(it))
        for j in range(3):
            next(it)
    n = int(next(it))
    i = zeros((n,1))
    z = zeros((n,1))
    x = zeros((n,3))
    for k in range(n):
        i[k,0] = int(next(it))
        z[k,0] = int(next(it))
        for j in range(3):
            x[k,j] = float(next(it))
        for j in range(3):
            next(it)
    return v,x,z,i        


def mineZTMax(param):
    # Extracao de dados
    data = read_trace(param)
    t = data['T'].unique()[0]
    dataOut = {}#pd.DataFrame()
    
    dataOut['N'] = linspace(data['N'].min(),data['N'].max(),data[data['T'] == t]['N'].size*2)
    T = data['T'].unique()
    
    zt = pd.DataFrame()
    for t in T:
        zt[t] = interp1d(data[data['T'] == t]['N'],data[data['T'] == t]['ZT'],fill_value=(0,0),bounds_error=False)(dataOut['N'])
    dataOut['ZT'] = zt.max(axis=1)
    dataOut['T'] = zt.idxmax(axis=1)
    dataOut['E'] = [data['E'].iloc[abs(data['N'][data['T'] == data['T'][icc]]-cc).idxmin()] for icc,cc in enumerate(dataOut['N'])]
    return dataOut

def read_condtens(prop,param):
    columns = {
        'S':   list(range(12,21)),
        's/t': list(range(3,12)),
        'ke':  list(range(21,30))
    }[prop]
    #print(list(map(str,range(len(columns)))))
    #print(['Ef','T','N']+list(map(str,range(len(columns))))
    return pd.read_csv(search('.condtens',param['dirBoltzTraP']),
                       sep='\s+',
                       usecols=[0,1,2]+columns,
                       skiprows=1,
                       names=['Ef','T','N']+list(map(str,range(len(columns))))
                      )