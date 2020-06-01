#!/opt/codes/anaconda2/bin/python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib as mpl
from xml.dom.minidom import parse
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
import os, sys
import numpy as np


# In[2]:


from struct import unpack
class WFSXReader():
    def __init__(self,fname):
        self.fp=open(fname,'rb')
        self.line_size=unpack('i',self.fp.read(4))[0]
        self.cont=self.line_size
    def read(self,dtype):
        if self.cont == 0:
            assert self.line_size == unpack('i',self.fp.read(4))[0]
            self.line_size = unpack('i',self.fp.read(4))[0]
            self.cont = self.line_size
        size_dtype= {'i':4,'f':4,'d':8,'s':20}[dtype]
        dtype= 'c'*20 if dtype == 's' else dtype
        self.cont-=size_dtype
        if  size_dtype == 20:
            return ''.join(map(str,unpack(dtype,self.fp.read(size_dtype)))).replace(' ','')
        return eval(''.join(map(str,unpack(dtype,self.fp.read(size_dtype)))))


# In[4]:


class WFSX(dict):
    def __init__(self,wfname):
        self.wfname=wfname
        self.wf=WFSXReader(wfname)
    def read(self):
        self['nk']=self.wf.read('i')
        gamma=self.wf.read('i')
        self['Num_spin']=self.wf.read('i')
        self['Num_Orb_Tot']=self.wf.read('i')
        
        self.species_index=np.zeros(self['Num_Orb_Tot'])
        self.species=np.zeros(self['Num_Orb_Tot'],dtype=object)
        self.orbitals_index=np.zeros(self['Num_Orb_Tot'])
        cnfigfio=np.zeros(self['Num_Orb_Tot'])
        symfio=np.zeros(self['Num_Orb_Tot'],dtype=object)
        self['k']=[{} for i in range(self['nk'])]
        
        for i in range(self['Num_Orb_Tot']):
            self.species_index[i]=self.wf.read('i')
            self.species[i]=self.wf.read('s')
            self.orbitals_index[i]=self.wf.read('i')
            cnfigfio[i]=self.wf.read('i')
            symfio[i]=self.wf.read('s')
        
        for i in range(len(symfio[1:])):
            i+=1
            lv=symfio[i-1]
            v=symfio[i]
            if v == 'p':
                symfio[i] = {'s':'py','py':'pz','pz':'px','px':'py'}[lv]
            elif v == 'P':
                symfio[i]='Pdxy' if 'p' in lv else {'Pdxy':'Pdyz','Pdyz':'Pdz2','Pdz2':'Pdxz','Pdxz':'Pdx2-y2'}[lv]
        
        orb = map(lambda u,v:'{:1.0f}'.format(u)+v,cnfigfio,symfio)
        self.orbitals = orb
        
        if self['Num_spin'] == 1:
            for ik in range(self['nk']):
                iik=self.wf.read('i')
                assert iik == ik + 1
                coord=[self.wf.read('d'),self.wf.read('d'),self.wf.read('d')]
                weight=self.wf.read('d')
                iispin= self.wf.read('i')
                assert iispin == 1
                nwflist=self.wf.read('i')
                
                self['k'][ik]['energy']=np.zeros(nwflist)
                self['k'][ik]['psi']=np.zeros(nwflist,dtype=dict)
                
                for iw in range(nwflist):
                    indwf=self.wf.read('i')
                    self['k'][ik]['energy'][iw]=self.wf.read('d')
                    wfw=np.array([abs(self.wf.read('f')+self.wf.read('f')*1j)**2 for j in range(self['Num_Orb_Tot'])])
                    wfw=wfw/sum(wfw)
                    self['k'][ik]['psi'][iw]={'value':wfw}
        else:
            for ik in range(self['nk']):
                self['k'][ik]['spin']=[{} for i in range(len(self['Num_spin']))]
                for ispin in range(self['Num_spin']):
                    iik=self.wf.read('i')
                    assert iik == ik + 1
                    coord=[self.wf.read('d'),self.wf.read('d'),self.wf.read('d')]
                    weight=self.wf.read('d')
                    iispin= self.wf.read('i')
                    assert iispin == ispin + 1
                    nwflist=self.wf.read('i')
                    self['k'][ik]['spin'][ispin]['energy']=np.zeros(nwflist)
                    self['k'][ik]['spin'][ispin]['psi']=np.zeros(nwflist)
                    for iw in range(nwflist):
                        indwf=self.wf.read('i')
                        self['k'][ik]['spin'][ispin]['energy'][iw]=self.wf.read('d')
                        wfw=np.array([abs(self.wf.read('f')+self.wf.read('f')*1j)**2 for j in range(self['Num_Orb_Tot'])])
                        wfw=wfw/sum(wfw)
                        self['k'][ik]['spin'][ispin]['psi'][iw]={'value':wfw}

    def generate_pcolor(self,band,apdos,palette,ef):
        pdos = map(lambda u: u.copy(),apdos)
        for i,v in enumerate(pdos):
            if not v.has_key('species'):
                v['species']=None
            if not v.has_key('n'):
                v['n']=None
            if not v.has_key('l'):
                v['l']=None
            if not v.has_key('m'):
                v['m']=None
            if not v.has_key('z'):
                v['z']=None
            pdos[i]['orbitals']={0:'s',None:'s'}[v['m']]                                               if v['l'] == 's' else                                 {-1:'py',0:'pz',1:'px',None:'p'}[v['m']]                              if v['l'] == 'p' else                                 {-2:'Pdxy',-1:'Pdyz',0:'Pdz2',1:'Pdxz',2:'Pdx2-y2',None:'Pd'}[v['m']] if v['l'] == 'd' else                                 None                                                                  if v['l'] == None else                                 str(v['m'])
        
        pcolor=np.zeros([len(band),len(pdos)])
        for ik,b in enumerate(band):
            tol = np.min(np.append(np.abs(np.diff(self['k'][ik]['energy'])),np.abs(np.diff(band))))
            e=self['k'][ik]['energy']-ef
            psi=self['k'][ik]['psi']
            for i,v in enumerate(pdos):
                wf = psi[np.abs(e-b).argmin()]
                filt={'species':np.zeros(self['Num_Orb_Tot'],dtype=bool),                      'orbitals_index':np.zeros(self['Num_Orb_Tot'],dtype=bool),                      'orbitals':np.zeros(self['Num_Orb_Tot'],dtype=bool)}
                for iorb in range(self['Num_Orb_Tot']):
                    # 1) Getting matching by species
                    filt['species'][iorb]=not v['species'] or self.species[iorb]==v['species']
                    
                    # 2) Getting matching by zetta number or orbital index
                    filt['orbitals_index'][iorb]=not v['z'] or                                                 (v['z'] == 1 and not self.orbitals_index[iorb] in (2,6,7,8,14,15,16,17) or                                                 (v['z'] == 2 and self.orbitals_index[iorb] in (2,6,7,8,14,15,16,17)))
                    # 3) Getting matching by orbitals
                    filt['orbitals'][iorb]=not v['orbitals'] or v['orbitals'] == self.orbitals[iorb][1:] or                                            (not v['m'] and v['orbitals']=='s'  and self.orbitals[iorb][1]=='s') or                                            (not v['m'] and v['orbitals']=='p'  and self.orbitals[iorb][1]=='p') or                                            (not v['m'] and v['orbitals']=='Pd' and self.orbitals[iorb][1:3]=='Pd')
                    # 4) Getting matching by first main number
                    filt['orbitals'][iorb]*=not v['n'] or v['n'] == self.orbitals[iorb][1]
        
                ifilt=filt['species']*filt['orbitals_index']*filt['orbitals']
                pcolor[ik,i]=np.sum(wf['value'][ifilt])
                
        pcolor=np.matmul(pcolor,np.array(palette)[:len(pdos)])
        return pcolor


# In[5]:


class BandFileReader(dict):
    def __init__(self,file):
        self.file=file
        
    def read(self):
        self.raw_data = open(self.file,'r').read()
        return self
    
    def parser(self):
        values=iter(self.raw_data.replace('\n','').replace(r"'"," ").split())
        #values=self.raw_data.replace('\n','').replace(r"'"," ").split()
        #print(values)
        #return
        ef=float(next(values))
        self['ef']=ef
        self['kmin']=float(next(values))
        self['kmax']=float(next(values))
        self['emin']=float(next(values))-ef
        self['emax']=float(next(values))-ef
        self['nband']=int(next(values))
        self['nspin']=int(next(values))
        self['nk']=int(next(values))
        self['x']=[]
        self['y']=[]
        if self['nspin'] == 1:
            for ik in range(self['nk']):
                self['x'].append(float(next(values)))
                self['y'].append([float(next(values))-ef for ib in range(self['nband'])])
        else:
            self['y2']=[]
            for ik in range(self['nk']):
                self['x'].append(float(next(values)))
                self['y'].append([float(next(values))-ef for ib in range(self['nband'])])
                self['y2'].append([float(next(values))-ef for ib in range(self['nband'])])
            self['y2']=np.array(self['y2'])
            
        self['x']=np.array(self['x'])
        self['y']=np.array(self['y'])
        
        nlines=int(next(values))
        self['sym_points']=[]
        self['sym_points_name']=[]
        
        x=self['x']
        de=[]
        for i in range(nlines):
            self['sym_points'].append(float(next(values)))
            val = next(values)
            val = val if val != r'Gamma' else '\Gamma'
            self['sym_points_name'].append('$'+val+'$')
            
            
            try:
                sp=self['sym_points'][-1]
                psp=self['sym_points'][-2]
                isp=list(x).index(sp)
                ipsp=list(x).index(psp)
                if isp - 1 == ipsp:
                    de+=[[isp,i],]
            except IndexError:
                continue
        self['sym_points']=np.array(self['sym_points'])
        sp = self['sym_points']
        spn = self['sym_points_name']
        for i,j in de:
            x[i:]-=x[i]-x[i-1]
            sp[j:]-=sp[j]-sp[j-1]
            spn[j]=spn[j-1]+'|'+spn[j]
            
        self['sym_points']=sp.tolist()
        sp = self['sym_points']
        for i,j in list(reversed(de)):
            spn.pop(j-1)
            sp.pop(j-1)
        self['sym_points'] = np.array(self['sym_points'])
        return self


# In[6]:


class PDOSFileReader(dict):
    def __init__(self,file):
        self.file=file
        self.orbitals=dict()
    
    def read(self):
        self.pdos=parse(self.file)
        return self
    
    def parser(self):
        self['nspin']=eval(self.pdos.getElementsByTagName('nspin')[0].childNodes[0].nodeValue)
        self['norbitals']=eval(self.pdos.getElementsByTagName('norbitals')[0].childNodes[0].nodeValue)
        self['energy_values']=np.array(map(eval,self.pdos.getElementsByTagName('energy_values')[0].childNodes[0].nodeValue.replace('\n','').split()))
        self['energy_unit']=self.pdos.getElementsByTagName('energy_values')[0].getAttribute('units')
        self['orbitals']=[]
        for io,elem in enumerate(self.pdos.getElementsByTagName('orbital')):
            self['orbitals'].append(dict())
            self['orbitals'][io]['data']=map(float,elem.getElementsByTagName('data')[0].firstChild.nodeValue.replace('\n',' ').split())
            self['orbitals'][io]['index']=int(elem.attributes.get('index').nodeValue)
            self['orbitals'][io]['atom_index']= int(elem.attributes.get('atom_index').nodeValue)
            self['orbitals'][io]['species']= elem.attributes.get('species').nodeValue
            self['orbitals'][io]['position']= map(float,elem.attributes.get('position').nodeValue.split())
            self['orbitals'][io]['n']= int(elem.attributes.get('n').nodeValue)
            self['orbitals'][io]['l']= int(elem.attributes.get('l').nodeValue)
            self['orbitals'][io]['m']= int(elem.attributes.get('m').nodeValue)
            self['orbitals'][io]['z']= int(elem.attributes.get('z').nodeValue)
        return self
    
    def set_data(self,data):
        self['x']=np.array(self['energy_values'])
        self['y']=[]
        self['y_tot']=np.zeros(len(self['x']))
        for io,do in enumerate(self['orbitals']):
                self['y_tot']=self['y_tot']+np.array(do['data'])
        data=np.array(data)
        for i,d in enumerate(data):
            to_plot=d.copy()
            
            to_plot.update(dict.fromkeys({'species','n','l','m','z'}-set(d.keys()),None))
            if type(to_plot['l']) == str:
                to_plot['l'] = {'s':0,'p':1,'d':2,'f':3,'g':4,'h':5,'i':6,'j':7}[to_plot['l']]
            
            self['y'].append(np.array(np.zeros(len(self['x']))))
            
            for io,do in enumerate(self['orbitals']):
                if (do['species'] == to_plot['species'] or to_plot['species'] == None) and                     (do['n'] == to_plot['n'] or to_plot['n'] == None) and                     (do['l'] == to_plot['l'] or to_plot['l'] == None) and                     (do['m'] == to_plot['m'] or to_plot['m'] == None) and                     (do['z'] == to_plot['z'] or to_plot['z'] == None):
                    self['y'][i]+=np.array(do['data'])
        self['y']=np.array(self['y']).transpose()


# In[11]:


class BandDosPlot():
    def __init__(self,parameters,**kwarg):
        
        # Input parameters
        
        if parameters.has_key('band_file') or parameters.has_key('wfsx_file'):
            band_file=parameters['band_file'] if parameters.has_key('band_file') else parameters['wfsx_file'].replace('.WFSX','')
            parameters['band_file']=band_file
        if parameters.has_key('pdos_file'):
            pdos_file=parameters['pdos_file']
        # self.output_file=parameters['output_file']
        self.graph_parameters=parameters
        
        # Graphics settings
        
        if not self.graph_parameters.has_key('axis'):
            if parameters.has_key('pdos_file') and parameters.has_key('band_file'):
                self.fig=plt.figure(dpi=150,figsize=(6,3),**kwarg)
                gs=GridSpec(1,3)
                self.axb=self.fig.add_subplot(gs[0,:2])
                self.axd=self.fig.add_subplot(gs[0,2],sharey=self.axb)
            elif parameters.has_key('pdos_file'):
                self.fig=plt.figure(dpi=120,figsize=(5,4),**kwarg)
                self.axd=self.fig.add_subplot(111)
            else:
                self.fig=plt.figure(dpi=150,figsize=(5,3),**kwarg)
                self.axb=self.fig.add_subplot(111)
        else:
            if self.graph_parameters['axis'].has_key('band'):
                self.axb = self.graph_parameters['axis']['band']
                self.fig = self.axb.get_figure()
            if self.graph_parameters['axis'].has_key('pdos'):
                self.axd = self.graph_parameters['axis']['pdos']
                self.fig = self.axd.get_figure()
        
        # Reading and parsing data
        
        if parameters.has_key('band_file'):
            self.band_data = BandFileReader(band_file).read().parser()
            if parameters.has_key('wfsx_file'):
                self.wfsx_data = WFSX(parameters['wfsx_file'])
                self.wfsx_data.read()
        if parameters.has_key('pdos_file'):
            self.pdos_data = PDOSFileReader(pdos_file).read().parser()
            self.pdos_data['energy_values']-=self.band_data['ef'] if parameters.has_key('band_file') else 0
        
        # Extra data heandeling
        
        self.extra_pdos_data = {}
        self.extra_band_data = {}
        if parameters.has_key('simple_band_file') and (parameters.has_key('band_file') or parameters.has_key('wfsx_file')):
            aux = np.loadtxt(parameters['simple_band_file'])
            self.extra_band_data['x'] = aux[:,0]
            self.extra_band_data['y'] = aux[:,1:]
        if parameters.has_key('simple_pdos_file') and parameters.has_key('pdos_file'):
            aux = np.loadtxt(parameters['simple_pdos_file'])
            self.extra_pdos_data['x'] = aux[:,0]
            self.extra_pdos_data['y'] = aux[:,1:]
    
    def plot(self,pdos=[{'l':'s'},{'l':'p'},{'l':'d'}],cmap='Set1',**kwargs):
        if self.graph_parameters.has_key('pdos_file'):
            self.pdos_data.set_data(pdos)
        legend=[ dic[dic.keys()[0]] if not dic.has_key('title') else dic['title'] for dic in pdos]
        palette=plt.get_cmap(cmap).colors
        
        
#         self.axb.tick_params(axis='both', which='major', pad=15)
        if self.graph_parameters.has_key('band_file'):
            for tick in self.axb.get_xaxis().get_major_ticks():
                tick.set_pad(2.)
                tick.label1 = tick._get_text1()
        
        # Shortcuts
        
        if self.graph_parameters.has_key('pdos_file'):
            ddx=self.pdos_data['x']
            ddy=self.pdos_data['y']
            ddyt=self.pdos_data['y_tot']
            # Trying to Avoid MemoryError
            del(self.pdos_data)
        if self.graph_parameters.has_key('band_file') or self.graph_parameters.has_key('wfsx_file'):
            bdx=self.band_data['x']
            bdy=self.band_data['y']
            lumo=bdy[bdy>0].min()
            homo=bdy[bdy<0].max()
            ef=self.band_data['ef']
            
#         Setting Energy Range
        if not self.graph_parameters.has_key('pdos_file'):
            if self.graph_parameters.has_key('energy_band_range'):
                ddx=self.graph_parameters['energy_band_range']
            elif not self.graph_parameters.has_key('energy_band_range'):
                ddx=(bdy.min(),bdy.max())
#         if self.graph_parameters.has_key('energy_band_range'):
#             ddx=self.graph_parameters['energy_band_range']
#         elif not self.graph_parameters.has_key('energy_band_range'):
#             ddx=(bdy.min(),bdy.max())
        
        # DOS Plot
        
        if self.graph_parameters.has_key('pdos_file'):
            if self.graph_parameters.has_key('band_file'):
                for iy,dy in enumerate(ddy.T):
                    self.axd.plot(dy,ddx,lw=0.8,color=mpl.colors.to_hex(palette[iy]))

                self.axd.fill_betweenx(ddx,0,ddyt,facecolor='grey',alpha=0.5)
                self.axd.plot(ddyt,ddx,'#888888',lw=0.8)
                self.axd.hlines(0,-1,ddyt.max()*1.2,'darksalmon',linestyles='--',lw=0.75)
                self.axd.hlines(homo,-1,ddyt.max()*1.2,'gray',linestyles='--',lw=0.75)
                self.axd.hlines(lumo,-1,ddyt.max()*1.2,'gray',linestyles='--',lw=0.75)

                # PDOS Plot Configs
                self.axd.set_xticks([])
                self.axd.set_xlabel('Density of States')
                self.axd.grid(True,)
                self.axd.set_xlim(0,ddyt.max()*1.1)
                self.axd.get_yaxis().set_visible(False)
                self.axd.grid(axis='y',ls='--',c='gainsboro')
                self.axd.grid(axis='x',ls='--')
                self.axd.legend(legend,loc=1)
            else:
                if self.graph_parameters.has_key('Ef'):
                    ddx-=self.graph_parameters['Ef']
                for iy,dy in enumerate(ddy.T):
                    self.axd.plot(ddx,ddy[:,iy],lw=0.8,color=mpl.colors.to_hex(palette[iy]))
                self.axd.fill_between(ddx,0,ddyt,facecolor='grey',alpha=0.5)
                self.axd.plot(ddx,ddyt,'#888888',lw=0.8)
#                 self.axd.vlines(0,-1,ddyt.max()*1.2,'darksalmon',linestyles='--',lw=0.75)

                # PDOS Plot Configs
                if self.graph_parameters.has_key('Ef'):
                    self.axd.set_xlabel('$E - E_f$ $(eV)$')
                else:
                    self.axd.set_xlabel('$E$ $(eV)$')
                self.axd.set_ylabel('Density of States')
                self.axd.grid(True,)
                self.axd.set_xlim(ddx.min(),ddx.max())
                self.axd.set_ylim(0,ddyt.max()*1.1)
                self.axd.grid(axis='x',ls='--',c='gainsboro')
                self.axd.grid(axis='y',ls='--')
                self.axd.legend(legend,loc=1)
        
        # Band Plot
        
        if self.graph_parameters.has_key('band_file') or self.graph_parameters.has_key('wfsx_file'):
        
            # Getting Colors and Painting band lines
            if self.graph_parameters.has_key('wfsx_file'):
                for band in bdy.T:
#                     if len(ddx[(ddx>band.min())&(ddx<band.max())]) == 0:
#                         continue
                    pcolor=self.wfsx_data.generate_pcolor(band,pdos,palette,ef)
                    pts = np.array([bdx,band]).T.reshape(-1, 1, 2)
                    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(seg,color=pcolor,linewidths=1)
                    self.axb.add_collection(lc)
            elif self.graph_parameters.has_key('pdos_file'):
                bcolor=np.zeros((len(ddx),len(pdos)))
                bcolor=np.matmul(ddy,np.array(palette)[:len(pdos)])
                for i in range(len(ddx)):
                    for j in range(len(bcolor[0,:])):
                        bcolor[i,j] = mpl.colors.Normalize(0,ddyt[i])(bcolor[i,j])
#                         bcolor[i,j] = mpl.colors.Normalize(0,bcolor[:,j].max())(bcolor[i,j]) # Highlight colors

                for band in bdy.T:
                    if len(ddx[(ddx>band.min())&(ddx<band.max())]) == 0:
                        continue
                    pcolor=bcolor[map(lambda y:np.abs(ddx-y).argmin(),band)]
                    pts = np.array([bdx,band]).T.reshape(-1, 1, 2)
                    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    lc = LineCollection(seg,color=pcolor,linewidths=1)
                    self.axb.add_collection(lc)
            else:
                for band in bdy.T:
                    self.axb.plot(bdx,band,'black')

            # Ploting extra bands
            if self.extra_band_data:
                if not self.graph_parameters.has_key('tight_extra'):
                    aux_x=[]
                    for i,v in enumerate(self.band_data['sym_points']):
                        try:
                            li=i-1
                            lv=self.band_data['sym_points'][i-1]
                        except IndexError:
                            continue
                        aux_x+=np.linspace(lv,v,                                           int(round(len(self.extra_band_data['x'])*len(bdx[(bdx>=lv)&(bdx<=v)])*1./len(bdx)))                                          ).tolist()[:-1]
                    self.extra_band_data['x']=np.array(aux_x+[self.band_data['sym_points'][-1]])
                else:
                    self.extra_band_data['x']=np.linspace(bdx[0],bdx[-1],len(self.extra_band_data['x']))

                if self.graph_parameters.has_key('shift_y_extra'):
                    self.extra_band_data['y']+=self.graph_parameters['shift_y_extra']
                else:
                    ebdy=self.extra_band_data['y']
                    elumo=ebdy[ebdy>0].min()
                    ehomo=ebdy[ebdy<0].max()
#                     self.extra_band_data['y']+=((lumo-elumo)+(ehomo-homo))/2
                    self.extra_band_data['y']+=lumo-elumo
                self.axb.plot(self.extra_band_data['x'],self.extra_band_data['y'],'black')

            # Annotaion gap
            if self.graph_parameters.has_key('show_gap'):
                gap={}
                gap['val']='{:.2f}'.format(lumo-homo)+' eV'
                skp=self.band_data['sym_points']
                bdxmax3=bdx.max()/3
                gap['x']=(skp[skp<bdxmax3].max()+skp[skp>bdxmax3].min())/2.
                gap['y']=(homo+lumo)/2.0
                self.axb.annotate("", xy=(gap['x'], lumo), xytext=(gap['x'], gap['y']),arrowprops=dict(arrowstyle="->",shrinkB=0))
                self.axb.annotate("", xy=(gap['x'], homo), xytext=(gap['x'], gap['y']),arrowprops=dict(arrowstyle="->",shrinkB=0))
                self.axb.text(gap['x'],gap['y'],gap['val'],ha='center',va='center',size='xx-small',bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8)))
                # Homo and Lumo lines (band and DOS)
                self.axb.hlines(homo,bdx.min(),bdx.max(),'gray',linestyles='--',lw=0.75)
                self.axb.hlines(lumo,bdx.min(),bdx.max(),'gray',linestyles='--',lw=0.75)

            # Fermi Energy line
            self.axb.hlines(0,bdx.min(),bdx.max(),'darksalmon',linestyles='--',lw=0.75)

            # Band Plot Configs
            self.axb.set_ylim(min(ddx),max(ddx))
            self.axb.set_xlim(min(bdx),max(bdx))
            self.axb.set_ylabel('$E-E_f$ $(eV)$')
            self.axb.set_xlabel('k-points')
            self.axb.set_xticks(self.band_data['sym_points'])
            self.axb.set_xticklabels(self.band_data['sym_points_name'],size='x-small')#,rotation=90)
            self.axb.grid(axis='y',ls='--')
            self.axb.grid(axis='x',ls='--',c='black')
        
#         if self.graph_parameters.has_key('band_file') and self.graph_parameters.has_key('pdos_file'):
        plt.tight_layout()
        self.fig.subplots_adjust(wspace=0, hspace=0)
        
    def save(self,fname=None):
        if fname:
            self.output_file = fname
        if self.graph_parameters.has_key('output_file'):
            self.output_file = self.graph_parameters['output_file']
        else:
            for ifname in os.walk('.').next()[2]:
                if ifname == self.output_file:
                    ans=raw_input('File '+ifname+' already exists. Overwrite? (y/n) ')
                    if ans.lower() == 'n':
                        self.output_file=raw_input('Enter with a new name:')
                    break
        self.fig.savefig(self.output_file)


# In[12]:


def show_help():
    print('''
Siesta2BanDos is a script to generate beautiful band-pdos graphs matplotlib based.

Usage: siesta2bandos [OPTIONS]

DESCRIPTION
    If neither -b nor -p options are used, this script will try to find both .bands and .PDOS
    files in this folder selecting the first in alphabetic order, if not found show this help and exit.

OPTIONS:
    -b, --bands (.bands file)
        sets .bands file (output from siesta) to be ploted. Must be followed by the .bands file.
        
    -p, --pdos (.PDOS file)
        sets .PDOS file (output from siesta) to be ploted.Must be followed by the .PDOS file.
        
    -f, --fat (wfsx file)
        sets WFSX format to be read. Generates fat bands.
    
    -ebr, --energy_band_range (emin,emax)
        energy limits to plot bands, given as a tuple (withou brackets). This value willbe setted
        automatically if plotted band.
    
    -ef, --fermi_energy (value)
        set fermi energy to adjust  x axis of PDOS plot. This value will be setted automatically if
        plotted band.
    
    -sb, --simplebands (input file)
        sets a new bands file (not output from siesta) to be ploted, gnuplot formated.
        
    -sp, --simplepdos (input file)
        sets a new PDOS file (not output from siesta) to be ploted, gnuplot formated.
        
    -o, --output
        sets output file name. If not setted, name in .bands file will be used.
    
    -te, -tight_extra
        turn off the try to adjust the position of high symmetry k-points on band plots of extra
        bands (i.e., the bands that come from -sb option)
        
    -sye, --shift_y_extra
        apply y shift in extra band graph.
        
    -g, --show_gap
        turn on calculation of gap and show the gap text in band plot.
        
    -ap, --add_projection (specie,n,l,m,z[,title])
        Add a desired quantum state to projected in the plot. You can use flag as long as you want.
        To use this you have to provide each number in order separated by comma and surrounded by brackets.
        The orders are:
        
         index   key           meaning             possible values
         ================================================================
           1    specie     specie label            Same as in block 
                                                   ChemicalSpeciesLabel
                                                   of fdf file
           2      n        main quantum namber     Any Natural number
           3      l        orbital index           s,p,d,f,g,h
           4      m        magnetization           Depends on l value
           5      z        zetta number            1,2
           6    title      title of label          String  (optionally)
           
        In order to do not specify any one of this numbers you may substitute it for a start (*),
        it will get the contribuctions of all possible numbers.
        If the title is not provided, the script will try to generate one that contains every
        information possible.
        Example: Consider a H_2O system.
          a) To get projections of H atoms we use: H,*,*,*,*
          b) To get projections of p orbitals in O atoms we use: O,*,p,*,*
          c) To get projections of pz orbitals in O atoms we use: O,*,p,0,*
        Remember that:
        
          l | -2 | -1 |  0 |  1 |  2
         ===============================
          s | -- | -- | 0  | -- |  --
          p | -- | py | pz | px |  --
          d | dxy| dyz| dz2| dxz| dx2-y2
          
        Others values are forbiden.
    
    --silece
        hide all messages
    
    -h, --help
        displays this help and exit.
        
AUTHOR
   Lucas Prett Campagna

COPYRIGHT
       Copyright © 2016 Free Software Foundation, Inc.   License  GPLv3+:  GNU
       GPL version 3 or later <http://gnu.org/licenses/gpl.html>.
       This  is  free  software:  you  are free to change and redistribute it.
       There is NO WARRANTY, to the extent permitted by law.
        
                ''')


# In[16]:


silence = False
def printf(s):
    global silence
    if not silence:
        print(s)


# In[13]:


def add_projection(i,v,next_arg):
    daux = {}
    get=lambda u,f=str: None if u == '*' else f(u)
    tostr = lambda u: u if type(u) == str else str(u) if u != None else ''
    if len(next_arg.split(',')) != 5:
        raise IOError('Invalid input: '+next_arg)
    aux=iter(next_arg.split(','))
    daux['species']=get(next(aux))
    daux['n']=get(next(aux),int)
    daux['l']=get(next(aux))
    daux['m']=get(next(aux),int)
    daux['z']=get(next(aux),int)
    
    # allowable values for n
    if daux['n'] < 1 and daux['n']:
        raise IOError('Invalid value n='+str(daux['n']))
        
    # allowable values for l
    if not daux['l'] in ('s','p','d','f','g','h') and daux['l']:
        raise IOError('Invalid value l='+daux['l'])
        
    # allowable values for m
    if daux['m']:
        if daux['l'] == 's':
            if not daux['m'] in (0,):
                raise IOError('Invalid combination of l='+daux['l']+', m='+str(daux['m']))
        if daux['l'] == 'p':
            if not daux['m'] in (-1,0,1):
                raise IOError('Invalid combination of l='+daux['l']+', m='+str(daux['m']))
        if daux['l'] == 'd':
            if not daux['m'] in (-2,-1,0,1,2):
                raise IOError('Invalid combination of l='+daux['l']+', m='+str(daux['m']))
                
    # allowable values for z
    if not daux['z'] in (1,2) and daux['z']:
        raise IOError('Invalid value z='+str(daux['z']))
        
    # getting title
    try:
        daux['title']=get(next(aux))
    except StopIteration:
        title=''.join([tostr(daux['n']),tostr(daux['l']),tostr(daux['m']) if daux['m'] == None else                 ''                                                   if daux['l'] == None else                 {0:''}[daux['m']]                                    if daux['l'] == 's' else                 {-1:'y',0:'z',1:'x'}[daux['m']]                      if daux['l'] == 'p' else                 {-2:'xy',-1:'yz',0:'z2',1:'xz',2:'x2-y2'}[daux['m']] if daux['l'] == 'd' else                 str(daux['m'])])
        if title == '':
                if daux['species']:
                    title=daux['species']
        else:
            if daux['species']:
                title=daux['species']+'-'+title
        
        if daux['z']:
            title+='-'+str(daux['z'])
        daux['title']=title
    return daux


# In[14]:


import sys

def param_select():
    par={}
    for i,v in enumerate(sys.argv):
        if v[0] == '-':
            try:
                next_arg=sys.argv[i+1]
            except:
                next_arg=sys.argv[i]
            if v in ('-b','--bands'):
                par['band_file'] = next_arg
                if not par.has_key('output_file'):
                    par['output_file'] = next_arg.replace('.bands','_bands.png')
            elif v in ('-p','--pdos'):
                par['pdos_file'] = next_arg
                if not par.has_key('output_file'):
                    par['output_file'] = next_arg.replace('.PDOS','_bands.png')
            elif v in ('-f','--fat'):
                par['wfsx_file'] = next_arg
            elif v in ('-ebr','--energy_band_range'):
                par['energy_band_range'] = map(float,next_arg.split(','))
            elif v in ('-ef','--fermi_energy'):
                par['Ef'] = float(next_arg)
            elif v in ('-sb','--simplebands'):
                try:
                    par['simple_band_file'].append(next_arg)
                except KeyError:
                    par['simple_band_file'] = next_arg
                except AttributeError:
                    par['simple_band_file'] = [next_arg]
                    par['simple_band_file'].append(next_arg)
            elif v in ('-sp', '--simplepdos'):
                try:
                    par['simple_pdos_file'].append(next_arg)
                except KeyError:
                    par['simple_pdos_file'] = next_arg
                except AttributeError:
                    par['simple_pdos_file'] = [next_arg]
                    par['simple_pdos_file'].append(next_arg)
            elif v in ('-o','--output'):
                par['output_file'] = next_arg
            elif v in ('-te','--tight_extra'):
                par['tight_extra'] = True
            elif v in ('-sye','--shift_y_extra'):
                par['shift_y_extra'] = float(next_arg)
            elif v in ('-g','--show_gap'):
                par['show_gap'] = True
            elif v in ('-ap','--add_projection'):
                if not par.has_key('projections'):
                    par['projections']=[]
                par['projections'].append(add_projection(i,v,next_arg))
            elif v in ('--silence',):
                global silence
                silence=True
            elif v in ('-h','--help'):
                show_help()
                exit()
    
    if not par.has_key('band_file') and not par.has_key('pdos_file'):
        printf('Tring to find .band and .PDOS files...')
        for fname in os.walk('.').next()[2]:
            if '.bands' == fname[-6:] and not par.has_key('band_file'):
                printf('file .bands found: '+fname)
                par['band_file'] = fname
                if not par.has_key('output_file'):
                    printf('Output set to: '+fname)
                    par['output_file'] = fname.replace('.bands','_bands.png')
            if '.PDOS' == fname[-5:] and not par.has_key('pdos_file'):
                printf('file .PDOS found: '+fname)
                par['pdos_file'] = fname
                if not par.has_key('output_file'):
                    printf('Output set to: '+fname)
                    par['output_file'] = fname.replace('.PDOS','_bands.png')
            if par.has_key('pdos_file') and par.has_key('band_file'):
                break
    if not par.has_key('pdos_file') and not par.has_key('band_file') and not par.has_key('wfsx_file'):
        printf('files .bands and .PDOS not found.')
        show_help()
        exit()
    return par


# In[15]:

# Program 
if __name__ == '__main__':
    par=param_select()
    printf('Reading and parsing files...')
    a=BandDosPlot(par)
#     print par['projections']
#     assert False
    printf('Ploting...')
#     a.plot([{'species':'O','l':'p','title':'Op'},{'species':'Nb','l':'d','title':'Nbd'},{'species':'Nb','l':'p','title':'Nbp'},{'species':'O','l':'d','title':'Od'}])
#     a.plot([{'species':'O'},{'species':'Nb'}])
    if par.has_key('projections'):
        a.plot(par['projections'])
    else:
        a.plot()
    printf('Saving plot: '+par['output_file']+' ...')
    a.save()
    printf('Done!')

