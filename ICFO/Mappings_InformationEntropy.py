
### Libraries

from pyscf import gto, scf, ao2mo
import pyscf
import functools
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
import itertools
from itertools import combinations

from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
from scipy.sparse import csr_matrix


### System Initialization


#########################################   Molecules initialization    #########################################
InitializeIsing=False

InitializeH2=True
InitializeFH=False
InitializeBeH2=False
InitializeN2=False
InitializeBH3=False

InitializeO2=False #not implemented yet

#########################################   Molecules    ########################################################

if InitializeIsing == True:
    pass  #### not implemented yet


if InitializeH2 == True:
    mol = pyscf.M(
                atom = 'H 0 0 0; H 0 0 0.735',  # in Angstrom
                        basis = 'ccpvqz',      #  'sto-3g' 'ccpvdz' 'ccpvtz' 'ccpvqz'
                        symmetry = True,
                        )         
if InitializeFH == True:
    mol = pyscf.M(
                atom = 'H 0 0 0; F 0 0 0.917',  # in Angstrom
                    basis = 'ccpvqz',
                        symmetry = True,
                        )

if InitializeN2 == True:
    mol = pyscf.M(
                atom = 'N 0 0 0; N 0 0 1.0975',  # in Angstrom
                    basis = 'ccpvqz',
                        symmetry = True,
                        )
    
if InitializeBeH2 == True:
    mol = pyscf.M(
                atom = 'H 0 0 -1.413; Be 0 0 0; H 0 0 1.413',  # in Angstrom
                    basis = 'ccpvqz',
                        symmetry = True,
                        )
    
if InitializeBH3 == True:
    xvalue=np.sqrt(3)*1.19/2
    yvalue=1.19/2
    quote= 'B 0 0 0; H 0 0 -1.19; H 0 '+str(xvalue)+' '+str(yvalue)+'; H 0 -'+str(xvalue)+' '+str(yvalue)
    mol = pyscf.M(
                atom = quote,  # in Angstrom
                    basis = 'ccpvqz',
                        symmetry = True,
                        )
    print(quote)
    
    

    
nao=mol.nao


#########################################   do hartree fock scf   ###############################

myhf = mol.HF()
myhf.kernel()


######################################### get HF data   #########################################

'''
# Orbital energies, Mulliken population etc.
myhf.analyze()

orb = myhf.mo_coeff
eri_4fold = ao2mo.kernel(mol, orb)
print('MO integrals (ij|kl) with 4-fold symmetry i>=j, k>=l have shape %s' %
              str(eri_4fold.shape))
'''



######## get one-electron matrix elements in Atomic Orbitals or Molecular Orbitals basis   ######

if InitializeIsing == False:
    hcore_ao = myhf.get_hcore(mol)  
    hcore_mo = functools.reduce(np.dot, (myhf.mo_coeff.conj().T, hcore_ao, myhf.mo_coeff))
else:
    hcore_mo=[]



#################### get two electron elements in AO or MO basis  ###############################


if InitializeIsing == False:
    eris = ao2mo.get_ao_eri(mol)
    eris_ao = ao2mo.restore(1, eris, nao)
    eris2 = ao2mo.get_mo_eri(eris, myhf.mo_coeff)
    eris_mo = ao2mo.restore(1,eris2,nao)
else:
    eris_mo = []
    
print('Built-in molecular orbitals:   ',mol.nao)
print('Total electrons:   ',mol.nelectron)



### MAPPINGS 


#########################################    JW MAPPING FUNCTION   #########################################

def JW_Mapping(OneBT,TwoBT,AO):

    AOnumber=AO #OneBT.shape[0]
    creation=[]
    anihilation=[]

    for FermionicMode in range(AOnumber):
        Creation_String=[]
        Anihilation_String=[]
        for j in range(AOnumber):
            if j<FermionicMode:
                Creation_String.append(sigmaz())
                Anihilation_String.append(sigmaz())
            elif j==FermionicMode:
                Creation_String.append(sigmam())
                Anihilation_String.append(sigmap())
            elif j>FermionicMode:
                Creation_String.append(qeye(2))
                Anihilation_String.append(qeye(2))

        creation.append(tensor(Creation_String))
        anihilation.append(tensor(Anihilation_String))

    
    hamiltonian=creation[0]*0                 #np.zeros((2**AOnumber,2**AOnumber)))

    for i in range(AOnumber):
        print('row ', i, ' / ',AOnumber)
        for j in range(AOnumber):
            hamiltonian += OneBT[i][j]*creation[i]*anihilation[j]
            for k in range(AOnumber):
                for l in range(AOnumber):
                    hamiltonian += TwoBT[i][j][k][l]*creation[i]*creation[j]*anihilation[k]*anihilation[l]
    print('JW Hamiltonian created. Type: ', type(hamiltonian))

    return hamiltonian



#########################################    PARITY MAPPING FUNCTION   #########################################



def Parity_Mapping(OneBT,TwoBT,AO):

    AOnumber=AO
    creation=[]
    anihilation=[]
    Pplus=tensor([sigmam(),(qeye(2)+sigmaz())/2])-tensor([sigmap(),(qeye(2)-sigmaz())/(2)])
    Pminus=tensor([sigmap(),(qeye(2)+sigmaz())/2])-tensor([sigmam(),(qeye(2)-sigmaz())/(2)])

    for FermionicMode in range(AOnumber):
        print('JW Fermionic mode= ', FermionicMode)
        Creation_Op=Qobj([[1]])
        Anihilation_Op=Qobj([[1]])
        for j in range(AOnumber):
            if j==0:
                if FermionicMode==0:
                    Creation_Op=sigmam()
                    Anihilation_Op=sigmap()
                elif FermionicMode==1:
                    Creation_Op=Pplus
                    Anihilation_Op=Pminus
                else:
                    Creation_Op=qeye(2)
                    Anihilation_Op=qeye(2)
            elif j==FermionicMode-1:
                Creation_Op=tensor([Pplus,Creation_Op])
                Anihilation_Op=tensor([Pminus,Anihilation_Op])
                
            elif j==FermionicMode:
                continue
                
            elif j < FermionicMode-1:
                Creation_Op=tensor([qeye(2),Creation_Op])
                Anihilation_Op=tensor([qeye(2),Anihilation_Op])
            elif j > FermionicMode:
                Creation_Op=tensor([sigmax(),Creation_Op])
                Anihilation_Op=tensor([sigmax(),Anihilation_Op])
                

        creation.append(Creation_Op)
        anihilation.append(Anihilation_Op)


    hamiltonian=creation[0]*0                 #np.zeros((2**AOnumber,2**AOnumber)))

    for i in range(AOnumber):
        for j in range(AOnumber):
            hamiltonian += OneBT[i][j]*creation[i]*anihilation[j]
            for k in range(AOnumber):
                for l in range(AOnumber):
                    hamiltonian += TwoBT[i][j][k][l]*creation[i]*creation[j]*anihilation[k]*anihilation[l]
        
    print('Parity Hamiltonian created. Type: ', type(hamiltonian))

    return hamiltonian



#########################################    Bravyi-Kitaev MAPPING FUNCTION   #########################################

def BK_Mapping(OneBT,TwoBT,AO):
    AOnumber=AO
    if int(np.log2(AOnumber))-np.log2(AOnumber) != 0:
        print('AOnumber error')

    TransformationMatrix=np.ones((1,1))
    for i in range(int(np.log2(AOnumber))):
        CurrentMatrix=np.zeros((2**(i+1),2**(i+1)))
        CurrentMatrix[:int(2**(i)),:int(2**(i))]=TransformationMatrix
        CurrentMatrix[int(2**(i)):,int(2**(i)):]=TransformationMatrix
        CurrentMatrix[-1]=np.ones(2**(i+1))
        TransformationMatrix=CurrentMatrix

    LowTriangMatrix=np.tril(np.ones([AOnumber,AOnumber]),k=-1)
    InvertedTransformationMatrix=np.linalg.inv(TransformationMatrix)

    ParityMatrix=np.matmul(LowTriangMatrix,InvertedTransformationMatrix) #Row i non-zero entries give the parity set for qubit i
    UpdateMatrix=TransformationMatrix  #Column j non-zero entries > from j give the update set for qubit j
    TransposedUpdateMatrix=np.transpose(UpdateMatrix)
    First_SetsMatrix=TransposedUpdateMatrix-ParityMatrix #1 correspond to update set, -1 to parity set. Diagonal 1s do not count
    SetsMatrix=First_SetsMatrix  

    for i in range(AOnumber):
        for j in range(AOnumber):
            if SetsMatrix[i][j] == -1:             #Condition for parity set
                if TransformationMatrix[i][j]==1:  #Condition for flip set
                    SetsMatrix[i][j]=2
                else:                              #Reminder Set
                    SetsMatrix[i][j]=3
    print(SetsMatrix)


    #Sets matrix 1 correspond to update set, 2 to flip set,3 to reminder set. Diagonal 1s do not count.

    creation=[]
    anihilation=[]

    for FermionicMode in range(AOnumber):

        OddEven=FermionicMode%2 #1if odd, 0 if even

        C1=Qobj([[1]])
        C2=Qobj([[1]])
        A1=Qobj([[1]])
        A2=Qobj([[1]])

        CurrentSet=SetsMatrix[FermionicMode]
        for j in range(AOnumber):
            if j==0:
                if FermionicMode==0:
                    C1=sigmax()
                    C2=sigmay()
                    A1=sigmax()
                    A2=sigmay()
                else:
                    if CurrentSet[0]==0:
                        C1=qeye(2)
                        C2=qeye(2)
                        A1=qeye(2)
                        A2=qeye(2)
                    else:
                        if OddEven==0:
                            if CurrentSet[0]==2 or CurrentSet[0]==3:
                                C1=sigmaz()
                                C2=sigmaz()
                                A1=sigmaz()
                                A2=sigmaz()

                        elif OddEven==1:
                            if CurrentSet[0]==2:
                                C1=sigmaz()
                                C2=qeye(2)
                                A1=sigmaz()
                                A2=qeye(2)
                            elif CurrentSet[0]==3:
                                C1=sigmaz()
                                C2=sigmaz()
                                A1=sigmaz()
                                A2=sigmaz() 

            elif j==FermionicMode:
                C1=tensor([sigmax(),C1])
                C2=tensor([sigmay(),C2])
                A1=tensor([sigmax(),A1])
                A2=tensor([sigmay(),A2])

            elif CurrentSet[j]==0:
                C1=tensor([qeye(2),C1])
                C2=tensor([qeye(2),C2])
                A1=tensor([qeye(2),A1])
                A2=tensor([qeye(2),A2])

            elif CurrentSet[j]==1:
                C1=tensor([sigmax(),C1])
                C2=tensor([1j*sigmax(),C2])
                A1=tensor([sigmax(),A1])
                A2=tensor([1j*sigmax(),A2])

            elif CurrentSet[j]==2:
                C1=tensor([sigmaz(),C1])
                A1=tensor([sigmaz(),A1])
                if OddEven==0:
                    C2=tensor([sigmaz(),C2])
                    A2=tensor([sigmaz(),A2])
                elif OddEven==1:
                    C2=tensor([qeye(2),C2])
                    A2=tensor([qeye(2),A2])

            elif CurrentSet[j]==3:
                C1=tensor([sigmaz(),C1])
                A1=tensor([sigmaz(),A1])
                if OddEven==0:
                    C2=tensor([sigmaz(),C2])
                    A2=tensor([sigmaz(),A2])
                elif OddEven==1:
                    C2=tensor([sigmaz(),C2])
                    A2=tensor([sigmaz(),A2])

        creation.append(0.5*(C1-C2))
        anihilation.append(0.5*(A1+A2))
    
                
        
    hamiltonian=creation[0]*0                 #np.zeros((2**AOnumber,2**AOnumber)))

    for i in range(AOnumber):
        for j in range(AOnumber):
            hamiltonian += OneBT[i][j]*creation[i]*anihilation[j]
            for k in range(AOnumber):
                for l in range(AOnumber):
                    hamiltonian += TwoBT[i][j][k][l]*creation[i]*creation[j]*anihilation[k]*anihilation[l]

    print('BK Hamiltonian created. Type: ', type(hamiltonian))
    
    return hamiltonian
    


### Mapping Entanglement Calculations

nao=8

JW_Matrix=JW_Mapping(hcore_mo,eris_mo,nao)
Parity_Matrix=Parity_Mapping(hcore_mo,eris_mo,nao)
BK_Matrix=BK_Mapping(hcore_mo,eris_mo,nao)



LoadGS=False
nao=8
Molecule_SL='H2'   #H2, HF, BH3, BeH2, N2, O2
Basis_SL='ccpvqz'         #ccpvdz, ccpvtz, ccpvqz

SaveGS=False
SL_GSPath="/aloy/home/tparella/AnchovieIcecream/molecules/"+Molecule_SL+'/'

if LoadGS==False:
    JW_eigval,JW_eigvec=JW_Matrix.groundstate(sparse=True)
    Parity_eigval,Parity_eigvec=Parity_Matrix.groundstate(sparse=True)
    BK_eigval,BK_eigvec=BK_Matrix.groundstate(sparse=True)
    if SaveGS==True:
        qsave(JW_eigvec, SL_GSPath+str(nao)+'q_JW_'+Basis_SL)
        qsave(Parity_eigvec,  SL_GSPath+str(nao)+'q_Parity_'+Basis_SL)
        qsave(BK_eigvec,  SL_GSPath+str(nao)+'q_BK_'+Basis_SL)
        
elif LoadGS==True:
    JW_eigvec=qload(SL_GSPath+str(nao)+'q_JW_'+Basis_SL)
    Parity_eigvec=qload(SL_GSPath+str(nao)+'q_Parity_'+Basis_SL)
    BK_eigvec=qload(SL_GSPath+str(nao)+'q_BK_'+Basis_SL)
    
print(JW_eigvec)


def ptracealt(rho,qkeep) :
    rd = rho.dims[0]
    nd = len(rd)
    qkeep = list(np.sort(qkeep))
    dkeep = (np.array(rd)[qkeep]).tolist()
    qtrace = list(set(np.arange(nd))-set(qkeep))
    dtrace = (np.array(rd)[qtrace]).tolist()
    if isket(rho) :
        vmat = (rho.full()
                .reshape(rd)
                .transpose(qkeep+qtrace)
                .reshape([np.prod(dkeep),np.prod(dtrace)]))
        rhomat = vmat.dot(vmat.conj().T)
    else :
        rhomat = np.trace(rho.full()
                          .reshape(rd+rd)
                          .transpose(qtrace+[nd+q for q in qtrace]+qkeep+[nd+q for q in qkeep])
                          .reshape([prod(dtrace),prod(dtrace),prod(dkeep),prod(dkeep)]))
    return Qobj(rhomat,dims=[dkeep, dkeep])


permutations_max=4  #includes all permutations of (2,permutations_max) qubits  #int(nao/2+1) includes all permutations

MO_Numbers=[i for i in range(nao)]

JW_Entropy=[[] for i in range(1,permutations_max)]
Parity_Entropy=[[] for i in range(1,permutations_max)]
BK_Entropy=[[] for i in range(1,permutations_max)]


for k in range(2,permutations_max+1):
    print(k)
    
    Available_combinations= list(itertools.combinations(MO_Numbers,k))

    
    for i in Available_combinations:
        
        JW_GSTraced=ptracealt(JW_eigvec,list(i))     #JW_GroundState.ptrace(list(i))
        JW_Entropy[k-2].append(entropy_vn(JW_GSTraced))
        
        Parity_GSTraced=ptracealt(Parity_eigvec,list(i))     #JW_GroundState.ptrace(list(i))
        Parity_Entropy[k-2].append(entropy_vn(Parity_GSTraced))
        
        BK_GSTraced=ptracealt(BK_eigvec,list(i))     #JW_GroundState.ptrace(list(i))
        BK_Entropy[k-2].append(entropy_vn(BK_GSTraced))
        


### Plot entropies for bipartitions

Boxplot=True
Violinplot=True
Scatterplot=True

Ylim=.5

###########################     PLOTS     #############################################################

if Boxplot==True:
    
    x_coordiantes= [i for i in range(2,permutations_max+1)]
    
    JW_x_coordinates=[i-0.25 for i in range(2,permutations_max+1)]
    Parity_x_coordinates=[i for i in range(2,permutations_max+1)]
    BK_x_coordinates=[i+0.25 for i in range(2,permutations_max+1)]

            
        


    fig, ax = plt.subplots()

    JW_Box = ax.boxplot(JW_Entropy, positions=JW_x_coordinates, widths=.2, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C3", "edgecolor": "white",
                              "linewidth": 0.5,"alpha":1},
                    whiskerprops={"color": "C3", "linewidth": 1.5},
                    capprops={"color": "C3", "linewidth": 1.5})


    BK_Box = ax.boxplot(BK_Entropy, positions=BK_x_coordinates, widths=.2, patch_artist=True,
                    showmeans=False, showfliers=False,
                    medianprops={"color": "white", "linewidth": 0.5},
                    boxprops={"facecolor": "C0", "edgecolor": "white",
                              "linewidth": 0.5,"alpha":1},
                    whiskerprops={"color": "C0", "linewidth": 1.5},
                    capprops={"color": "C0", "linewidth": 1.5})
    
    Parity_Box = ax.boxplot(Parity_Entropy, positions=Parity_x_coordinates, widths=.2, patch_artist=True,
                showmeans=False, showfliers=False,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C2", "edgecolor": "white",
                              "linewidth": 0.5,"alpha":1},
                whiskerprops={"color": "C2", "linewidth": 1.5},
                capprops={"color": "C2", "linewidth": 1.5})
    

    ax.legend([JW_Box["boxes"][0],Parity_Box["boxes"][0],BK_Box["boxes"][0]], ['JW','Parity','BK'], loc='upper left')

    plt.xlabel('Qubit number bipartition')
    plt.ylabel('Entanglement entropy')
    
    ax.set(xlim=(1.5, 6.5), xticks=[2,3,4,5,6],
           ylim=(0, Ylim), yticks=[0,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.2, 0.25,0.5,0.75,1])

    plt.show()
    
    
    
    
    
    
    
if Violinplot==True:
    fig, ax = plt.subplots()

    JW_Violin = ax.violinplot(JW_Entropy, x_coordiantes, widths=.9,
                       showmeans=False, showmedians=False, showextrema=False)
    # styling:
    for body in JW_Violin['bodies']:
        body.set_facecolor('red')
        body.set_alpha(0.3)
    

    Parity_Violin = ax.violinplot(Parity_Entropy, x_coordiantes, widths=.9,
                       showmeans=False, showmedians=False, showextrema=False)
    # styling:
    for body in Parity_Violin['bodies']:
        body.set_facecolor('green')
        body.set_alpha(0.3)

    
    BK_Violin = ax.violinplot(BK_Entropy, x_coordiantes, widths=.9,
                       showmeans=False, showmedians=False, showextrema=False)
    # styling:
    for body in BK_Violin['bodies']:
        body.set_facecolor('blue')
        body.set_alpha(0.3)
    ax.set(xlim=(1, 5), xticks=np.arange(2, permutations_max+1),
           ylim=(0, Ylim), yticks=[0,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.2, 0.25,0.3,0.35,0.4,0.45,0.5,0.75,1])
    
    
    
    plt.xlabel('Qubit number bipartition')
    plt.ylabel('Entanglement entropy')

    plt.show()
    



print(len(JW_Entropy[2])) #28,56,70


if Scatterplot == True:
    alphanum=0.005
    x_coordiantes= [i for i in range(2,permutations_max+1)]
    
    JW_x_coordinates=[i-0.25 for i in range(2,permutations_max+1)]
    Parity_x_coordinates=[i for i in range(2,permutations_max+1)]
    BK_x_coordinates=[i+0.25 for i in range(2,permutations_max+1)]

    fig, ax = plt.subplots()
    
    for k in range(len(JW_x_coordinates)):
        entr_coord=[]
        for i in range(len(JW_Entropy[k])):
            entr_coord.append(JW_x_coordinates[k])
        ax.scatter(entr_coord, JW_Entropy[k], color='red', alpha=alphanum)
        
    for k in range(len(Parity_x_coordinates)):
        entr_coord=[]
        for i in range(len(Parity_Entropy[k])):
            entr_coord.append(Parity_x_coordinates[k])
        ax.scatter(entr_coord, Parity_Entropy[k], color='green', alpha=alphanum)
        
    for k in range(len(BK_x_coordinates)):
        entr_coord=[]
        for i in range(len(BK_Entropy[k])):
            entr_coord.append(BK_x_coordinates[k])
        ax.scatter(entr_coord, BK_Entropy[k], color='blue', alpha=alphanum)
    

    #BK_scatter = ax.scatter(BK_x_coordinates, BK_Entropy)
    
    #Parity_Box = ax.boxplot(Parity_x_coordinates, Parity_Entropy)

    #ax.legend([JW_scatter["boxes"][0],Parity_scatter["boxes"][0],BK_scatter["boxes"][0]], ['JW','Parity','BK'], loc='upper left')

    plt.xlabel('Qubit number bipartition')
    plt.ylabel('Entanglement entropy')
    
    ax.set(xlim=(1.5, 6.5), xticks=[2,3,4,5,6],
           ylim=(0, Ylim), yticks=[0,0.025,0.050,0.075,0.100,0.125,0.150,0.175,0.2, 0.25,0.3,0.35,0.4,0.45,0.5,0.75,1])

    plt.show()



