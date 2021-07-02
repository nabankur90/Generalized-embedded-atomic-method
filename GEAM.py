import os
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class pot_c33:
	pot_c33_value: np.ndarray=np.array([5.0,7.0])
	pot_E_c33_value: np.ndarray=np.array([5.0,7.0])
	pot_Fx_c33_value: np.ndarray=np.array([5.0,7.0])
	pot_Fy_c33_value: np.ndarray=np.array([5.0,7.0])
	pot_Fz_c33_value: np.ndarray=np.array([5.0,7.0])
	jac_c33_value: np.ndarray=np.array([1.0,2.0])

@dataclass
class pot_c32:
	pot_c32_value: np.ndarray=np.array([5.0,7.0])
	pot_E_c32_value: np.ndarray=np.array([5.0,7.0])
	pot_Fx_c32_value: np.ndarray=np.array([5.0,7.0])
	pot_Fy_c32_value: np.ndarray=np.array([5.0,7.0])
	pot_Fz_c32_value: np.ndarray=np.array([5.0,7.0])
	jac_c32_value: np.ndarray=np.array([1.0,2.0])


def read_CONTCAR_OUTCAR(fname1, fname2, n, fpath1, fpath2):
	with open(os.path.join(fpath1, fname1 %n), "r") as f1:
		line=f1.read().splitlines()
	box_mat=np.array([i.split() for i in line[2:5]]).astype("float")
	box_info=np.array([i.split() for i in line[5:7]])
	atomtype=np.array(box_info[0]).astype("str")
	natoms=np.array(box_info[1]).astype("int")
	tot_atoms=np.sum(natoms)
	elements=[]
	if len(natoms)==1:
		for i in range(natoms[0]):
			if atomtype[0]=="W":
				elements.append(0)
			else:
				elements.append(1)
	else:
		for i in range(natoms[0]):
			elements.append(0)
		for i in range(natoms[0],tot_atoms):
			elements.append(1)
	ax=np.array(box_mat[0])
	ay=np.array(box_mat[1])
	az=np.array(box_mat[2])
	A=np.array([i.split() for i in line[8:tot_atoms+8]]).astype("float")
	coords_redu=A[:,0:3].astype("float")
	coords_cart=np.dot(coords_redu,box_mat)
	with open(os.path.join(fpath2, fname2 %n), "r") as f2:
		for num,line in enumerate(f2, 1):
			if  "TOTAL-FORCE" in line:
				p=num
				break
	f2=open(os.path.join(fpath2, fname2 %n))
	l=f2.read().splitlines()[p+1:p+tot_atoms+1]
	B=np.array([o.split() for o in l])
	Force=B[:,3:6].astype("float")	
	return [elements, box_mat, coords_redu, Force]

def distance(box_mat,A,p,q):
	dr=np.array([(A[p,0]-A[q,0]),(A[p,1]-A[q,1]),(A[p,2]-A[q,2])])
	for i in range(len(dr)):
		if dr[i]<-0.50:
			dr[i]+=1
		elif dr[i]>=0.50:
			dr[i]-=1
		else:
			pass
	dR=np.linalg.norm(np.dot(dr,box_mat))
	dx=np.dot(dr,box_mat)[0]
	dy=np.dot(dr,box_mat)[1]
	dz=np.dot(dr,box_mat)[2]
	R=np.array([dR, dx, dy, dz])
	return R

def neigh_list(box_mat,coords_redu,elements,i):
	neigh=[]
	neigh_dis=[]
	for j in range(i+1,len(elements)):
		R=distance(box_mat,coords_redu,i,j)
		if R[0]<r_cutoff:
			neigh.append(j)
			neigh_dis.append(R)	
	return [neigh, neigh_dis]
			

def calculate_gaussian(gamma,a,b,l,R):
	i=a+b
	j=a+b+3
	g_energy=np.exp(-(gamma[i]*(gamma[j]**l)*(R[0]**2)))
	g_force_x=2*(gamma[i]*(gamma[j]**l))*np.exp(-(gamma[i]*(gamma[j]**l)*(R[0]**2)))*R[1]
	g_force_y=2*(gamma[i]*(gamma[j]**l))*np.exp(-(gamma[i]*(gamma[j]**l)*(R[0]**2)))*R[2]
	g_force_z=2*(gamma[i]*(gamma[j]**l))*np.exp(-(gamma[i]*(gamma[j]**l)*(R[0]**2)))*R[3]
	g=np.array([g_energy, g_force_x, g_force_y, g_force_z])
	return g


def two_body(gamma,elements,coords_redu,box_mat):

	A2_E=np.zeros((1,(3*Ng)))
	A2_Fx=np.zeros((len(elements),(3*Ng)))
	A2_Fy=np.zeros((len(elements),(3*Ng)))
	A2_Fz=np.zeros((len(elements),(3*Ng)))
	for i in range(len(elements)):
		neighbors=neigh_list(box_mat,coords_redu,elements,i)
		closest_atoms=neighbors[0]
		R=neighbors[1]
		for j in range(len(closest_atoms)):
			for l in q:
				idx=(elements[i]+elements[closest_atoms[j]])*Ng+l
				gaussian=calculate_gaussian(gamma,elements[i],elements[closest_atoms[j]],l,R[j])
				A2_E[0,idx]+=gaussian[0]
				A2_Fx[i,idx]+=gaussian[1]
				A2_Fx[closest_atoms[j],idx]+=gaussian[1]
				A2_Fy[i,idx]+=gaussian[2]
				A2_Fy[closest_atoms[j],idx]+=gaussian[2]
				A2_Fz[i,idx]+=gaussian[3]
				A2_Fz[closest_atoms[j],idx]+=gaussian[3]

	return [A2_E, A2_Fx, A2_Fy, A2_Fz]


def three_body_two_bond(c32,elements,coords_redu,box_mat):
	A32_E=np.zeros((1))
	A32_Fx=np.zeros((len(elements)))
	A32_Fy=np.zeros((len(elements)))
	A32_Fz=np.zeros((len(elements)))
	J32_E=np.zeros((1,(3*Ng)))
	J32_Fx=np.zeros((len(elements),(3*Ng)))
	J32_Fy=np.zeros((len(elements),(3*Ng)))
	J32_Fz=np.zeros((len(elements),(3*Ng)))
	
	for i in range(len(elements)):
		neighbors=neigh_list(box_mat,coords_redu,elements,i)
		closest_atoms=neighbors[0]
		R=neighbors[1]
		for j in range(len(closest_atoms)):
			for k in range(j+1,len(closest_atoms)):
				R_jk=distance(box_mat,coords_redu,closest_atoms[j],closest_atoms[k])
				V_ij=np.zeros((4,(3*Ng)))
				V_ik=np.zeros((4,(3*Ng)))
				V_jk=np.zeros((4,(3*Ng)))
				for m in q:
					idx_ij=(elements[i]+elements[closest_atoms[j]])*Ng+m
					idx_ik=(elements[i]+elements[closest_atoms[k]])*Ng+m
					idx_jk=(elements[closest_atoms[j]]+elements[closest_atoms[k]])*Ng+m
					gaussian_ij=calculate_gaussian(gamma,elements[i],elements[closest_atoms[j]],m,R[j])
					gaussian_ik=calculate_gaussian(gamma,elements[i],elements[closest_atoms[k]],m,R[k])
					gaussian_jk=calculate_gaussian(gamma,elements[closest_atoms[j]],elements[closest_atoms[k]],m,R_jk)
					V_ij[:,idx_ij]=gaussian_ij[:]
					V_ik[:,idx_ik]=gaussian_ik[:]
					V_jk[:,idx_jk]=gaussian_jk[:]
				A32_E[0]+=(np.dot(V_ij[0,:],c32)*np.dot(V_ik[0,:],c32))
				A32_Fx[i]+=(np.dot(V_ij[1,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[1,:],c32))
				A32_Fy[i]+=(np.dot(V_ij[2,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[2,:],c32))
				A32_Fz[i]+=(np.dot(V_ij[3,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[3,:],c32))
				J32_E[0,:]+=(V_ij[0,:]*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*V_ik[0,:])
				J32_Fx[i,:]+=(V_ij[1,:]*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[1,:],c32)*V_ik[0,:])+(V_ij[0,:]*np.dot(V_ik[1,:],c32))+(np.dot(V_ij[0,:],c32)*V_ik[1,:])
				J32_Fy[i,:]+=(V_ij[2,:]*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[2,:],c32)*V_ik[0,:])+(V_ij[0,:]*np.dot(V_ik[2,:],c32))+(np.dot(V_ij[0,:],c32)*V_ik[2,:])
				J32_Fz[i,:]+=(V_ij[3,:]*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[3,:],c32)*V_ik[0,:])+(V_ij[0,:]*np.dot(V_ik[3,:],c32))+(np.dot(V_ij[0,:],c32)*V_ik[3,:])
				if R_jk[0]<r_cutoff:
					A32_E[0]+=(np.dot(V_ij[0,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[0,:],c32)*np.dot(V_jk[0,:],c32))
					A32_Fx[closest_atoms[j]]+=(np.dot(V_ij[1,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_jk[1,:],c32))
					A32_Fy[closest_atoms[j]]+=(np.dot(V_ij[2,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_jk[2,:],c32))
					A32_Fz[closest_atoms[j]]+=(np.dot(V_ij[3,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_jk[3,:],c32))					
					A32_Fx[closest_atoms[k]]+=(np.dot(V_ik[1,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[0,:],c32)*np.dot(V_jk[1,:],c32))
					A32_Fy[closest_atoms[k]]+=(np.dot(V_ik[2,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[0,:],c32)*np.dot(V_jk[2,:],c32))
					A32_Fz[closest_atoms[k]]+=(np.dot(V_ik[3,:],c32)*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[0,:],c32)*np.dot(V_jk[3,:],c32))
					J32_E[0,:]+=(V_ij[0,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[0,:],c32)*V_jk[0,:])+(V_ik[0,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[0,:],c32)*V_jk[0,:])
					J32_Fx[closest_atoms[j],:]+=(V_ij[1,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[1,:],c32)*V_jk[0,:])+(V_ij[0,:]*np.dot(V_jk[1,:],c32))+(np.dot(V_ij[0,:],c32)*V_jk[1,:])
					J32_Fy[closest_atoms[j],:]+=(V_ij[2,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[2,:],c32)*V_jk[0,:])+(V_ij[0,:]*np.dot(V_jk[2,:],c32))+(np.dot(V_ij[0,:],c32)*V_jk[2,:])
					J32_Fz[closest_atoms[j],:]+=(V_ij[3,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ij[3,:],c32)*V_jk[0,:])+(V_ij[0,:]*np.dot(V_jk[3,:],c32))+(np.dot(V_ij[0,:],c32)*V_jk[3,:])
					J32_Fx[closest_atoms[k],:]+=(V_ik[1,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[1,:],c32)*V_jk[0,:])+(V_ik[0,:]*np.dot(V_jk[1,:],c32))+(np.dot(V_ik[0,:],c32)*V_jk[1,:])
					J32_Fy[closest_atoms[k],:]+=(V_ik[2,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[2,:],c32)*V_jk[0,:])+(V_ik[0,:]*np.dot(V_jk[2,:],c32))+(np.dot(V_ik[0,:],c32)*V_jk[2,:])
					J32_Fz[closest_atoms[k],:]+=(V_ik[3,:]*np.dot(V_jk[0,:],c32))+(np.dot(V_ik[3,:],c32)*V_jk[0,:])+(V_ik[0,:]*np.dot(V_jk[3,:],c32))+(np.dot(V_ik[0,:],c32)*V_jk[3,:])
				
	return [A32_E, A32_Fx, A32_Fy, A32_Fz, J32_E, J32_Fx, J32_Fy, J32_Fz]


def three_body_three_bond(c33,elements,coords_redu,box_mat):

	A33_E=np.zeros((1))
	A33_Fx=np.zeros((len(elements)))
	A33_Fy=np.zeros((len(elements)))
	A33_Fz=np.zeros((len(elements)))
	J33_E=np.zeros((1,(3*Ng)))
	J33_Fx=np.zeros((len(elements),(3*Ng)))
	J33_Fy=np.zeros((len(elements),(3*Ng)))
	J33_Fz=np.zeros((len(elements),(3*Ng)))

	for i in range(len(elements)):
		neighbors=neigh_list(box_mat,coords_redu,elements,i)
		closest_atoms=neighbors[0]
		R=neighbors[1]
		for j in range(len(closest_atoms)):
			for k in range(j+1,len(closest_atoms)):
				R_jk=distance(box_mat,coords_redu,closest_atoms[j],closest_atoms[k])
				if R_jk[0]<r_cutoff:
					V_ij=np.zeros((4,(3*Ng)))
					V_ik=np.zeros((4,(3*Ng)))
					V_jk=np.zeros((4,(3*Ng)))
					for m in q:
						idx_ij=(elements[i]+elements[closest_atoms[j]])*Ng+m
						idx_ik=(elements[i]+elements[closest_atoms[k]])*Ng+m
						idx_jk=(elements[closest_atoms[j]]+elements[closest_atoms[k]])*Ng+m
						gaussian_ij=calculate_gaussian(gamma,elements[i],elements[closest_atoms[j]],m,R[j])
						gaussian_ik=calculate_gaussian(gamma,elements[i],elements[closest_atoms[k]],m,R[k])
						gaussian_jk=calculate_gaussian(gamma,elements[closest_atoms[j]],elements[closest_atoms[k]],m,R_jk)
						V_ij[:,idx_ij]=gaussian_ij[:]
						V_ik[:,idx_ik]=gaussian_ik[:]
						V_jk[:,idx_jk]=gaussian_jk[:]
					A33_E[0]+=(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))
					A33_Fx[i]+=(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))
					A33_Fy[i]+=(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))
					A33_Fz[i]+=(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))
					A33_Fx[closest_atoms[j]]+=(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))
					A33_Fy[closest_atoms[j]]+=(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))
					A33_Fz[closest_atoms[j]]+=(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))
					A33_Fx[closest_atoms[k]]+=(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))
					A33_Fy[closest_atoms[k]]+=(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))
					A33_Fz[closest_atoms[k]]+=(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33)+np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))
					J33_E[0,:]+=((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))
					J33_Fx[i,:]+=((V_ij[1,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[1,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[1,:]))		
					J33_Fy[i,:]+=((V_ij[2,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[2,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[2,:]))
					J33_Fz[i,:]+=((V_ij[3,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[3,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[3,:]))
					J33_Fx[closest_atoms[j],:]+=((V_ij[1,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[1,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[1,:]))
					J33_Fy[closest_atoms[j],:]+=((V_ij[2,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[2,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[2,:]))
					J33_Fz[closest_atoms[j],:]+=((V_ij[3,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[3,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[3,:]))
					J33_Fx[closest_atoms[k],:]+=((V_ij[1,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[1,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[1,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[1,:]))
					J33_Fy[closest_atoms[k],:]+=((V_ij[2,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[2,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[2,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[2,:]))
					J33_Fz[closest_atoms[k],:]+=((V_ij[3,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*V_ik[0,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[3,:]*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*V_jk[0,:]))+((V_ij[0,:]*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*V_ik[0,:]*np.dot(V_jk[3,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*V_jk[3,:]))


	return [A33_E, A33_Fx, A33_Fy, A33_Fz, J33_E, J33_Fx, J33_Fy, J33_Fz]


def potential_three_body_two_bond(c32):
	A32=np.zeros((75666))
	A32_E=np.zeros((Ns))
	A32_Fx=np.zeros((75606))
	A32_Fy=np.zeros((75606))
	A32_Fz=np.zeros((75606))
	J32=np.zeros((75666,(3*Ng)))
	J32_E=np.zeros((Ns,(3*Ng)))
	J32_Fx=np.zeros((75606,(3*Ng)))
	J32_Fy=np.zeros((75606,(3*Ng)))
	J32_Fz=np.zeros((75606,(3*Ng)))
	a=0
	b=0
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,fname2,(i+1),fpath1,fpath2)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		V=three_body_two_bond(c32,elements,coords_redu,box_mat)
		A32[a]=V[0]
		A32[a+1:a+1+len(elements)]=V[1]
		A32[a+1+len(elements):a+1+(2*len(elements))]=V[2]
		A32[a+1+(2*len(elements)):a+1+(3*len(elements))]=V[3]
		A32_E[i]=V[0]
		A32_Fx[b:b+len(elements)]=V[1]
		A32_Fy[b:b+len(elements)]=V[2]
		A32_Fz[b:b+len(elements)]=V[3]
		J32[a,:]=V[4]
		J32[a+1:a+1+len(elements),:]=V[5]
		J32[a+1+len(elements):a+1+(2*len(elements)),:]=V[6]
		J32[a+1+(2*len(elements)):a+1+(3*len(elements)),:]=V[7]
		a=a+(3*len(elements))+1
		b=b+len(elements)
	return [A32, A32_E, A32_Fx, A32_Fy, A32_Fz, J32]


def potential_three_body_three_bond(c33):
	A33=np.zeros((75666))
	A33_E=np.zeros((Ns))
	A33_Fx=np.zeros((75606))
	A33_Fy=np.zeros((75606))
	A33_Fz=np.zeros((75606))
	J33=np.zeros((75666,(3*Ng)))
	J33_E=np.zeros((Ns,(3*Ng)))
	J33_Fx=np.zeros((75606,(3*Ng)))
	J33_Fy=np.zeros((75606,(3*Ng)))
	J33_Fz=np.zeros((75606,(3*Ng)))
	a=0
	b=0
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,fname2,(i+1),fpath1,fpath2)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		V=three_body_two_bond(c33,elements,coords_redu,box_mat)
		A33[a]=V[0]
		A33[a+1:a+1+len(elements)]=V[1]
		A33[a+1+len(elements):a+1+(2*len(elements))]=V[2]
		A33[a+1+(2*len(elements)):a+1+(3*len(elements))]=V[3]
		A33_E[i]=V[0]
		A33_Fx[b:b+len(elements)]=V[1]
		A33_Fy[b:b+len(elements)]=V[2]
		A33_Fz[b:b+len(elements)]=V[3]
		J33[a,:]=V[4]
		J33[a+1:a+1+len(elements),:]=V[5]
		J33[a+1+len(elements):a+1+(2*len(elements)),:]=V[6]
		J33[a+1+(2*len(elements)):a+1+(3*len(elements)),:]=V[7]
		a=a+(3*len(elements))+1
		b=b+len(elements)
	return [A33, A33_E, A33_Fx, A33_Fy, A33_Fz, J33]


def cost_func_33(c33,D,c32,pot_c33_class):
	calc_pot_c33(c33, pot_c33_class)
	V33=pot_c33_class.pot_c33_value
	global Nstep
	V=np.dot(A2,D[0:30])+V32+V33
	V21_E=np.dot(A2_E,D[0:30])
	V21_Fx=np.dot(A2_Fx,D[0:30])
	V21_Fy=np.dot(A2_Fy,D[0:30])
	V21_Fz=np.dot(A2_Fz,D[0:30])
	V33_E=pot_c33_class.pot_E_c33_value
	V33_Fx=pot_c33_class.pot_Fx_c33_value
	V33_Fy=pot_c33_class.pot_Fy_c33_value
	V33_Fz=pot_c33_class.pot_Fz_c33_value
	V_E=V21_E+V32_E+V33_E
	V_Fx=V21_Fx+V32_Fx+V33_Fx
	V_Fy=V21_Fy+V32_Fy+V33_Fy
	V_Fz=V21_Fz+V32_Fz+V33_Fz
	r=np.sum((E_F-V)**2)
	RMSE=np.sqrt(r/len(E_F))
	f1=open("error_three_body_three_bond.dat","a")
	f1.write("%6d %20.5f \n" % (Nstep, RMSE))
	a=0
	f2=open("iteration_three_body_three_bond_energy.dat","a")
	f3=open("iteration_three_body_three_bond_force_x.dat","a")
	f4=open("iteration_three_body_three_bond_force_y.dat","a")
	f5=open("iteration_three_body_three_bond_force_z.dat","a")
	f6=open("c33_vector.dat","a")
	for i in range(Ns):
		f2.write("%6d %20.5f %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), E[i], V_E[i], V21_E[i], V32_E[i], V33_E[i]))
		f3.write("%6d %20.5f %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fx[a:a+Natoms[i]]), np.sum(V_Fx[a:a+Natoms[i]]), np.sum(V21_Fx[a:a+Natoms[i]]), np.sum(V32_Fx[a:a+Natoms[i]]), np.sum(V33_Fx[a:a+Natoms[i]])))
		f4.write("%6d %20.5f %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fy[a:a+Natoms[i]]), np.sum(V_Fy[a:a+Natoms[i]]), np.sum(V21_Fy[a:a+Natoms[i]]), np.sum(V32_Fy[a:a+Natoms[i]]), np.sum(V33_Fy[a:a+Natoms[i]])))
		f5.write("%6d %20.5f %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fz[a:a+Natoms[i]]), np.sum(V_Fz[a:a+Natoms[i]]), np.sum(V21_Fz[a:a+Natoms[i]]), np.sum(V32_Fz[a:a+Natoms[i]]), np.sum(V33_Fz[a:a+Natoms[i]])))
		a=a+Natoms[i]
	f6.write(str(c33)+"\n")
	f2.write("\n")
	f3.write("\n")
	f4.write("\n")
	f5.write("\n")
	Nstep=Nstep+1
	return r

def jac_cf_33(c33,D,c32,pot_c33_class):
	J33=pot_c33_class.jac_c33_value
	V33=pot_c33_class.pot_c33_value
	R=E-(np.dot(A2,D[0:30])+V32+V33)
	J=(-1)*J33
	Dr=2*np.dot(J.T,R)
	return Dr


def cost_func_32(c32,D,pot_c32_class):
	calc_pot_c32(c32, pot_c32_class)
	V32=pot_c32_class.pot_c32_value
	global Niter
	V=np.dot(A2,D[0:30])+V32
	V21_E=np.dot(A2_E,D[0:30])
	V32_E=pot_c32_class.pot_E_c32_value
	V_E=V21_E+V32_E
	V21_Fx=np.dot(A2_Fx,D[0:30])
	V32_Fx=pot_c32_class.pot_Fx_c32_value
	V_Fx=V21_Fx+V32_Fx
	V21_Fy=np.dot(A2_Fy,D[0:30])
	V32_Fy=pot_c32_class.pot_Fy_c32_value
	V_Fy=V21_Fy+V32_Fy
	V21_Fz=np.dot(A2_Fz,D[0:30])
	V32_Fz=pot_c32_class.pot_Fz_c32_value
	V_Fz=V21_Fz+V32_Fz
	r=np.sum((E_F-V)**2)
	RMSE_total=np.sqrt(r/len(E_F))
	f1=open("error_three_body_two_bond.dat","a")
	f1.write("%6d %20.5f \n" % (Niter, RMSE_total))
	f2=open("iteration_plot_energies_21_32.dat","a")
	f3=open("iteration_plot_forcex_21_32.dat","a")
	f4=open("iteration_plot_forcey_21_32.dat","a")
	f5=open("iteration_plot_forcez_21_32.dat","a")	
	f6=open("c32_vector.dat","a")
	a=0
	for i in range(Ns):
		f2.write("%6d %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), E[i], V_E[i], V21_E[i], V32_E[i]))
		f3.write("%6d %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fx[a:a+Natoms[i]]), np.sum(V_Fx[a:a+Natoms[i]]), np.sum(V21_Fx[a:a+Natoms[i]]), np.sum(V32_Fx[a:a+Natoms[i]])))
		f4.write("%6d %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fy[a:a+Natoms[i]]), np.sum(V_Fy[a:a+Natoms[i]]), np.sum(V21_Fy[a:a+Natoms[i]]), np.sum(V32_Fy[a:a+Natoms[i]])))
		f5.write("%6d %20.5f %20.5f %20.5f %20.5f \n" % ((i+1), np.sum(Fz[a:a+Natoms[i]]), np.sum(V_Fz[a:a+Natoms[i]]), np.sum(V21_Fz[a:a+Natoms[i]]), np.sum(V32_Fz[a:a+Natoms[i]])))		
		a=a+Natoms[i]
	f6.write(str(c32)+"\n")
	f2.write("\n")
	f3.write("\n")
	f4.write("\n")
	f5.write("\n")
	Niter=Niter+1
	return r

def jac_cf_32(c32,D,pot_c32_class):
	V32=pot_c32_class.pot_c32_value
	J32=pot_c32_class.jac_c32_value
	r=E_F-(np.dot(A2,D[0:30])+V32)
	J=(-1)*J32
	Dr=2*np.dot(J.T,r)
	return Dr
	

def calculate_c(gamma):
	E_F=np.zeros((75666))
	E=np.zeros((Ns))
	Fx=np.zeros((75606))
	Fy=np.zeros((75606))
	Fz=np.zeros((75606))
	Natoms=[]
	A2=np.zeros((75666,(3*Ng)))
	A2_E=np.zeros((Ns,(3*Ng)))
	A2_Fx=np.zeros((75606,(3*Ng)))
	A2_Fy=np.zeros((75606,(3*Ng)))
	A2_Fz=np.zeros((75606,(3*Ng)))
	a=0
	b=0
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,fname2,(i+1),fpath1,fpath2)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		Force=info[3]
		Natoms.append(len(elements))
		E_F[a]=Energy[i]
		E_F[a+1:a+1+len(elements)]=Force[:,0]
		E_F[a+1+len(elements):a+1+(2*len(elements))]=Force[:,1]
		E_F[a+1+(2*len(elements)):a+1+(3*len(elements))]=Force[:,2]
		E[i]=Energy[i]
		Fx[b:b+len(elements)]=Force[:,0]
		Fy[b:b+len(elements)]=Force[:,1]
		Fz[b:b+len(elements)]=Force[:,2]
		Ap=two_body(gamma,elements,coords_redu,box_mat)
		A2[a,:]=Ap[0]
		A2[a+1:a+1+len(elements),:]=Ap[1]
		A2[a+1+len(elements):a+1+(2*len(elements)),:]=Ap[2]
		A2[a+1+(2*len(elements)):a+1+(3*len(elements)),:]=Ap[3]
		A2_E[i,:]=Ap[0]
		A2_Fx[b:b+len(elements),:]=Ap[1]
		A2_Fy[b:b+len(elements),:]=Ap[2]
		A2_Fz[b:b+len(elements),:]=Ap[3]
		a=a+(3*len(elements))+1
		b=b+len(elements)		
	c=np.dot(np.linalg.pinv(A2),E_F)
	return [A2, A2_E, A2_Fx, A2_Fy, A2_Fz, E_F, E, Fx, Fy, Fz, Natoms, c]		

def cost_function(gamma):
	info=calculate_c(gamma)
	A2=info[0]
	A2_E=info[1]
	A2_Fx=info[2]
	A2_Fy=info[3]
	A2_Fz=info[4]
	E_F=info[5]
	E=info[6]
	Fx=info[7]
	Fy=info[8]
	Fz=info[9]
	Natoms=info[10]
	c=info[11]
	V21=np.dot(A2,c)
	V21_E=np.dot(A2_E,c)
	V21_Fx=np.dot(A2_Fx,c)
	V21_Fy=np.dot(A2_Fy,c)
	V21_Fz=np.dot(A2_Fz,c)
	
	L2_error_total=np.sum((E_F-V21)**2)
	MSE_total=L2_error_total/len(E_F)
	RMSE_total=np.sqrt(MSE_total)

	global Nfeval
	f1=open("error.txt","a")
	for i in range(len(gamma)):
		f1.write("%4d %15.4f %15.4f \n" % (Nfeval, RMSE_total, gamma[i]))
	f2=open("iteration_plot.txt","a")
	a=0
	for i in range(Ns):
		f2.write("%4d %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), E[i], V21_E[i], np.sum(Fx[a:a+Natoms[i]]), np.sum(V21_Fx[a:a+Natoms[i]]), np.sum(Fy[a:a+Natoms[i]]), np.sum(V21_Fy[a:a+Natoms[i]]), np.sum(Fz[a:a+Natoms[i]]), np.sum(V21_Fz[a:a+Natoms[i]])))
		a=a+Natoms[i]
	f2.write("\n")
	Nfeval=Nfeval+1
	return L2_error_total

Nfeval=1
Ns=60
Ng=10
fname1="CONTCAR%i"
fname2="OUTCAR%i"
fpath1="/storage/home/nkd5102/scratch/GEAM/CONTCAR_files/"
fpath2="/storage/home/nkd5102/scratch/GEAM/OUTCAR_files/"
Energy=np.loadtxt("data1_energies.txt")
r_cutoff=5
q=[0,1,2,3,4,5,6,7,8,9]
gamma = np.array([0.1, 0.15, 0.2, 1.00, 1.50, 2.00])
res = minimize(cost_function, gamma, method='nelder-mead', options={'maxiter':70, 'disp':True})

gamma=res.x
info=calculate_c(gamma)
A2=info[0]
A2_E=info[1]
A2_Fx=info[2]
A2_Fy=info[3]
A2_Fz=info[4]
E_F=info[5]
E=info[6]
Fx=info[7]
Fy=info[8]
Fz=info[9]
Natoms=info[10]
c21=info[11]
print (c21)
print (gamma)
c32 = c21
Niter=1


def calc_pot_c32(c32,pot_c32_class):
	pot=potential_three_body_two_bond(c32)
	pot_c32_value=pot[0]
	pot_E_c32_value=pot[1]
	pot_Fx_c32_value=pot[2]
	pot_Fy_c32_value=pot[3]
	pot_Fz_c32_value=pot[4]
	jac_c32_value=pot[5]
	pot_c32_class.pot_c32_value = pot_c32_value
	pot_c32_class.pot_E_c32_value = pot_E_c32_value
	pot_c32_class.pot_Fx_c32_value = pot_Fx_c32_value
	pot_c32_class.pot_Fy_c32_value = pot_Fy_c32_value
	pot_c32_class.pot_Fz_c32_value = pot_Fz_c32_value
	pot_c32_class.jac_c32_value = jac_c32_value


A=np.zeros((75666,31))
D=np.zeros((31))
D[0:30]=c21
pot_c32_class=pot_c32()

for i in range(20):
	res_c32=minimize(cost_func_32, c32, args=(D,pot_c32_class), method='L-BFGS-B', jac=jac_cf_32)
	c32=res_c32.x
	pot32=potential_three_body_two_bond(c32)
	V32=pot32[0]
	A[:,0:30]=A2
	A[:,30]=V32
	RMSE=np.sqrt(np.sum((E_F-(np.dot(A2,D[0:30])+V32))**2).mean())
	D=np.dot(np.linalg.pinv(A),E_F)
	f1=open("error_lambda.dat","a")
	f1.write("%20d %15.4f %15.4f \n" % (i, RMSE, D[30]))
	f2=open("c32_vector.dat","a")
	f2.write(str(c32)+"\n")
	f3=open("D_vector.dat","a")
	f3.write(str(D)+"\n")
	c32=np.sqrt(D[30])*c32


print (D[0:30])
print (c32)
print (gamma)
f5=open("Energy_plot_21_32.dat","w")
f6=open("Forcex_plot_21_32.dat","w")
f7=open("Forcey_plot_21_32.dat","w")
f8=open("Forcez_plot_21_32.dat","w")
a=0
pot32=potential_three_body_two_bond(c32)
V32=pot32[0]
V32_E=pot32[1]
V32_Fx=pot32[2]
V32_Fy=pot32[3]
V32_Fz=pot32[4]
V21_E=np.dot(A2_E,D[0:30])
V_E=V21_E+V32_E
V21_Fx=np.dot(A2_Fx,D[0:30])
V_Fx=V21_Fx+V32_Fx
V21_Fy=np.dot(A2_Fy,D[0:30])
V_Fy=V21_Fy+V32_Fy
V21_Fz=np.dot(A2_Fz,D[0:30])
V_Fz=V21_Fz+V32_Fz

for i in range(Ns):
	f5.write("%4d %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), E[i], V_E[i], V21_E[i], V32_E[i]))
	f6.write("%4d %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fx[a:a+Natoms[i]]), np.sum(V_Fx[a:a+Natoms[i]]), np.sum(V21_Fx[a:a+Natoms[i]]), np.sum(V32_Fx[a:a+Natoms[i]])))
	f7.write("%4d %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fy[a:a+Natoms[i]]), np.sum(V_Fy[a:a+Natoms[i]]), np.sum(V21_Fy[a:a+Natoms[i]]), np.sum(V32_Fy[a:a+Natoms[i]])))
	f8.write("%4d %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fz[a:a+Natoms[i]]), np.sum(V_Fz[a:a+Natoms[i]]), np.sum(V21_Fz[a:a+Natoms[i]]), np.sum(V32_Fz[a:a+Natoms[i]])))
	a=a+Natoms[i]
f5.close()
f6.close()
f7.close()
f8.close()


c21=D[0:30]
c33 = c21

def calc_pot_c33(c33,pot_c33_class):
	pot=potential_three_body_three_bond(c33)
	pot_c33_value=pot[0]
	pot_E_c33_value=pot[1]
	pot_Fx_c33_value=pot[2]
	pot_Fy_c33_value=pot[3]
	pot_Fz_c33_value=pot[4]
	jac_c33_value=pot[5]
	pot_c33_class.pot_c33_value = pot_c33_value
	pot_c33_class.pot_E_c33_value = pot_E_c33_value
	pot_c33_class.pot_Fx_c33_value = pot_Fx_c33_value
	pot_c33_class.pot_Fy_c33_value = pot_Fy_c33_value
	pot_c33_class.pot_Fz_c33_value = pot_Fz_c33_value
	pot_c33_class.jac_c33_value = jac_c33_value

Nstep=1
A=np.zeros((75666,32))
D=np.zeros((32))
D[0:30]=c21
pot_c33_class=pot_c33()

for i in range(20):
	
	res_c33=minimize(cost_func_33, c33, args=(D,c32,pot_c33_class), method='L-BFGS-B', jac=jac_cf_33)

	c33=res_c33.x
	pot33=potential_three_body_three_bond(c33)
	V33=pot33[0]
	pot32=potential_three_body_two_bond(c32)
	V32=pot32[0]
	A[:,0:30]=A2
	A[:,30]=V32
	A[:,31]=V33
	D=np.dot(np.linalg.pinv(A),E_F)
	RMSE=np.sqrt(np.sum((E-(np.dot(A2,D[0:30])+V32+V33))**2).mean())
	f1=open("error_lambda_new.dat","a")
	f1.write("%20d %15.4f %15.4f %15.4f \n" % (i, RMSE, D[30], D[31]))
	f2=open("c32_new_vector.dat","a")
	f2.write(str(c32)+"\n")
	f3=open("c33_new_vector.dat","a")
	f3.write(str(c33)+"\n")	
	f4=open("D_new_vector.dat","a")
	f4.write(str(D)+"\n")
	c32=np.sqrt(D[30])*c32
	c33=np.cbrt(D[31])*c33

f5=open("Energy_plot_21_32_33.dat","w")
f6=open("Forcex_plot_21_32_33.dat","w")
f7=open("Forcey_plot_21_32_33.dat","w")
f8=open("Forcez_plot_21_32_33.dat","w")
a=0
pot33=potential_three_body_three_bond(c33)
V33=pot33[0]
V33_E=pot33[1]
V33_Fx=pot33[2]
V33_Fy=pot33[3]
V33_Fz=pot33[4]
pot32=potential_three_body_two_bond(c32)
V32=pot32[0]
V32_E=pot32[1]
V32_Fx=pot32[2]
V32_Fy=pot32[3]
V32_Fz=pot32[4]
V21_E=np.dot(A2_E,D[0:30])
V_E=V21_E+V32_E+V33_E
V21_Fx=np.dot(A2_Fx,D[0:30])
V_Fx=V21_Fx+V32_Fx+V33_Fx
V21_Fy=np.dot(A2_Fy,D[0:30])
V_Fy=V21_Fy+V32_Fy+V33_Fz
V21_Fz=np.dot(A2_Fz,D[0:30])
V_Fz=V21_Fz+V32_Fz+V33_Fz

for i in range(Ns):
        f5.write("%4d %15.4f %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), E[i], V_E[i], V21_E[i], V32_E[i], V33_E[i]))
        f6.write("%4d %15.4f %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fx[a:a+Natoms[i]]), np.sum(V_Fx[a:a+Natoms[i]]), np.sum(V21_Fx[a:a+Natoms[i]]), np.sum(V32_Fx[a:a+Natoms[i]]), np.sum(V33_Fx[a:a+Natoms[i]])))
        f7.write("%4d %15.4f %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fy[a:a+Natoms[i]]), np.sum(V_Fy[a:a+Natoms[i]]), np.sum(V21_Fy[a:a+Natoms[i]]), np.sum(V32_Fy[a:a+Natoms[i]]), np.sum(V33_Fy[a:a+Natoms[i]])))
        f8.write("%4d %15.4f %15.4f %15.4f %15.4f %15.4f \n" % ((i+1), np.sum(Fz[a:a+Natoms[i]]), np.sum(V_Fz[a:a+Natoms[i]]), np.sum(V21_Fz[a:a+Natoms[i]]), np.sum(V32_Fz[a:a+Natoms[i]]), np.sum(V33_Fz[a:a+Natoms[i]])))
        a=a+Natoms[i]
f5.close()
f6.close()
f7.close()
f8.close()

