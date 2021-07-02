import os
import numpy as np

def read_CONTCAR_OUTCAR(fname1, n, fpath1):
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

	return [elements, box_mat, coords_redu]

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

def neigh_list_33(box_mat,coords_redu,elements,i):
	neigh=[]
	neigh_dis=[]
	for j in range(len(elements)):
		if j!=i:
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
	A2_Fx=np.zeros((1,(3*Ng)))
	A2_Fy=np.zeros((1,(3*Ng)))
	A2_Fz=np.zeros((1,(3*Ng)))
	i=1
	neighbors=neigh_list_33(box_mat,coords_redu,elements,i)
	closest_atoms=neighbors[0]
	R=neighbors[1]
	for j in range(len(closest_atoms)):
		for l in q:
			idx=(elements[i]+elements[closest_atoms[j]])*Ng+l
			gaussian=calculate_gaussian(gamma,elements[i],elements[closest_atoms[j]],l,R[j])
			A2_E[0,idx]+=gaussian[0]
			A2_Fx[0,idx]+=gaussian[1]
			A2_Fy[0,idx]+=gaussian[2]
			A2_Fz[0,idx]+=gaussian[3]

	return [A2_E, A2_Fx, A2_Fy, A2_Fz]


def three_body_two_bond(gamma,c32,elements,coords_redu,box_mat):

	A32_E=np.zeros((1))
	A32_Fx=np.zeros((1))
	A32_Fy=np.zeros((1))
	A32_Fz=np.zeros((1))
	i=1
	neighbors=neigh_list_33(box_mat,coords_redu,elements,i)
	closest_atoms=neighbors[0]
	R=neighbors[1]
	for j in range(len(closest_atoms)):
		for k in range(j+1,len(closest_atoms)):
			V_ij=np.zeros((4,(3*Ng)))
			V_ik=np.zeros((4,(3*Ng)))
			for l in q:
				idx_ij=(elements[i]+elements[closest_atoms[j]])*Ng+l
				idx_ik=(elements[i]+elements[closest_atoms[k]])*Ng+l
				gaussian_ij=calculate_gaussian(gamma,elements[i],elements[closest_atoms[j]],l,R[j])
				gaussian_ik=calculate_gaussian(gamma,elements[i],elements[closest_atoms[k]],l,R[k])
				V_ij[:,idx_ij]=gaussian_ij[:]
				V_ik[:,idx_ik]=gaussian_ik[:]
			A32_E[0]+=(np.dot(V_ij[0,:],c32)*np.dot(V_ik[0,:],c32))
			A32_Fx[0]+=(np.dot(V_ij[1,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[1,:],c32))
			A32_Fy[0]+=(np.dot(V_ij[2,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[2,:],c32))
			A32_Fz[0]+=(np.dot(V_ij[3,:],c32)*np.dot(V_ik[0,:],c32))+(np.dot(V_ij[0,:],c32)*np.dot(V_ik[3,:],c32))

	return [A32_E, A32_Fx, A32_Fy, A32_Fz]

def three_body_three_bond(gamma,c33,elements,coords_redu,box_mat):

	A33_E=np.zeros((1))
	A33_Fx=np.zeros((1))
	A33_Fy=np.zeros((1))
	A33_Fz=np.zeros((1))
	i=1

	neighbors=neigh_list_33(box_mat,coords_redu,elements,i)
	closest_atoms=neighbors[0]
	R=neighbors[1]
	for j in range(len(closest_atoms)):
		
		for k in range(j+1,len(closest_atoms)):
			R_jk=distance(box_mat,coords_redu,closest_atoms[j],closest_atoms[k])
			V_ij=np.zeros((4,(3*Ng)))
			V_ik=np.zeros((4,(3*Ng)))
			V_jk=np.zeros((4,(3*Ng)))
			if R_jk[0]<r_cutoff:
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
				A33_Fx[0]+=0.5*((np.dot(V_ij[1,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[1,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[1,:],c33)))
				A33_Fy[0]+=0.5*((np.dot(V_ij[2,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[2,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[2,:],c33)))
				A33_Fz[0]+=0.5*((np.dot(V_ij[3,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[3,:],c33)*np.dot(V_jk[0,:],c33))+(np.dot(V_ij[0,:],c33)*np.dot(V_ik[0,:],c33)*np.dot(V_jk[3,:],c33)))
	return [A33_E, A33_Fx, A33_Fy, A33_Fz]


def potential_two_body(gamma):
	A2_E=np.zeros((9,(3*Ng)))
	A2_Fx=np.zeros((9,(3*Ng)))
	A2_Fy=np.zeros((9,(3*Ng)))
	A2_Fz=np.zeros((9,(3*Ng)))
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,(i+1),fpath1)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		A2=two_body(gamma,elements,coords_redu,box_mat)
		A2_E[i,:]=A2[0]
		A2_Fx[i,:]=A2[1]
		A2_Fy[i,:]=A2[2]
		A2_Fz[i,:]=A2[3]
	return [A2_E, A2_Fx, A2_Fy, A2_Fz]

def potential_three_body_two_bond(gamma):
	A32_E=np.zeros((9))
	A32_Fx=np.zeros((9))
	A32_Fy=np.zeros((9))
	A32_Fz=np.zeros((9))
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,(i+1),fpath1)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		A32=three_body_two_bond(gamma,c32,elements,coords_redu,box_mat)
		A32_E[i]=A32[0]
		A32_Fx[i]=A32[1]
		A32_Fy[i]=A32[2]
		A32_Fz[i]=A32[3]
	return [A32_E, A32_Fx, A32_Fy, A32_Fz]

def potential_three_body_three_bond(gamma):
	A33_E=np.zeros((9))
	A33_Fx=np.zeros((9))
	A33_Fy=np.zeros((9))
	A33_Fz=np.zeros((9))
	for i in range(Ns):
		info=read_CONTCAR_OUTCAR(fname1,(i+1),fpath1)
		elements=info[0]
		box_mat=info[1]
		coords_redu=info[2]
		A33=three_body_three_bond(gamma,c33,elements,coords_redu,box_mat)
		A33_E[i]=A33[0]
		A33_Fx[i]=A33[1]
		A33_Fy[i]=A33[2]
		A33_Fz[i]=A33[3]
	return [A33_E, A33_Fx, A33_Fy, A33_Fz]

Nfeval=1
Ns=9
Ng=10
fname1="CONTCAR%i"
fpath1="/storage/home/nkd5102/scratch/GEAM/1D_CONTCAR_files/"
r_cutoff=5
q=[0,1,2,3,4,5,6,7,8,9]

gamma=np.array([0.1,0.2,0.3,0.4,0.5,0.6])
c21=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
c32=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
c33=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

info=potential_two_body(gamma)

A2_E=info[0]
A2_Fx=info[1]
A2_Fy=info[2]
A2_Fz=info[3]

E21=np.dot(A2_E,c21)
F21_x=np.dot(A2_Fx,c21)
F21_y=np.dot(A2_Fy,c21)
F21_z=np.dot(A2_Fz,c21)

f1=open("energy_two_body.txt","w")
f2=open("force_x_two_body.txt","w")
f3=open("force_y_two_body.txt","w")
f4=open("force_z_two_body.txt","w")


for i in range(Ns):
	f1.write("%4d %20.15f \n" % ((i+1), E21[i]))
	f2.write("%4d %20.15f \n" % ((i+1), F21_x[i]))
	f3.write("%4d %20.15f \n" % ((i+1), F21_y[i]))
	f4.write("%4d %20.15f \n" % ((i+1), F21_z[i]))     
        
f1.close()
f2.close()
f3.close()
f4.close()

info=potential_three_body_two_bond(gamma)

A32_E=info[0]
A32_Fx=info[1]
A32_Fy=info[2]
A32_Fz=info[3]

f1=open("energy_three_body_two_bond.txt","w")
f2=open("force_x_three_body_two_bond.txt","w")
f3=open("force_y_three_body_two_bond.txt","w")
f4=open("force_z_three_body_two_bond.txt","w")


for i in range(Ns):
	f1.write("%4d %20.15f \n" % ((i+1), A32_E[i]))
	f2.write("%4d %20.15f \n" % ((i+1), A32_Fx[i]))
	f3.write("%4d %20.15f \n" % ((i+1), A32_Fy[i]))
	f4.write("%4d %20.15f \n" % ((i+1), A32_Fz[i]))

f1.close()
f2.close()
f3.close()
f4.close()

info=potential_three_body_three_bond(gamma)

A33_E=info[0]
A33_Fx=info[1]
A33_Fy=info[2]
A33_Fz=info[3]

f1=open("energy_three_body_three_bond.txt","w")
f2=open("force_x_three_body_three_bond.txt","w")
f3=open("force_y_three_body_three_bond.txt","w")
f4=open("force_z_three_body_three_bond.txt","w")


for i in range(Ns):
	f1.write("%4d %20.15f \n" % ((i+1), A33_E[i]))
	f2.write("%4d %20.15f \n" % ((i+1), A33_Fx[i]))
	f3.write("%4d %20.15f \n" % ((i+1), A33_Fy[i]))
	f4.write("%4d %20.15f \n" % ((i+1), A33_Fz[i]))

f1.close()
f2.close()
f3.close()
f4.close()

E=E21+A32_E+A33_E
Fx=F21_x+A32_Fx+A33_Fx
Fy=F21_y+A32_Fy+A33_Fy
Fz=F21_z+A32_Fz+A33_Fz

f1=open("total_energy.txt","w")
f2=open("force_x_total.txt","w")
f3=open("force_y_total.txt","w")
f4=open("force_z_total.txt","w")

for i in range(Ns):
	f1.write("%4d %20.15f \n" % ((i+1), E[i]))
	f2.write("%4d %20.15f \n" % ((i+1), Fx[i]))
	f3.write("%4d %20.15f \n" % ((i+1), Fy[i]))
	f4.write("%4d %20.15f \n" % ((i+1), Fz[i]))

f1.close()
f2.close()
f3.close()
f4.close()

