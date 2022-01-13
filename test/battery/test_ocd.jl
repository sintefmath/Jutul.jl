using Jutul

##
ac = ACMaterial()
grafite = Grafite()
nmc111 = NMC111()
T=300;
c=0.1;
a=OCD(T,c,nmc111)
ba=OCD(T,c,grafite)
