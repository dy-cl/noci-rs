from pyscf import gto, scf
import numpy as np 
import h5py

def build_mol(r, basis, atom1, atom2, unit):

    mol = gto.M(atom = f'{atom1} 0 0 {-0.5 * r}; {atom2} 0 0 {0.5 * r}', basis = basis, unit = unit, spin = 0)

    return mol 

def calculate_integrals(mol):
   
    eris = mol.intor('int2e', aosym = 's1')
    S = scf.hf.get_ovlp(mol)
    h = scf.hf.get_hcore(mol)
    rhf = scf.RHF(mol)
    dm = rhf.get_init_guess()

    return eris, S, h, dm

def get_misc(mol):

    Enuc = mol.energy_nuc()
    nao = mol.nao 
    nelec = mol.nelec
    labels = mol.ao_labels()
    a, b = 0, 1
    ia = [i for i,s in enumerate(labels) if s.split()[0] == str(a)]
    ib = [i for i,s in enumerate(labels) if s.split()[0] == str(b)]
    aolabels = [ia, ib]

    return Enuc, nao, nelec, aolabels

def dump_hdf5(eri, S, h, dm, Enuc, nao, nelec, aolabels, path):
    
    with h5py.File(path, 'w') as f:
        f.create_dataset('eri', data = eris)
        f.create_dataset('S', data = S)
        f.create_dataset('h', data = h)
        f.create_dataset('dm', data = dm)
        f.create_dataset('Enuc', data = Enuc)
        f.create_dataset('nao', data = nao)
        f.create_dataset('nelec', data = np.array(nelec, dtype = np.int64))
        f.create_dataset('aolabels', data = aolabels)

if __name__ == '__main__':
    
    r = 5.5
    basis = 'STO-3G'
    atom1 = 'H'
    atom2 = 'H'
    unit = 'Ang'

    mol = build_mol(r, basis, atom1, atom2, unit)
    eris, S, h, dm = calculate_integrals(mol)
    Enuc, nao, nelec, aolabels = get_misc(mol)
    dump_hdf5(eris, S, h, dm, Enuc, nao, nelec, aolabels, 'data.h5')

