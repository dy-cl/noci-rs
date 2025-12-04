from pyscf import gto, scf, fci
import numpy as np 
import h5py
import argparse, json

def build_mol(basis, atoms, unit):

    return gto.M(atom = '; '.join(atoms), basis = basis, unit = unit, spin = 0)

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
    aolabels = mol.ao_labels()

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
        if fci_energy is not None:
            f.create_dataset('E_fci', data = fci_energy)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--atoms', type = str, required = True,)
    parser.add_argument('--basis', type = str, required = True,)
    parser.add_argument('--unit', type = str, default = 'Ang')
    parser.add_argument('--out', type = str, default = 'data.h5')
    parser.add_argument('--fci', type = lambda s: s.lower() == "true")
    args = parser.parse_args()
    atoms = json.loads(args.atoms)

    mol = build_mol(args.basis, atoms, args.unit)
    eris, S, h, dm = calculate_integrals(mol)
    Enuc, nao, nelec, aolabels = get_misc(mol)
    fci_energy = None
    if args.fci:
        mf = scf.RHF(mol).run()
        cisolver = fci.FCI(mol, mf.mo_coeff)
        fci_energy, fcivec = cisolver.kernel()
    dump_hdf5(eris, S, h, dm, Enuc, nao, nelec, aolabels, args.out)

