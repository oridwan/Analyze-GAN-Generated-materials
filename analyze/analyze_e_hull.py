import numpy as np
from view_atoms_mgmno import view_atoms
from mace.calculators import mace_mp
from ase.optimize import BFGS
import numpy as np
from mp_api.client import MPRester
from pymatgen.entries.compatibility import MaterialsProjectCompatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.entries.computed_entries import ComputedEntry
from concurrent.futures import ProcessPoolExecutor

def compute_par(data, pd_mace, calculator):
    print("Parallel process in compute_par", len(data))
    for i, d in enumerate(data):
        #try:
        if True:
            atoms, image = view_atoms(d, view=False)
            formula = atoms.get_chemical_formula()
            atoms.set_calculator(calculator)
            eng0 = atoms.get_potential_energy()
            dyn = BFGS(atoms, logfile='ase.log')
            dyn.run(fmax=0.05, steps=50)
            eng1 = atoms.get_potential_energy()
            #print(i, atoms, eng1)
            if eng1 < 0:
                pmg = AseAtomsAdaptor.get_structure(atoms)
                eng = atoms.get_potential_energy()
                entry = ComputedEntry(pmg.composition, eng1)
                e_hull = pd_mace.get_e_above_hull(entry, allow_negative=True)
                strs = "{:4d} {:10s} {:12.3f} {:12.3f} {:12.3f}".format(i, formula, e_hull, eng0, eng1)
                if e_hull > -0.05 and e_hull < 0.05:
                    strs += ' +++++++'
                print(strs)
        #except Exception as e:
            #print(f"Error processing data {i}: {e}")
    return None


if __name__ == "__main__":

    # Download the known structures from Materials Project
    API = "PicmHpi7uqeTyxmDLULfduhV5iS27OtJ"
    mpr = MPRester(API)
    entries = mpr.get_entries_in_chemsys("Mg-Mn-O",
                                         additional_criteria={"is_stable": True})
    # Get the stable structures
    pd = PhaseDiagram(entries)
    ref_structures = []
    for entry in entries:
        e_above_hull = pd.get_e_above_hull(entry)
        if e_above_hull < 1e-5:
            ase_atoms = AseAtomsAdaptor.get_atoms(entry.structure)
            ref_structures.append(ase_atoms)

    # Recompute the energy for the stable structures with the MACE calculator
    calculator = mace_mp(model="medium", dispersion=False)

    # Assuming `pmg` is your Pymatgen structure
    pmgs = []
    engs = []
    for struc in ref_structures:
        pmg = AseAtomsAdaptor.get_structure(struc)
        struc.set_calculator(calculator)
        eng0 = struc.get_potential_energy()
        if eng0 < 0:
            # Relaxation
            dyn = BFGS(struc)
            dyn.run(fmax=0.05, steps=10)
            eng1 = struc.get_potential_energy()
            if abs(eng1-eng0)/len(struc) < 0.5:
                pmgs.append(pmg)
                engs.append(eng1)

    # Create the reference convex hull in MACE model
    mace_entries = []
    for structure, energy in zip(pmgs, engs):
        entry = ComputedEntry(structure.composition, energy)
        mace_entries.append(entry)

    pd_mace = PhaseDiagram(mace_entries)
    for entry in mace_entries:
        e_above_hull = pd_mace.get_e_above_hull(entry)
        #print(e_above_hull)

    print("Now check the stability for each of the GAN structure\n")
    data = np.load('gen_image_cwgan_mgmno/gen_images_250.npy')
    #data = data[:24]
    ncpu = 48


    print("Parallel processes")
    N_cycle = int(np.ceil(len(data) / ncpu))
    args_list = []
    for i in range(ncpu):
        id1 = i * N_cycle
        id2 = min([id1 + N_cycle, len(data)])
        args_list.append((data[id1:id2], pd_mace, calculator))

    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        results = [executor.submit(compute_par, *p) for p in args_list]
