#!/usr/bin/env python

import numpy as np
import warnings, time, os, shutil, csv, glob, argparse, sys
from pathlib import Path
import pandas as pd
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

warnings.filterwarnings("ignore")

# --- Utility Function: CalcLigRMSD ---
def CalcLigRMSD(lig1, lig2, rename_lig2=True, output_filename="tmp.pdb"):
    if lig1 is None or lig2 is None:
        return float('inf')

    lig1_no_h = Chem.RemoveHs(lig1)
    lig2_no_h = Chem.RemoveHs(lig2)

    if lig1_no_h.GetNumAtoms() == 0 or lig2_no_h.GetNumAtoms() == 0:
        return float('inf')

    coordinates_lig2 = lig2_no_h.GetConformer().GetPositions()
    coordinates_lig1 = lig1_no_h.GetConformer().GetPositions()

    res = rdFMCS.FindMCS([lig1_no_h, lig2_no_h])
    if res.smartsString == '':  # Handle cases where no common MCS is found
        return float('inf')  # Return a very large number for no match

    ref_mol = Chem.MolFromSmarts(res.smartsString)
    if ref_mol is None: # Should not happen if smartsString is not empty, but good to check
        return float('inf')

    mas1 = list(lig1_no_h.GetSubstructMatch(ref_mol))
    mas2_list = lig2_no_h.GetSubstructMatches(ref_mol, uniquify=False)

    if not mas1 or not mas2_list:
        return float('inf')  # No substructure match

    coordinates_lig1_mcs = coordinates_lig1[mas1]
    list_rmsd = []

    for match2 in mas2_list:
        if len(match2) != len(mas1): # MCS must have the same number of atoms
            continue
        coordinates_lig2_tmp = coordinates_lig2[list(match2)]
        diff = coordinates_lig2_tmp - coordinates_lig1_mcs
        list_rmsd.append(np.sqrt((diff * diff).sum() / len(coordinates_lig2_tmp)))

    if not list_rmsd:
        return float('inf')

    lig_rmsd = min(list_rmsd)

    if rename_lig2:
        best_match_idx = np.argmin(list_rmsd)
        mas2 = mas2_list[best_match_idx]

        if lig2.GetNumAtoms() > 0 and lig1.GetNumAtoms() > 0:
            correspondence_key2_item1 = dict(zip(mas2, mas1))

            atom_names_lig1 = []
            for i, atom1 in enumerate(lig1.GetAtoms()):
                pdb_info_lig1 = atom1.GetPDBResidueInfo()
                if pdb_info_lig1:
                    atom_names_lig1.append(pdb_info_lig1.GetName())
                else:
                    atom_names_lig1.append(f'ATOM{i+1}') # Fallback for atom names

            lig1_ResName = "LIG"
            if lig1.GetAtoms()[0].GetPDBResidueInfo():
                lig1_ResName = lig1.GetAtoms()[0].GetPDBResidueInfo().GetResidueName()

            for i, atom2 in enumerate(lig2.GetAtoms()):
                pdb_info = atom2.GetPDBResidueInfo()
                if pdb_info:
                    pdb_info.SetResidueName(lig1_ResName)
                    if i in correspondence_key2_item1:
                        # Ensure index is within bounds for atom_names_lig1
                        mapped_idx = correspondence_key2_item1[i]
                        if mapped_idx < len(atom_names_lig1):
                            pdb_info.SetName(atom_names_lig1[mapped_idx])
                else:
                    # Create PDBInfo if not present
                    new_pdb_info = Chem.AtomPDBResidueInfo(resName=lig1_ResName)
                    if i in correspondence_key2_item1:
                        mapped_idx = correspondence_key2_item1[i]
                        if mapped_idx < len(atom_names_lig1):
                            new_pdb_info.SetName(atom_names_lig1[mapped_idx])
                    atom2.SetMonomerInfo(new_pdb_info)

        Chem.MolToPDBFile(lig2, output_filename)

    return lig_rmsd

# --- Main Script ---
def main():
    start_time = time.time()

    # --- Setup Directories ---
    current_path = Path.cwd()
    complex_folder = current_path / 'complex'
    docked_ligands_folder = current_path / 'docked_ligands'
    complex_folder.mkdir(parents=True, exist_ok=True)
    docked_ligands_folder.mkdir(parents=True, exist_ok=True)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run a docking simulation with Vina-GPU.")
    # Required arguments
    parser.add_argument('--mgl_tools_path', type=str, required=True,
                        help="Path to MGLTools (e.g., /home/MGLTools-1.5.7)")
    parser.add_argument('--vina_path', type=str, required=True,
                        help="Path to Vina-GPU (e.g., /home/Vina-GPU-2-1)")
    parser.add_argument('--protein_file', type=str, required=True,
                        help="Protein/Target PDB File Name with .pdb")
    parser.add_argument('--lig_smiles_file', type=str, required=True,
                        help="Ligand smiles file (e.g., ligands.csv)")
    # Optional arguments
    parser.add_argument('--chain', type=str, default="",
                        help="Desired Chain ID of Protein (leave empty for first chain found)")
    parser.add_argument('--lig_id', type=str, default="",
                        help="Ligand ID in PDB Crystal Structure (leave empty for blind docking)")
    parser.add_argument('--number_of_poses', type=int, default=9,
                        help="Number of Ligand poses to generate (default: 9)")
    parser.add_argument('--threshold', type=float, default=-1.0,
                        help="Threshold value of Docking Score (e.g., -7.0, default: -1.0)")
    parser.add_argument('--gpu_id', type=str, default="",
                        help="GPU ID to use (e.g., 0, 1, or empty for default)")
    # Center and Size coordinates - optional but often used together
    parser.add_argument('--center_x', type=float, help="Center X coordinate")
    parser.add_argument('--center_y', type=float, help="Center Y coordinate")
    parser.add_argument('--center_z', type=float, help="Center Z coordinate")
    parser.add_argument('--size_x', type=float, help="Size X coordinate")
    parser.add_argument('--size_y', type=float, help="Size Y coordinate")
    parser.add_argument('--size_z', type=float, help="Size Z coordinate")
    args = parser.parse_args()

    # --- Process Arguments ---
    print("\n##########################################")
    print("User Inputs:")
    print(f"MGLTools Path: {args.mgl_tools_path}")
    print(f"Vina Path: {args.vina_path}")
    print(f"Protein PDB File: {args.protein_file}")
    print(f"Ligand SMILES File: {args.lig_smiles_file}")
    print(f"Protein Chain ID: {args.chain if args.chain else 'First chain found'}")
    print(f"Ligand ID: {args.lig_id if args.lig_id else 'Blind docking'}")
    print(f"Number of Poses: {args.number_of_poses}")
    print(f"Docking Score Threshold: {args.threshold}")

    if args.gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
        print(f"Using GPU ID: {args.gpu_id}")
    else:
        print("No GPU ID provided. Vina-GPU will use its default behavior.")

    # --- Protein Preparation ---
    pdb_file_name = args.protein_file
    ppdb = PandasPdb().read_pdb(pdb_file_name)
    protein_name = Path(pdb_file_name).stem

    print("\n--- Protein Chain Selection ---")
    p_chain_id = args.chain
    if not p_chain_id:
        # If no chain specified, use the first chain found in ATOM records
        available_chains = ppdb.df['ATOM']['chain_id'].unique().tolist()
        if available_chains:
            p_chain_id = available_chains[0]
            print(f"No specific chain ID provided. Using the first chain found: '{p_chain_id}'.")
        else:
            print("Error: No ATOM records found in the protein PDB file to determine chain ID.")
            sys.exit(1)
    else:
        # Validate if the specified chain exists
        if p_chain_id not in ppdb.df['ATOM']['chain_id'].unique().tolist():
            print(f"Error: Specified chain ID '{p_chain_id}' not found in the protein PDB file.")
            sys.exit(1)
        print(f"Using specified protein chain ID: '{p_chain_id}'.")

    ppdb.df['ATOM'] = ppdb.df['ATOM'][ppdb.df['ATOM']['chain_id'] == p_chain_id]
    ppdb_path = 'final_model.pdb'
    ppdb.to_pdb(path=str(ppdb_path), records=['ATOM'], gz=False, append_newline=True)
    print(f"Filtered protein saved to: {ppdb_path}")

    # --- Calculate Docking Box Coordinates ---
    cent_x, cent_y, cent_z = args.center_x, args.center_y, args.center_z
    size_x, size_y, size_z = args.size_x, args.size_y, args.size_z

    ldb_path = None # Initialize ldb_path for potential RMSD calculation later

    if all(coord is not None for coord in [cent_x, cent_y, cent_z, size_x, size_y, size_z]):
        print("\nUsing provided center and size coordinates for docking box.")
        with open(current_path / "user_defined_pocket.txt", "w") as ftt:
            ftt.write("User-defined pocket:\n")
            ftt.write(f"Center: {cent_x}\t{cent_y}\t{cent_z}\n")
            ftt.write(f"Size: {size_x}\t{size_y}\t{size_z}\n")
    else:
        # Condition: Center and Size coordinates not fully provided via arguments
        if args.lig_id:
            # Condition: Reference Ligand-based pocket docking
            print(f"\nUsing reference ligand '{args.lig_id}' for pocket definition...")
            ldb = PandasPdb().read_pdb(pdb_file_name)
            
            # Filter HETATM records for the specified ligand ID
            hetatm_df = ldb.df['HETATM'][ldb.df['HETATM']['residue_name'] == args.lig_id]

            if hetatm_df.empty:
                print(f"Error: Ligand ID '{args.lig_id}' not found in HETATM records. Cannot define pocket.")
                sys.exit(1)

            ldb_path = current_path / f'ref_{args.lig_id}_lig.pdb'
            ldb_filtered = PandasPdb()
            ldb_filtered.df['HETATM'] = hetatm_df
            ldb_filtered.to_pdb(path=str(ldb_path), records=['HETATM'], gz=False, append_newline=True)
            print(f"Reference ligand saved to: {ldb_path}")

            minx = hetatm_df['x_coord'].min()
            maxx = hetatm_df['x_coord'].max()
            cent_x = round((maxx + minx) / 2, 2)
            size_x = round(abs(maxx - minx) + 2, 2) # Smaller buffer for focused docking

            miny = hetatm_df['y_coord'].min()
            maxy = hetatm_df['y_coord'].max()
            cent_y = round((maxy + miny) / 2, 2)
            size_y = round(abs(maxy - miny) + 2, 2)

            minz = hetatm_df['z_coord'].min()
            maxz = hetatm_df['z_coord'].max()
            cent_z = round((maxz + minz) / 2, 2)
            size_z = round(abs(maxz - minz) + 2, 2)

            with open(current_path / "Ref_Lig_Based_pocket.txt", "w") as ftt:
                ftt.write(f"Reference Ligand: {args.lig_id}\n")
                ftt.write(f"Center: {cent_x}\t{cent_y}\t{cent_z}\n")
                ftt.write(f"Size: {size_x}\t{size_y}\t{size_z}\n")
            print("Pocket coordinates calculated based on reference ligand.")

        else:
            # Condition: Blind docking (whole protein pocket)
            print("\nPerforming blind docking (whole protein pocket)...")
            minx = ppdb.df['ATOM']['x_coord'].min()
            maxx = ppdb.df['ATOM']['x_coord'].max()
            cent_x = round((maxx + minx) / 2, 2)
            size_x = round(abs(maxx - minx) + 12, 2) # Adding buffer for blind docking

            miny = ppdb.df['ATOM']['y_coord'].min()
            maxy = ppdb.df['ATOM']['y_coord'].max()
            cent_y = round((maxy + miny) / 2, 2)
            size_y = round(abs(maxy - miny) + 12, 2)

            minz = ppdb.df['ATOM']['z_coord'].min()
            maxz = ppdb.df['ATOM']['z_coord'].max()
            cent_z = round((maxz + minz) / 2, 2)
            size_z = round(abs(maxz - minz) + 12, 2)

            with open(current_path / "whole_protein_pocket.txt", "w") as ftt:
                ftt.write("Blind_docking:\n")
                ftt.write(f"Center: {cent_x}\t{cent_y}\t{cent_z}\n")
                ftt.write(f"Size: {size_x}\t{size_y}\t{size_z}\n")
            print("Pocket coordinates calculated for whole protein blind docking.")
    print("##########################################\n")

    # --- Prepare Receptor PDBQT ---
    print("Converting receptor to PDBQT...")
    receptor_pdbqt_path = 'final_model.pdbqt'
    prepprot_cmd = (
        f"{args.mgl_tools_path}/bin/pythonsh "
        f"{args.mgl_tools_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py "
        f"-A bonds_hydrogens -U nphs_lps_waters_nonstdres_deleteAltB "
        f"-r {ppdb_path} -o {receptor_pdbqt_path}"
    )
    os.system(prepprot_cmd)
    if not str(receptor_pdbqt_path):
        print(f"Error: Failed to create receptor PDBQT at {receptor_pdbqt_path}. Check MGLTools path and installation.")
        sys.exit(1)
    print(f"Receptor PDBQT saved to: {receptor_pdbqt_path}")

    # --- Prepare Ligand CSV ---
    print("\nProcessing ligand smiles file...")
    try:
        df = pd.read_csv(args.lig_smiles_file)
        cID = 'ID'
        cls = 'Compound_ID'
        smi = 'Smiles'
        if cID in df.head() or cls in df.head() or smi in df.head() or cID.lower() in df.head() or cls.lower() in df.head() or cls.upper() in df.head() or smi.lower() in df.head() or smi.upper() in df.head():
            print(cID)
            df.to_csv('dummy.csv', header=False, index=False)
        else:
            print("No cID")
            df.to_csv('dummy.csv', header=True, index=False)
        df1 = pd.read_csv("dummy.csv", names=['Compound_ID', 'Smiles'])
        df1.to_csv('ligands_file.csv', index=False)
    except Exception as e:
        print(f"Error processing ligand smiles file {args.lig_smiles_file}: {e}")
        sys.exit(1)

    df_act = pd.read_csv('ligands_file.csv')
    smiles_act = df_act['Smiles'].tolist()
    smiles_id = [str(s).strip() for s in df_act['Compound_ID'].tolist()] # Ensure IDs are strings and stripped

    # --- Create Results CSV File ---
    results_csv_path = Path(f'results_{protein_name}_{Path(args.lig_smiles_file).name}')
    with open(results_csv_path, "w", newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=["Compound_ID", "Smiles", "Docking_Score", "Docking_and_Cluster_pose_RMSD"]
        )
        writer.writeheader()

    # --- Ligand Preparation and Docking Loop ---
    print("\nStarting ligand preparation and docking...")
    for idx, smi in zip(smiles_id, smiles_act):
        print(f"\n--- Processing ligand: {idx} ({smi}) ---")
        lig_opt_pdb_path = current_path / 'lig_opt.pdb'
        lig_pdbqt_path = current_path / 'lig.pdbqt'

        mol_rdkit = None
        try:
            # Try RDKit for 3D generation
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Could not parse SMILES with RDKit.")

            remover = SaltRemover()
            mol = remover.StripMol(mol)
            Chem.Kekulize(mol)

            # Generate 2D image
            # mol2d = Chem.MolFromSmiles(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True))
            # if mol2d:
            #     AllChem.Compute2DCoords(mol2d)
            #     Draw.MolToImageFile(mol2d, str(compound_images_folder / f"{idx}.png"), size=(900, 900), kekulize=True, wedgeBonds=False)
            # else:
            #     print(f"Warning: Could not generate 2D image for {idx} with RDKit.")
            mol_with_hs = Chem.AddHs(mol, addCoords=True)
            AllChem.ComputeGasteigerCharges(mol_with_hs)

            # Attempt to embed 3D coordinates
            res = AllChem.EmbedMolecule(mol_with_hs, maxAttempts=500, useRandomCoords=True, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            if res == -1:
                print(f"RDKit EmbedMolecule failed for {idx}. Trying another attempt or falling back.")
                # Try a different embedding method or increase attempts for robustness
                res = AllChem.EmbedMolecule(mol_with_hs, maxAttempts=1000, useRandomCoords=True, useExpTorsionAnglePrefs=False, useBasicKnowledge=True, ETKDGv3=True)
                if res == -1:
                    raise RuntimeError("RDKit 3D embedding failed after multiple attempts")

            AllChem.MMFFOptimizeMolecule(mol_with_hs, maxIters=50000)
            Chem.MolToPDBFile(mol_with_hs, str(lig_opt_pdb_path))
            mol_rdkit = mol_with_hs # Store for later use in RMSD and bond order assignment

        except Exception as e:
            print(f"RDKit 3D generation failed for {idx} ({smi}): {e}. Falling back to OpenBabel.")
            # Fallback to OpenBabel for 3D generation and image
            # Generate 2D image with OpenBabel
            # os.system(f'obabel -:"{smi}" -osdf -O {current_path / "lig2d.sdf"} --gen2D')
            # if (current_path / "lig2d.sdf").exists():
            #     os.system(f'obabel {current_path / "lig2d.sdf"} -opng -O {compound_images_folder / f"{idx}.png"}')
            # else:
            #     print(f"Warning: OpenBabel could not generate 2D SDF for {idx}.")

            # Generate 3D PDB with OpenBabel
            os.system(f'obabel -:"{smi}" -opdb -O {lig_opt_pdb_path} --gen3D --minimize --sd --ff UFF -n 50000')

            # Clean up intermediate obabel files
            if (current_path / 'lig.sdf').exists(): os.remove(current_path / 'lig.sdf')
            if (current_path / 'lig2d.sdf').exists(): os.remove(current_path / 'lig2d.sdf')

        # Check if lig_opt.pdb was created successfully
        if not lig_opt_pdb_path.exists():
            print(f"Skipping {idx} as 3D ligand generation failed.")
            with open(results_csv_path, "a", newline='') as fg:
                writer = csv.writer(fg)
                writer.writerow([idx, smi, "Error", "N/A"])
            continue # Move to next ligand

        # Prepare ligand PDBQT
        preplig_cmd = (
            f"{args.mgl_tools_path}/bin/pythonsh "
            f"{args.mgl_tools_path}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py "
            f"-A bonds_hydrogens -U nphs_lps -l {lig_opt_pdb_path} -o {lig_pdbqt_path}"
        )
        os.system(preplig_cmd)

        # Clean up lig_opt.pdb
        os.remove(lig_opt_pdb_path)

        if not lig_pdbqt_path.exists():
            print(f"Skipping {idx} as ligand PDBQT preparation failed.")
            with open(results_csv_path, "a", newline='') as fg:
                writer = csv.writer(fg)
                writer.writerow([idx, smi, "Error", "N/A"])
            continue

        # Write Vina config file
        config_path = current_path / 'config.txt'
        with open(config_path, "w") as f_config:
            f_config.write(f"receptor = {receptor_pdbqt_path}\n")
            f_config.write(f"ligand = {lig_pdbqt_path}\n")
            f_config.write(f"opencl_binary_path = {args.vina_path}\n")
            f_config.write(f"center_x = {cent_x}\n")
            f_config.write(f"center_y = {cent_y}\n")
            f_config.write(f"center_z = {cent_z}\n")
            f_config.write(f"size_x = {size_x}\n")
            f_config.write(f"size_y = {size_y}\n")
            f_config.write(f"size_z = {size_z}\n")
            f_config.write(f"num_modes={args.number_of_poses}\n")
            f_config.write("thread = 8000\n")
            #f_config.write("energy_range = 3.0\n")
            f_config.close()

        # Run Vina-GPU docking
        dock_log_path = current_path / 'dock-lig_out.log'
        dock_pdbqt_path = current_path / 'dock-lig_out.pdbqt'
        vina_cmd = (
            f"{args.vina_path}/AutoDock-Vina-GPU-2-1 "
            f"--config {config_path} --rilc_bfgs 1 --search_depth 100 --randomize_only "
            f"--out {dock_pdbqt_path} --log {dock_log_path}"
        )
        os.system(vina_cmd)

        # Clean up config file
        os.remove(config_path)

        if not dock_pdbqt_path.exists() or not dock_log_path.exists():
            print(f"Docking failed for {idx}. Skipping.")
            with open(results_csv_path, "a", newline='') as fg:
                writer = csv.writer(fg)
                writer.writerow([idx, smi, "Docking_Failed", "N/A"])
            if lig_pdbqt_path.exists(): os.remove(lig_pdbqt_path)
            continue

        # Extract best pose from PDBQT and convert to PDB
        # Using OpenBabel to get the first model and convert to PDB
        best_pose_pdb = current_path / f'{idx}_best_pose.pdb'
        os.system(f"obabel -ipdbqt {dock_pdbqt_path} -opdb -O {best_pose_pdb} -f 1 -l 1") # -f 1 -l 1 to get only the first model

        # Modify ATOM to HETATM in the docked ligand PDB (crucial for complex files)
        if best_pose_pdb.exists():
            with open(best_pose_pdb, 'r') as f_in:
                lines = f_in.readlines()
            with open(best_pose_pdb, 'w') as f_out:
                for line in lines:
                    if line.startswith('ATOM '):
                        f_out.write('HETATM' + line[6:])
                    else:
                        f_out.write(line)
        else:
            print(f"Warning: Best pose PDB not created for {idx}. Cannot process further.")
            with open(results_csv_path, "a", newline='') as fg:
                writer = csv.writer(fg)
                writer.writerow([idx, smi, "No_Best_Pose_PDB", "N/A"])
            # Clean up intermediate files before continuing
            if lig_pdbqt_path.exists(): os.remove(lig_pdbqt_path)
            if dock_pdbqt_path.exists(): os.remove(dock_pdbqt_path)
            if dock_log_path.exists(): os.remove(dock_log_path)
            continue

        # Get docking score
        ds = "N/A"
        try:
            with open(dock_log_path, 'r') as f_log:
                log_content = f_log.read()
                for line in log_content.splitlines():
                    if line.strip().startswith('1 '):
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                ds = float(parts[1])
                                if ds <= args.threshold:
                                    print(f"Docking Score for {idx}: {ds} (meets threshold)")
                                else:
                                    print(f"Docking Score for {idx}: {ds} (does not meet threshold {args.threshold})")
                                break
                            except ValueError:
                                ds = "Invalid_Score"
                                print(f"Warning: Could not parse docking score for {idx}.")
        except Exception as e:
            print(f"Error parsing docking log for {idx}: {e}")
            ds = "Log_Parse_Error"

        # Re-assign bond orders and add hydrogens to docked ligand (RDKit) and save as SDF
        lig_dock_out_sdf = docked_ligands_folder / f'{idx}_dock_score{ds}.sdf'
        lig_dock_out_pdb_for_rmsd = current_path / f'{idx}_docked_for_rmsd.pdb' # Temp PDB for RMSD calc

        if mol_rdkit: # Only attempt RDKit processing if original mol was successfully created by RDKit
            try:
                docked_pose_mol = AllChem.MolFromPDBFile(str(best_pose_pdb))
                if docked_pose_mol:
                    # Assign bond orders from template if possible
                    # Ensure molecules have the same number of atoms before assigning bond orders
                    if mol_rdkit.GetNumAtoms() == docked_pose_mol.GetNumAtoms():
                        newMol = AllChem.AssignBondOrdersFromTemplate(mol_rdkit, docked_pose_mol)
                    else:
                        print(f"Warning: Atom count mismatch for {idx}. Skipping bond order assignment.")
                        newMol = docked_pose_mol
                    
                    # Add Hs with coordinates from the docked pose
                    newMol_H = Chem.AddHs(newMol, addCoords=True)
                    Chem.MolToMolFile(newMol_H, str(lig_dock_out_sdf))
                    Chem.MolToPDBFile(newMol_H, str(lig_dock_out_pdb_for_rmsd))
                else:
                    raise ValueError("Could not load docked PDB pose with RDKit.")
            except Exception as e:
                print(f"RDKit processing of docked pose failed for {idx}: {e}. Falling back to obabel for SDF/PDB output.")
                # Fallback for SDF/PDB conversion if RDKit fails
                os.system(f'obabel -ipdb {best_pose_pdb} -osdf -O {lig_dock_out_sdf}')
                os.system(f'obabel -ipdb {best_pose_pdb} -opdb -O {lig_dock_out_pdb_for_rmsd}') # For RMSD calculation
        else: # If RDKit failed for initial 3D generation, rely on OpenBabel for this step too
            print(f"RDKit was not used for initial 3D generation for {idx}. Using OpenBabel for final SDF/PDB.")
            os.system(f'obabel -ipdb {best_pose_pdb} -osdf -O {lig_dock_out_sdf}')
            os.system(f'obabel -ipdb {best_pose_pdb} -opdb -O {lig_dock_out_pdb_for_rmsd}')

        # RMSD Calculation (if reference ligand exists)
        crystal_rmsd = "N/A"
        if args.lig_id and ldb_path and ldb_path.exists() and lig_dock_out_pdb_for_rmsd.exists():
            try:
                docked_ligand_mol_rmsd = Chem.MolFromPDBFile(str(lig_dock_out_pdb_for_rmsd))
                crystal_ligand_mol_rmsd = Chem.MolFromPDBFile(str(ldb_path))

                if docked_ligand_mol_rmsd and crystal_ligand_mol_rmsd:
                    rms = CalcLigRMSD(crystal_ligand_mol_rmsd, docked_ligand_mol_rmsd, rename_lig2=False) # No need to rename for RMSD calc temp file
                    if rms != float('inf'):
                        crystal_rmsd = round(rms, 2)
                    else:
                        crystal_rmsd = "NoMCS" # Indicates no common substructure for RMSD calculation
                else:
                    print(f"Warning: Could not load RDKit Mol objects for RMSD calculation for {idx}.")
                    crystal_rmsd = "ErrorLoadingMols"

            except Exception as e:
                print(f"Error calculating RMSD for {idx}: {e}")
                crystal_rmsd = "RMSD_Calc_Error"
        else:
            print(f"RMSD calculation skipped for {idx}. Reference ligand not provided or files missing.")


        # Create Protein-Ligand Complex PDB
        # Only create complex if docking score meets threshold (if threshold is not default -1.0)
        if isinstance(ds, float) and ds <= args.threshold:
            try:
                with open(ppdb_path, 'r') as fp1:
                    receptor_data = fp1.read().replace('END', '') # Remove END before appending
                with open(best_pose_pdb, 'r') as fp2: # Use the best_pose_pdb (with HETATM)
                    ligand_data = fp2.read()

                complex_data = receptor_data + "TER\n" + ligand_data + "END\n" # Add TER and END
                
                # Format score for filename
                formatted_ds = f"{ds:.2f}"#.replace('.', '_').replace('-', 'neg_')
                complex_filename = complex_folder / f'{idx}_{protein_name}_score{formatted_ds}.pdb'

                with open(complex_filename, 'w') as fp3:
                    fp3.write(complex_data)
                print(f"Complex PDB saved to: {complex_filename}")
            except Exception as e:
                print(f"Error creating complex PDB for {idx}: {e}")

        # Clean up temporary ligand PDB file used for RMSD and the best_pose_pdb
        if lig_dock_out_pdb_for_rmsd.exists():
            os.remove(lig_dock_out_pdb_for_rmsd)
        if best_pose_pdb.exists():
            os.remove(best_pose_pdb)

        # Write results to CSV
        with open(results_csv_path, "a", newline='') as fg:
            writer = csv.writer(fg)
            writer.writerow([idx, smi, ds, crystal_rmsd])

        # Clean up common intermediate files for the current ligand
        if dock_pdbqt_path.exists(): os.remove(dock_pdbqt_path)
        if dock_log_path.exists(): os.remove(dock_log_path)
        if lig_pdbqt_path.exists(): os.remove(lig_pdbqt_path)


    # --- Final Cleanup ---
    print("\nCleaning up remaining temporary files...")
    # Using glob for patterns and pathlib for existence check
    files_to_clean = [
        current_path / 'ligands_file.csv',
        current_path / 'new.log',
        current_path / 'RMSD_renamed.pdb', # From CalcLigRMSD if rename_lig2 was True
        current_path / 'dummy_dock.pdb', # If created by some internal process not in snippet
        #current_path / 'whole_protein_pocket.txt',
        #current_path / 'Ref_Lig_Based_pocket.txt',
        #current_path / 'user_defined_pocket.txt',
        current_path / 'dummy.csv',
        # Clean any stray pdb, sdf, pdbqt, log files in the current directory from failed runs
    ]
    if args.lig_id:
        files_to_clean.append(current_path / f'ref_{args.lig_id}_lig.pdb')

    for f_path in files_to_clean:
        if f_path.exists():
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"Error removing file {f_path}: {e}")

    # Remove any stray pdbqt, log, or sdf files that might have been left
    for f_ext in ['*.pdbqt', '*.log', '*.sdf', 'final_model.pdb']:#, '*.pdb']:
        for f_path in current_path.glob(f_ext):
            try:
                os.remove(f_path)
            except OSError as e:
                print(f"Error removing file {f_path}: {e}")

    # Remove __pycache__ if it exists
    pycache_dir = current_path / '__pycache__'
    if pycache_dir.exists() and pycache_dir.is_dir():
        shutil.rmtree(pycache_dir)
        print(f"Removed {pycache_dir}")


    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"\n##########################################")
    print(f"The Computation Time: {total_time_minutes:.2f} Minutes")
    print(f"Results saved to: {results_csv_path}")
    print("##########################################")

if __name__ == "__main__":
    main()