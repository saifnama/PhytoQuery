"""
RDKit molecular structure image generator.

Provides utilities to render molecular structures from SMILES strings.
"""

import base64
import io
from typing import Optional


def smiles_to_image_base64(
    smiles: str,
    width: int = 300,
    height: int = 200,
    bond_line_width: int = 1.5,
    font_size: int = 12
) -> Optional[str]:
    """
    Convert a SMILES string to a base64-encoded PNG image of the molecular structure.
    
    Args:
        smiles: SMILES string representing the molecule
        width: Image width in pixels
        height: Image height in pixels  
        bond_line_width: Width of bond lines
        font_size: Font size for atom labels
    
    Returns:
        Base64-encoded PNG image string, or None if conversion fails
    """
    if not smiles or not smiles.strip():
        return None
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Draw molecule to image (using default options)
        img = Draw.MolToImage(
            mol,
            size=(width, height)
        )
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    except Exception as e:
        print(f"Error rendering SMILES '{smiles}': {e}")
        return None


def get_mol_info(smiles: str) -> Optional[dict]:
    """
    Get molecular information from SMILES.
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary with molecular weight and formula, or None if invalid
    """
    if not smiles or not smiles.strip():
        return None
    
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, MolFormula
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "formula": str(MolFormula.MolFormula(mol)),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
        }
    
    except Exception as e:
        print(f"Error getting mol info for SMILES '{smiles}': {e}")
        return None


if __name__ == "__main__":
    # Test with a few common molecules
    test_smiles = [
        "CCO",  # ethanol
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "CC1=CC=C(C=C1)C(=O)O",  # benzoic acid
    ]
    
    for smi in test_smiles:
        print(f"\nTesting: {smi}")
        img = smiles_to_image_base64(smi)
        if img:
            print(f"  Generated image: {len(img)} bytes (base64)")
        else:
            print("  Failed to generate image")