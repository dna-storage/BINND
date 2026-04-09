import primer3
import logging

def calculate_deltag_primer3(seq1:str, seq2:str, temperature:float, sodium_concentration:float, magnesium_concentration:float) -> float:
    """Calculate deltaG using primer3 library.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        temperature (float): The temperature in degrees Celsius.
        sodium_concentration (float): The sodium concentration in M.
        magnesium_concentration (float): The magnesium concentration in M.
    Returns:
        float: The calculated deltaG value in kcal/mol.
    """
    
    model = primer3.thermoanalysis.ThermoAnalysis(temp_c=temperature, 
                                        mv_conc=sodium_concentration * 1000, 
                                        dv_conc=magnesium_concentration * 1000)

    result = model.calc_heterodimer(seq1, seq2)
    result = result.todict()
    
    dG = result['dg']
    dG = round(dG / 1000, 6)  # Convert from cal/mol to kcal/mol and round to 6 decimal places
    return dG

def test_primer3_deltag():
    seq1 = "CGAGATGGGGGTGGTTAGGC"
    seq2 = "ACGACTTTAGCCTAACCAGT"
    temperature = 25.0 # degrees Celsius
    sodium_concentration = 0.05  # 50 mM
    magnesium_concentration = 0.0

    dG = calculate_deltag_primer3(seq1, seq2, temperature, sodium_concentration, magnesium_concentration)
    print(f"Calculated deltaG: {dG} kcal/mol")
    
if __name__ == "__main__":
    test_primer3_deltag()