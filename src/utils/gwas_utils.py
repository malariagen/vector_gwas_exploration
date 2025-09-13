import pandas as pd
from typing import List, Tuple

def parse_regions(ag3, regions: str | list[str]) -> List[Tuple[str, int, int]]:
    """
    Parses a variety of region inputs into a standardized list of (contig, start, end) tuples.

    Parameters
    ----------
    ag3 : Ag3
        An instantiated Ag3 data resource object.
    regions : str or list of str
        The genomic region(s) to parse. Can be a contig name ('2L'), a region
        string ('2L:1-10,000,000'), or a list of these.

    Returns
    -------
    list of tuples
        A list where each tuple is (contig, start_pos, end_pos).
    """
    if regions is None:
        # If no region is specified, default to all major chromosome arms
        regions_to_parse = ag3.contigs
    elif isinstance(regions, str):
        regions_to_parse = [regions]
    else:
        regions_to_parse = regions
        
    parsed = []
    for r in regions_to_parse:
        if ":" in r:
            contig = r.split(":")[0]
            start_pos, end_pos = map(int, r.split(":")[1].replace(",", "").split("-"))
        else:
            contig = r
            start_pos = 1
            end_pos = ag3.genome_sequence(contig).shape[0]
        parsed.append((contig, start_pos, end_pos))
        
    return parsed