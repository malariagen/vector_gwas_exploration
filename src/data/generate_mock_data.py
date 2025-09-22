import pandas as pd
import numpy as np
import os


def _inject_ld_peak(df, contig, peak_center, max_log10_p, peak_width=200_000):
    """Injects a realistic, spiky LD peak into the GWAS data."""
    print(f"Injecting a realistic peak on {contig} at position {peak_center:,}...")
    # Find all SNPs within the defined window for the peak
    peak_region = df[
        (df["contig"] == contig)
        & (df["pos"] > peak_center - peak_width)
        & (df["pos"] < peak_center + peak_width)
    ].copy()

    if peak_region.empty:
        return df

    # Calculate distance from the peak center
    distance = np.abs(peak_region["pos"] - peak_center)

    # Simulate LD decay: p-value significance decreases exponentially with distance
    # The decay rate is tuned to create a spiky but extended peak
    decay_rate = 50 / peak_width
    new_log10_p = max_log10_p * np.exp(-decay_rate * distance)

    # Add some random noise to make it look less perfect
    noise = np.random.normal(0, 0.2, len(new_log10_p))
    noisy_log10_p = new_log10_p + noise

    # Ensure we don't go below the background noise level
    final_log10_p = np.maximum(noisy_log10_p, peak_region["-log10(p)"])

    # Update the main DataFrame
    df.loc[peak_region.index, "-log10(p)"] = final_log10_p
    return df


def _generate_textured_background(n_snps):
    """Generates a non-uniform background noise that mimics a real GWAS floor."""
    # An exponential distribution creates a floor with many low-p-value points
    # and a tail of slightly more significant noise, which is more realistic.
    return np.random.exponential(scale=0.4, size=n_snps)


def _add_ld_bands(df, contig_lengths, n_bands=20, band_strength=1.5):
    """Adds horizontal bands to simulate large LD blocks."""
    print("Adding horizontal LD bands for texture...")
    for _ in range(n_bands):
        # Pick a random chromosome and position for the band
        contig = np.random.choice(list(contig_lengths.keys()))
        band_width = np.random.randint(200_000, 1_000_000)
        start_pos = np.random.randint(1, contig_lengths[contig] - band_width)
        end_pos = start_pos + band_width

        # Assign all SNPs in this block a similar p-value from a narrow distribution
        band_mask = (
            (df["contig"] == contig) & (df["pos"] >= start_pos) & (df["pos"] <= end_pos)
        )
        n_snps_in_band = band_mask.sum()
        if n_snps_in_band > 0:
            band_p_values = np.random.normal(
                loc=band_strength, scale=0.1, size=n_snps_in_band
            )
            df.loc[band_mask, "-log10(p)"] = np.maximum(
                df.loc[band_mask, "-log10(p)"], band_p_values
            )
    return df


def generate_all_mock_data():
    """
    Generates a more realistic three-part mock dataset for the GWAS Explorer,
    including spiky peaks and a textured background.
    """
    print("--- Starting Realistic Mock Data Generation ---")

    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    output_dir = os.path.join(project_root, "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Generate Genome-Wide Scan Data---
    print("Simulating genome-wide scan data...")
    contigs = ["2L", "2R", "3L", "3R", "X"]
    contig_lengths = {
        "2L": 49_364_325,
        "2R": 61_545_105,
        "3L": 41_963_435,
        "3R": 53_200_684,
        "X": 24_393_108,
    }
    n_snps_per_chrom = 100_000
    all_dfs = []

    for contig in contigs:
        pos = np.sort(np.random.randint(1, contig_lengths[contig], n_snps_per_chrom))
        # Generate a more realistic, textured background noise
        log10_p_background = _generate_textured_background(n_snps_per_chrom)
        df = pd.DataFrame(
            {"contig": contig, "pos": pos, "-log10(p)": log10_p_background}
        )
        all_dfs.append(df)

    gwas_full_df = pd.concat(all_dfs, ignore_index=True)

    # Add horizontal bands to mimic large LD blocks
    gwas_full_df = _add_ld_bands(
        gwas_full_df, contig_lengths, n_bands=50, band_strength=1.5
    )

    # Inject realistic, spiky peaks
    gwas_full_df = _inject_ld_peak(
        gwas_full_df, contig="2L", peak_center=2_422_652, max_log10_p=8.5
    )
    gwas_full_df = _inject_ld_peak(
        gwas_full_df, contig="3R", peak_center=21_430_000, max_log10_p=6.0
    )

    # Finalize the p-value column from the -log10(p)
    gwas_full_df["p_value"] = 10 ** (-gwas_full_df["-log10(p)"])
    gwas_full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    gwas_full_df.dropna(subset=["-log10(p)"], inplace=True)

    output_path_full = os.path.join(output_dir, "mock_gwas_full_scan.csv")
    gwas_full_df.to_csv(output_path_full, index=False)
    print(
        f"Generated {len(gwas_full_df):,} genome-wide SNP results -> {output_path_full}"
    )

    # --- 2. Generate Verification Hits Data ---
    print("Simulating verification data for top hits...")
    significant_hits_df = gwas_full_df[gwas_full_df["-log10(p)"] > 5].copy()
    n_hits = len(significant_hits_df)
    significant_hits_df["log_odds_mixed"] = np.random.normal(1.2, 0.3, n_hits)
    significant_hits_df["ci_lower_mixed"] = significant_hits_df[
        "log_odds_mixed"
    ] - np.random.uniform(0.1, 0.3, n_hits)
    significant_hits_df["ci_upper_mixed"] = significant_hits_df[
        "log_odds_mixed"
    ] + np.random.uniform(0.1, 0.3, n_hits)
    output_path_verified = os.path.join(output_dir, "mock_verification_hits.csv")
    significant_hits_df.to_csv(output_path_verified, index=False)
    print(
        f"Generated verification data for {n_hits} significant SNPs -> {output_path_verified}"
    )

    # --- 3. Generate Gene Annotation Data ---
    print("Creating mock gene annotation data...")
    gene_data = {
        "contig": ["2L", "2L", "3R"],
        "start": [2_358_158, 2_800_000, 21_400_000],
        "end": [2_431_617, 2_805_000, 21_408_000],
        "gene_name": ["Vgsc", "GeneB", "GeneC"],
        "strand": ["+", "-", "+"],
    }
    gene_annotations_df = pd.DataFrame(gene_data)
    output_path_genes = os.path.join(output_dir, "mock_gene_annotations.csv")
    gene_annotations_df.to_csv(output_path_genes, index=False)
    print(f"Saved mock gene annotations -> {output_path_genes}")
    print("\n--- Mock Data Generation Complete ---")


if __name__ == "__main__":
    generate_all_mock_data()
