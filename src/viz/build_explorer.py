import pandas as pd
import numpy as np
import os
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, CustomJS, TapTool, Div, NumeralTickFormatter, TabPanel, Tabs
from bokeh.layouts import column

def build_standalone_html():
    """
    Loads mock GWAS data and builds a robust, standalone, interactive Bokeh dashboard.
    """
    print("--- Building GWAS Explorer HTML ---")

    # --- 1. Load Data ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../'))
    data_path = os.path.join(project_root, 'data/')
    output_path = os.path.join(project_root, 'output/')
    
    try:
        gwas_full_df = pd.read_csv(os.path.join(data_path, 'mock_gwas_full_scan.csv'))
        verified_hits_df = pd.read_csv(os.path.join(data_path, 'mock_verification_hits.csv'))
        gene_annotations_df = pd.read_csv(os.path.join(data_path, 'mock_gene_annotations.csv'))
        print("All mock data files loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure mock data is generated."); return

    # --- 2. Prepare Data Sources ---
    gwas_full_df['-log10(p)_display'] = gwas_full_df['-log10(p)'].clip(upper=12)
    contig_order = ['2L', '2R', '3L', '3R', 'X']; contig_starts = {}
    gwas_full_df['contig_cat'] = pd.Categorical(gwas_full_df['contig'], categories=contig_order, ordered=True)
    gwas_full_df = gwas_full_df.sort_values(['contig_cat', 'pos'])

    cumulative_pos = 0; contig_centers = {}
    gwas_full_df['pos_cumulative'] = 0
    for contig in contig_order:
        subset = gwas_full_df[gwas_full_df['contig'] == contig]
        if not subset.empty:
            gwas_full_df.loc[subset.index, 'pos_cumulative'] = subset['pos'] + cumulative_pos
            contig_starts[contig] = cumulative_pos
            contig_centers[contig] = cumulative_pos + subset['pos'].median()
            cumulative_pos += subset['pos'].max()

    gwas_full_df['color'] = gwas_full_df['contig_cat'].cat.codes.apply(lambda x: '#2c3e50' if x % 2 == 0 else '#7f8c8d')
    verified_hits_df['-log10(p)'] = -np.log10(verified_hits_df['p_value'])

    # Downsample for performance
    n_points_to_plot = 150000
    significant_points = gwas_full_df[gwas_full_df['-log10(p)'] > 4]
    non_significant_points = gwas_full_df[gwas_full_df['-log10(p)'] <= 4]
    n_background = n_points_to_plot - len(significant_points)
    background_sample = non_significant_points.sample(n=min(n_background, len(non_significant_points)), random_state=42)
    plot_df = pd.concat([significant_points, background_sample])
    
    # Bokeh ColumnDataSources
    full_scan_source = ColumnDataSource(plot_df)
    regional_scan_source = ColumnDataSource(data={'pos': [], '-log10(p)': []}) 
    regional_hits_source = ColumnDataSource(data={'pos': [], '-log10(p)': [], 'log_odds_mixed': [], 'ci_lower_mixed': [], 'ci_upper_mixed': []})
    gene_source = ColumnDataSource(data={'start': [], 'end': [], 'gene_name': [], 'y': []})
    detail_title = Div(text="<h3>SNP Details (Click a red point in the Regional Plot)</h3>", width=800)

    # --- 3. Create Plots and Widgets ---
    p1 = figure(height=400, width=1200, title="GWAS Explorer: Genome-Wide Manhattan Plot", tools="pan,wheel_zoom,box_select,reset,save", active_drag="box_select")
    p1.add_tools(HoverTool(tooltips=[("Contig", "@contig"), ("Position", "@pos"), ("P-value", "@p_value{1.1e}")]))
    p1.scatter(x='pos_cumulative', y='-log10(p)_display', color='color', source=full_scan_source, alpha=0.7, size=4)
    p1.xaxis.ticker = list(contig_centers.values()); p1.xaxis.major_label_overrides = {int(v): k for k, v in contig_centers.items()}
    p1.x_range.range_padding = 0.02; p1.yaxis.axis_label = "-log10(p-value)"; p1.xaxis.axis_label = "Genomic Position"
    
    p2_hover = HoverTool(tooltips=[("Position", "@pos"), ("Log-Odds", "@log_odds_mixed{%.2f}"), ("95% CI", "[@{ci_lower_mixed}{%.2f}, @{ci_upper_mixed}{%.2f}]")], name="verified_hits_glyph")
    p2 = figure(height=400, width=1200, title="Regional Plot (Select a region on the Manhattan Plot to zoom)", tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', TapTool(), p2_hover])
    p2.scatter(x='pos', y='-log10(p)', source=regional_scan_source, size=5, color="gray", alpha=0.6, legend_label="All Scan SNPs")
    p2.scatter(x='pos', y='-log10(p)', source=regional_hits_source, size=8, color="red", legend_label="Verified Hits (Mixed-Effects)", name="verified_hits_glyph")
    p2.quad(top=0.2, bottom=-0.2, left='start', right='end', source=gene_source, fill_color="blue", line_color=None, alpha=0.3, legend_label="Genes")
    p2.legend.location = "top_left"; p2.legend.click_policy="hide"; p2.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p2.yaxis.axis_label = "-log10(p-value)"; p2.xaxis.axis_label = "Position on Contig"
    
    # --- 4. Create Tabs ---
    tabs = Tabs(tabs=[
        TabPanel(child=column(p1), title="1. Genome-Wide Scan"),
        TabPanel(child=column(p2), title="2. Regional Detail"),
        TabPanel(child=column(detail_title), title="3. SNP Detail")
    ])

    # --- 5. Define Interactivity with CustomJS ---
    box_select_callback = CustomJS(
        args=dict(full_scan_source_all=ColumnDataSource(gwas_full_df), regional_scan_source=regional_scan_source,
                  all_hits_data=ColumnDataSource(verified_hits_df), regional_hits_source=regional_hits_source,
                  all_genes=ColumnDataSource(gene_annotations_df), gene_source=gene_source, 
                  p2=p2, tabs=tabs, contig_starts=contig_starts, contig_order=contig_order, detail_title=detail_title),
        code="""
            const geometry = cb_obj.geometry;
            const selection_start = geometry.x0;
            const selection_end = geometry.x1;
            
            if (selection_start == null) { return; }

            let selected_contig = null, contig_offset = 0;
            for (const contig of contig_order) {
                const start_pos = contig_starts[contig];
                if (selection_start >= start_pos) {
                    selected_contig = contig;
                    contig_offset = start_pos;
                } else { break; }
            }
            if (!selected_contig) { return; }
            
            const min_pos = selection_start - contig_offset;
            const max_pos = selection_end - contig_offset;

            detail_title.text = "<h3>SNP Details (Click a red point in the Regional Plot)</h3>";

            const all_scan = full_scan_source_all.data;
            const new_regional_scan = {pos: [], '-log10(p)': []};
            let max_logp = 0;
            for (let i=0; i < all_scan.pos.length; i++) {
                if (all_scan.contig[i] === selected_contig && all_scan.pos[i] >= min_pos && all_scan.pos[i] <= max_pos) {
                    new_regional_scan.pos.push(all_scan.pos[i]);
                    const logp = all_scan['-log10(p)'][i];
                    new_regional_scan['-log10(p)'].push(logp);
                    if (logp > max_logp) max_logp = logp;
                }
            }
            regional_scan_source.data = new_regional_scan;

            const all_hits = all_hits_data.data;
            const new_regional_hits = {pos: [], '-log10(p)': [], log_odds_mixed: [], ci_lower_mixed: [], ci_upper_mixed: []};
            for (let i=0; i < all_hits.pos.length; i++) {
                if (all_hits.contig[i] === selected_contig && all_hits.pos[i] >= min_pos && all_hits.pos[i] <= max_pos) {
                    new_regional_hits.pos.push(all_hits.pos[i]);
                    new_regional_hits['-log10(p)'].push(all_hits['-log10(p)'][i]);
                    new_regional_hits.log_odds_mixed.push(all_hits.log_odds_mixed[i]);
                    new_regional_hits.ci_lower_mixed.push(all_hits.ci_lower_mixed[i]);
                    new_regional_hits.ci_upper_mixed.push(all_hits.ci_upper_mixed[i]);
                }
            }
            regional_hits_source.data = new_regional_hits;

            const all_gene_data = all_genes.data;
            const new_gene = {start: [], end: [], gene_name: [], y: []};
            for (let i=0; i < all_gene_data.start.length; i++) {
                if (all_gene_data.contig[i] === selected_contig && all_gene_data.end[i] >= min_pos && all_gene_data.start[i] <= max_pos) {
                    new_gene.start.push(all_gene_data.start[i]);
                    new_gene.end.push(all_gene_data.end[i]);
                    new_gene.gene_name.push(all_gene_data.gene_name[i]);
                    new_gene.y.push(-0.5);
                }
            }
            gene_source.data = new_gene;

            p2.x_range.start = min_pos; p2.x_range.end = max_pos;
            p2.y_range.start = -1; p2.y_range.end = max_logp > 0 ? max_logp * 1.1 : 10;
            p2.title.text = `Regional Plot: ${selected_contig} from ${Math.round(min_pos).toLocaleString()} to ${Math.round(max_pos).toLocaleString()}`;
            
            p2.change.emit();
            tabs.active = 1;
        """
    )
    p1.js_on_event('selectiongeometry', box_select_callback)
    
    tap_callback = CustomJS(
        args=dict(source=regional_hits_source, detail_title=detail_title, tabs=tabs),
        code="""
            const indices = source.selected.indices;
            if (!indices || indices.length == 0) { return; }
            const i = indices[0];
            const data = source.data;
            const pos = data['pos'][i].toLocaleString();
            const log_odds = data['log_odds_mixed'][i].toFixed(2);
            const lower = data['ci_lower_mixed'][i].toFixed(2);
            const upper = data['ci_upper_mixed'][i].toFixed(2);
            
            detail_title.text = `
                <h3>SNP Details: Position ${pos}</h3>
                <p><b>Mixed-Effects Model Result:</b></p>
                <ul>
                    <li><b>Log-Odds Ratio:</b> ${log_odds}</li>
                    <li><b>95% Confidence Interval:</b> [${lower}, ${upper}]</li>
                </ul>
            `;
            tabs.active = 2;
        """
    )
    # Correctly select the GLYPH renderer and attach the callback to its data source's selection
    p2.select(name="verified_hits_glyph")[0].data_source.selected.js_on_change('indices', tap_callback)
    
    # --- 6. Save ---
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_html_path = os.path.join(output_path, "gwas_explorer.html")
    output_file(output_html_path, title="GWAS Explorer")
    show(tabs)
    print(f"Successfully generated standalone HTML file at: {os.path.abspath(output_html_path)}")

if __name__ == '__main__':
    build_standalone_html()