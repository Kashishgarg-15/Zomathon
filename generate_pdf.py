"""
Convert submission.md to PDF using fpdf2.
Produces analysis_output/submission.pdf (<1MB target).
"""
import json, os, re, textwrap
from fpdf import FPDF

# Load data
with open('analysis_output/analysis_results.json') as f:
    analysis = json.load(f)
with open('analysis_output/latency_results.json') as f:
    latency = json.load(f)
with open('analysis_output/submission.md', 'r', encoding='utf-8') as f:
    md_content = f.read()

lat = latency['gbdt_latency']
biz = analysis['business_projections']

class SubmissionPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        if self.page_no() > 1:
            self.set_font('Helvetica', 'I', 7)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, 'Zomathon - CSAO Rail Recommendation System', align='C')
            self.ln(3)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
    
    def footer(self):
        self.set_y(-12)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(128, 128, 128)
        self.cell(0, 8, f'Page {self.page_no()}/{{nb}}', align='C')
    
    def section_title(self, title, level=1):
        if level == 1:
            self.set_font('Helvetica', 'B', 14)
            self.set_text_color(0, 100, 180)
            self.ln(4)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(0, 100, 180)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(3)
        elif level == 2:
            self.set_font('Helvetica', 'B', 11)
            self.set_text_color(40, 40, 40)
            self.ln(3)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
        elif level == 3:
            self.set_font('Helvetica', 'B', 9.5)
            self.set_text_color(60, 60, 60)
            self.ln(2)
            self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)
    
    def body_text(self, text):
        self.set_font('Helvetica', '', 8.5)
        self.set_text_color(30, 30, 30)
        self.set_x(10)
        self.multi_cell(190, 4.5, text)
        self.ln(1)
    
    def bullet_item(self, text):
        self.set_font('Helvetica', '', 8.5)
        self.set_text_color(30, 30, 30)
        self.set_x(10)
        self.multi_cell(190, 4.5, '  - ' + text)
    
    def code_block(self, text):
        self.set_font('Courier', '', 7)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(245, 245, 245)
        lines = text.split('\n')
        for line in lines:
            # Replace box drawing chars with ASCII equivalents
            line = line.replace('\u250c', '+').replace('\u2510', '+')
            line = line.replace('\u2514', '+').replace('\u2518', '+')
            line = line.replace('\u251c', '+').replace('\u2524', '+')
            line = line.replace('\u252c', '+').replace('\u2534', '+')
            line = line.replace('\u253c', '+')
            line = line.replace('\u2500', '-').replace('\u2502', '|')
            line = line.replace('\u2530', '+').replace('\u2538', '+')
            line = line.replace('\u2550', '=').replace('\u2551', '|')
            line = line.replace('\u256c', '+')
            line = line.replace('\u2560', '+').replace('\u2563', '+')
            line = line.replace('\u2566', '+').replace('\u2569', '+')
            line = line.replace('\u2554', '+').replace('\u2557', '+')
            line = line.replace('\u255a', '+').replace('\u255d', '+')
            line = line.replace('\u2591', '#')
            line = line.replace('\u25bc', 'v').replace('\u25b6', '>')
            line = line.replace('\u25b2', '^').replace('\u25c0', '<')
            line = line.replace('\u2190', '<-').replace('\u2192', '->')
            line = line.replace('\u2191', '^').replace('\u2193', 'v')
            line = line.replace('\u2588', '#')
            line = line.replace('\u25cf', '*')
            line = line.replace('\u2022', '*')
            line = line.replace('\u21b3', '|_')
            line = line.replace('\u2198', '\\')
            line = line.replace('\u2197', '/')
            line = line.replace('\u2196', '\\')
            line = line.replace('\u2199', '/')
            line = line.replace('\u2713', '[OK]')
            line = line.replace('\u2717', '[X]')
            line = line.replace('\u25a0', '#')
            line = line.replace('\u25a1', '[ ]')
            line = line.replace('\u2b9e', '=>')
            line = line.replace('\u21d2', '=>')
            line = line.replace('\u25b6', '>')
            line = line.replace('\u25c6', '*')
            line = line.replace('\u2605', '*')
            line = line.replace('\u2606', '*')
            line = line.replace('\u21d3', 'v')
            line = line.replace('\u2193', 'v')
            line = line.replace('\u25bc', 'v')
            line = line.replace('\u21d0', '<=')
            line = line.replace('\u2190', '<-')
            line = line.replace('\u21d1', '^')
            # fallback: replace any non-latin1
            line = line.encode('latin-1', errors='replace').decode('latin-1')
            # Truncate very long lines
            if len(line) > 95:
                line = line[:92] + '...'
            self.cell(0, 3.5, '  ' + line, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)
    
    def add_table(self, headers, rows):
        self.set_font('Helvetica', 'B', 7.5)
        self.set_fill_color(0, 100, 180)
        self.set_text_color(255, 255, 255)
        
        n_cols = len(headers)
        col_w = (190) / n_cols
        
        # Variable column widths based on content
        if n_cols <= 3:
            col_w = 190 / n_cols
            widths = [col_w] * n_cols
        elif n_cols == 4:
            widths = [50, 45, 45, 50]
        elif n_cols == 5:
            widths = [40, 30, 35, 35, 50]
        else:
            widths = [190 / n_cols] * n_cols
        
        # Adjust widths to sum to 190
        total = sum(widths)
        widths = [w * 190 / total for w in widths]
        
        for i, h in enumerate(headers):
            self.cell(widths[i], 5, clean_text(h), border=1, align='C', fill=True)
        self.ln()
        
        self.set_font('Helvetica', '', 7.5)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(rows):
            fill = row_idx % 2 == 1
            if fill:
                self.set_fill_color(240, 245, 255)
            else:
                self.set_fill_color(255, 255, 255)
            
            # Check for bold row (our model)
            is_bold = any('**' in str(c) for c in row)
            if is_bold:
                self.set_font('Helvetica', 'B', 7.5)
                self.set_fill_color(230, 255, 230)
            
            for i, cell in enumerate(row):
                cell_text = clean_text(str(cell))
                self.cell(widths[i], 5, cell_text, border=1, align='C', fill=True)
            self.ln()
            
            if is_bold:
                self.set_font('Helvetica', '', 7.5)
        self.ln(2)


def clean_text(text):
    """Remove markdown formatting for plain text"""
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # italic
    text = re.sub(r'`([^`]+)`', r'\1', text)  # code
    text = text.replace('\u2605', '*')  # star
    text = text.replace('\u20b9', 'Rs ')  # rupee sign
    text = text.replace('\u2713', ' [OK]')  # checkmark  
    text = text.replace('\u2192', '->')  # arrow
    text = text.replace('\u2014', '--')  # em dash
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2018', "'")  # left single quote
    text = text.replace('\u2019', "'")  # right single quote
    text = text.replace('\u201c', '"')  # left double quote
    text = text.replace('\u201d', '"')  # right double quote
    text = text.replace('\u2026', '...')  # ellipsis
    text = text.replace('\u2264', '<=')  # less than or equal
    text = text.replace('\u2265', '>=')  # greater than or equal
    text = text.replace('\u2208', 'in')  # element of
    # Replace any remaining non-latin1 characters
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text.strip()


# ================================================================
# BUILD PDF
# ================================================================
pdf = SubmissionPDF()
pdf.alias_nb_pages()

# --- Title Page ---
pdf.add_page()
pdf.ln(30)
pdf.set_font('Helvetica', 'B', 24)
pdf.set_text_color(0, 100, 180)
pdf.cell(0, 12, 'Cart Super Add-On (CSAO)', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 12, 'Rail Recommendation System', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, 'Zomathon Hackathon Submission', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font('Helvetica', 'I', 11)
pdf.cell(0, 8, 'ML-Powered Cross-Sell Recommendations for Zomato Cart Page', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.ln(20)

# Key results box
pdf.set_fill_color(240, 248, 255)
pdf.set_draw_color(0, 100, 180)
pdf.rect(25, pdf.get_y(), 160, 45, style='DF')
pdf.ln(3)
pdf.set_font('Helvetica', 'B', 12)
pdf.set_text_color(0, 100, 180)
pdf.cell(0, 7, 'Key Results', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(30, 30, 30)
pdf.cell(0, 6, f'3-Model Ensemble: AUC = 0.902  |  NDCG@10 = 0.876  |  Hit@10 = 0.999', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, f'+19.5% AUC / +11.6% NDCG@10 improvement over best baseline', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, f'70 features (23 LLM-derived)  |  P95 inference: {lat["p95_ms"]:.0f}ms  |  AOV lift: +Rs {biz["aov_lift_per_order"]:.0f}/order', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 6, 'LightGBM + XGBoost + DCN-v2 ensemble with Optuna hyperparameter tuning', align='C', new_x="LMARGIN", new_y="NEXT")

# GitHub + Notebook links
pdf.ln(10)
pdf.set_font('Helvetica', 'B', 10)
pdf.set_text_color(0, 100, 180)
pdf.cell(0, 7, 'GitHub Repository: https://github.com/Kashishgarg-15/Zomathon', align='C', new_x="LMARGIN", new_y="NEXT")
pdf.set_font('Helvetica', '', 9)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 6, 'Notebook: CSAO_Recommendation_System.ipynb (in repository root)', align='C', new_x="LMARGIN", new_y="NEXT")

# --- Parse Markdown and build pages ---
pdf.add_page()

# Process markdown line by line
lines = md_content.split('\n')
i = 0
in_code = False
code_buffer = []

while i < len(lines):
    line = lines[i]
    
    # Skip title page content (already rendered)
    if i < 8:
        i += 1
        continue
    
    # Code blocks
    if line.strip() == '```':
        if in_code:
            pdf.code_block('\n'.join(code_buffer))
            code_buffer = []
            in_code = False
        else:
            in_code = True
        i += 1
        continue
    
    if in_code:
        code_buffer.append(line)
        i += 1
        continue
    
    # Headers
    if line.startswith('# ') and not line.startswith('## '):
        title = line[2:].strip()
        if title and 'Cart Super Add-On' not in title:
            pdf.section_title(clean_text(title), level=1)
        i += 1
        continue
    
    if line.startswith('## '):
        pdf.section_title(clean_text(line[3:].strip()), level=2)
        i += 1
        continue
    
    if line.startswith('### '):
        pdf.section_title(clean_text(line[4:].strip()), level=3)
        i += 1
        continue
    
    # Tables
    if line.startswith('| ') and i + 1 < len(lines) and '---' in lines[i + 1]:
        headers = [c.strip() for c in line.split('|')[1:-1]]
        i += 2  # skip header and separator
        rows = []
        while i < len(lines) and lines[i].startswith('| '):
            row = [c.strip() for c in lines[i].split('|')[1:-1]]
            rows.append(row)
            i += 1
        pdf.add_table(headers, rows)
        continue
    
    # Bullets
    if line.startswith('- '):
        pdf.bullet_item(clean_text(line[2:].strip()))
        i += 1
        continue
    
    # Horizontal rule
    if line.strip() == '---':
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        i += 1
        continue
    
    # Regular text
    text = line.strip()
    if text and text != '```':
        pdf.body_text(clean_text(text))
    
    i += 1

# --- Embed Key Visualizations ---
plot_pages = [
    ('Model Evaluation: LightGBM + XGBoost (GBDT v3)', 'model_output_v3/evaluation_plots.png'),
    ('DCN-v2 Deep & Cross Network: Training Results', 'model_output_dcn_v2/dcn_v2_plots.png'),
    ('Final 3-Model Ensemble Comparison', 'model_output_final_v2/final_comparison.png'),
]

for title, path in plot_pages:
    if os.path.exists(path):
        pdf.add_page()
        pdf.section_title(title, level=1)
        pdf.ln(2)
        # Fit image to page width with margin
        try:
            pdf.image(path, x=10, w=190)
        except Exception as e:
            pdf.body_text(f'[Plot could not be embedded: {e}]')

# --- Footer page with links ---
pdf.add_page()
pdf.section_title('References & Links', level=1)
pdf.ln(3)
pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(0, 100, 180)
pdf.body_text('GitHub Repository:')
pdf.body_text('  https://github.com/Kashishgarg-15/Zomathon')
pdf.ln(2)
pdf.body_text('Jupyter Notebook:')
pdf.body_text('  CSAO_Recommendation_System.ipynb (in repository root)')
pdf.ln(2)
pdf.body_text('Key Files:')
pdf.bullet_item('train_model_v3.py - GBDT training with LLM features + Optuna tuning')
pdf.bullet_item('train_dcn_v2.py - DCN-v2 deep learning model')
pdf.bullet_item('final_ensemble_v2.py - 3-model ensemble')
pdf.bullet_item('generate_llm_features.py - Sentence-transformer feature engineering')
pdf.bullet_item('baseline_and_analysis.py - Baselines + business impact analysis')
pdf.bullet_item('inference_benchmark.py - Production latency benchmark')
pdf.bullet_item('DATA_DICTIONARY.md - Complete data pipeline documentation')

# --- Save ---
OUT = "analysis_output"
pdf_path = os.path.join(OUT, "submission.pdf")
pdf.output(pdf_path)
file_size = os.path.getsize(pdf_path)
print(f"PDF saved to {pdf_path}")
print(f"File size: {file_size:,} bytes ({file_size/1024:.0f} KB)")
print(f"Under 1MB limit: {'YES' if file_size < 1_000_000 else 'NO'}")
print("DONE!")
