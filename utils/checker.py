import os
import re
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from joblib import Parallel, delayed
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from html import escape

def tokenize_code(code, file_ext):
    # Remove comments but preserve code structure
    if file_ext == 'py':
        code = re.sub(r'#.*?\n', '\n', code)
    elif file_ext in ['c', 'cpp', 'java']:
        code = re.sub(r'//.*?\n|/\*.*?\*/', '\n', code, flags=re.DOTALL)
    
    code = code.strip()
    # Tokenize while preserving all whitespace and symbols
    tokens = []
    current_token = ''
    for char in code:
        if char.isalnum() or char == '_':
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            tokens.append(char)
    if current_token:
        tokens.append(current_token)
    return tokens, code

def get_graphcodebert_embedding(code, tokenizer, model, device):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding

def highlight_similar_portions(code1, code2, file_ext):
    tokens1, raw_code1 = tokenize_code(code1, file_ext)
    tokens2, raw_code2 = tokenize_code(code2, file_ext)
    
    matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
    highlighted1, highlighted2 = [], []
    matching_tokens = []
    
    # Highlight identical tokens
    for op, i1, i1_end, i2, i2_end in matcher.get_opcodes():
        if op == 'equal':
            for i in range(i1, i1_end):
                highlighted1.append(f'<font color="red">{escape(tokens1[i])}</font>')
                matching_tokens.append(tokens1[i])
            for i in range(i2, i2_end):
                highlighted2.append(f'<font color="red">{escape(tokens2[i])}</font>')
        else:
            for i in range(i1, i1_end):
                highlighted1.append(escape(tokens1[i]))
            for i in range(i2, i2_end):
                highlighted2.append(escape(tokens2[i]))
    
    # Debug: Log all matching tokens
    print(f"Matching tokens for pair: {matching_tokens}")
    
    # Reconstruct code by joining tokens
    highlighted_code1 = ''.join(highlighted1)
    highlighted_code2 = ''.join(highlighted2)
    
    # Truncate to 50 lines for readability
    lines1 = highlighted_code1.split('\n')[:50]
    lines2 = highlighted_code2.split('\n')[:50]
    
    return '\n'.join(lines1), '\n'.join(lines2)

def compute_similarity_pair(i, j, files, embeddings, file_ext):
    score = cosine_similarity(embeddings[i], embeddings[j])[0][0]
    if score > 0.3:  # 30% threshold
        highlighted1, highlighted2 = highlight_similar_portions(files[i]['content'], files[j]['content'], file_ext)
        return {
            'file1': files[i]['filename'],
            'file2': files[j]['filename'],
            'score': score,
            'highlight': {
                'file1': files[i]['filename'],
                'file2': files[j]['filename'],
                'code1': highlighted1,
                'code2': highlighted2
            }
        }
    return {
        'file1': files[i]['filename'],
        'file2': files[j]['filename'],
        'score': score,
        'highlight': None
    }

def generate_pdf_report(scores, output_file, file_count):
    doc = SimpleDocTemplate(output_file, pagesize=A4, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()
    code_style = ParagraphStyle(name='Code', fontName='Courier', fontSize=8, leading=10, wordWrap='CJK')
    
    elements = []
    
    elements.append(Paragraph("Code Similarity Report for Classroom", styles['Title']))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Summary", styles['Heading2']))
    elements.append(Paragraph(f"Total Files Processed: {file_count}", styles['Normal']))
    elements.append(Paragraph(f"Total Comparisons: {len(scores)}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Similarity Scores (Sorted by Score)", styles['Heading2']))
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    table_data = [['File Pair', 'Similarity Score']]
    for score in sorted_scores:
        table_data.append([f"{score['file1']} vs {score['file2']}", f"{score['score']:.2%}"])
    
    table = Table(table_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))
    
    elements.append(Paragraph("Highlighted Similarities (Above 30%)", styles['Heading2']))
    if not any(score['highlight'] for score in sorted_scores):
        elements.append(Paragraph("No pairs with similarity above 30% found.", styles['Normal']))
    
    for score in sorted_scores:
        if score['highlight']:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(f"{score['highlight']['file1']} vs {score['highlight']['file2']} ({score['score']:.2%})", styles['Heading3']))
            elements.append(Paragraph(f"{score['highlight']['file1']}:", styles['Heading4']))
            elements.append(Paragraph(score['highlight']['code1'], code_style))
            elements.append(Paragraph(f"{score['highlight']['file2']}:", styles['Heading4']))
            elements.append(Paragraph(score['highlight']['code2'], code_style))
            elements.append(Spacer(1, 0.1*inch))
    
    doc.build(elements)

def main():
    parser = argparse.ArgumentParser(description="CodeSim Report")
    parser.add_argument('directory', help="Directory containing student code files")
    parser.add_argument('--file-type', required=True, choices=['py', 'java', 'c', 'cpp'], 
                       help="File type to process (py, java, c, cpp)")
    parser.add_argument('--output', default='classroom_similarity_report.pdf', 
                       help="Output PDF file for similarity report")
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return

    files = []
    for filename in os.listdir(args.directory):
        if filename.endswith(f'.{args.file_type}'):
            with open(os.path.join(args.directory, filename), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                files.append({'filename': filename, 'content': content, 'ext': args.file_type})

    if len(files) < 2:
        print("Error: At least two files are required for comparison")
        return
    if len(files) > 40:
        print(f"Warning: Found {len(files)} files, expected up to 40")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = AutoModel.from_pretrained("microsoft/graphcodebert-base").to(device)

    embeddings = [get_graphcodebert_embedding(file['content'], tokenizer, model, device) for file in files]

    scores = Parallel(n_jobs=-1)(
        delayed(compute_similarity_pair)(i, j, files, embeddings, args.file_type)
        for i in range(len(files))
        for j in range(i + 1, len(files))
    )

    try:
        generate_pdf_report(scores, args.output, len(files))
        print(f"PDF report generated at {args.output}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == '__main__':
    main()