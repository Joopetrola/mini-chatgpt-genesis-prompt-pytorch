Mini-GPT: projeto didático para treinar um modelo de linguagem pequeno (char-level)
Rodar em CPU/GPU. Inspirado em NanoGPT + implementações didáticas.

Arquivos esperados:
 - data.txt             # corpus de texto (em UTF-8)

Como usar:
 1) Criar um ambiente virtual e instalar dependências:
    python -m venv venv
    source venv/bin/activate  # ou venv\Scripts\activate no Windows
    pip install -U pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu    # se não houver GPU
    pip install tqdm

 2) Treinar:
    python mini_gpt_project.py --mode train --data_path data.txt --epochs 10 --batch_size 64

 3) Gerar texto a partir de um prompt:
    python mini_gpt_project.py --mode generate --checkpoint ckpt.pth --prompt "Olá"

Observações:
 - Este é um modelo char-level (token por caractere) para simplificar.
 - Para melhorar: substituir tokenizer por BPE, aumentar tamanho do modelo, usar GPU,
   ajustar hiperparâmetros e usar técnicas como AdamW / weight decay / lr scheduler.
