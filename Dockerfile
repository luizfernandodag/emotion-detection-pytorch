FROM public.ecr.aws/lambda/python:3.9

# Instalar dependências do sistema
RUN yum install -y \
    mesa-libGL \
    libXext \
    libSM \
    libXrender \
    && yum clean all

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

# Copiar código
COPY . ${LAMBDA_TASK_ROOT}

# Baixar pesos na build (evita cold start)
RUN python scripts/download_weights.py

# Handler da Lambda
CMD ["lambda_function.handler"]
