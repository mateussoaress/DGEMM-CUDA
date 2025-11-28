# Implementação DGEMM – CUDA

## Descrição Geral

Este projeto implementa o algoritmo DGEMM (Double-Precision General Matrix Multiplication) em três versões:

* **CPU otimizada** (tiling + register blocking 2×2)
* **CUDA básica** (1 thread por elemento)
* **CUDA com memória compartilhada** (tiling 32×32)

O objetivo é comparar desempenho, verificar corretude numérica e analisar eficiência paralela.

---

## Fórmula DGEMM

C = α·A·B + β·C

A, B e C são matrizes quadradas de ordem **N**.

---

## Estrutura do Código

### 1. dgemm_seq (CPU otimizada)

* Transpõe B para melhorar localidade de memória.
* Aplica tiling (blocos 64×64).
* Utiliza register blocking 2×2.

### 2. dgemm_basic_kernel (CUDA básica)

* Cada thread calcula **um elemento** de C.
* Todo acesso é feito diretamente na memória global.

### 3. dgemm_tiled_kernel (CUDA otimizada)

* Usa memória compartilhada **shared** para carregar submatrizes.
* Tamanho do tile: 32×32 (BLOCK_SIZE).
* Reduz acessos à memória global e melhora throughput.

### 4. Funções CUDA de Host

Responsáveis por:

* Alocação de memória no dispositivo.
* Transferência de dados Host ↔ Device.
* Medição de tempo via *CUDA Events*.
* Execução dos kernels.

### 5. Métricas Calculadas

* Tempo da CPU.
* Tempo da GPU (básica e otimizada).
* Speedup CPU → GPU.
* Eficiência paralela.
* Diferença relativa entre CPU e GPU.

---

## Compilação

```
nvcc -O3 -arch=sm_75 -o dgemm_cuda dgemm_cuda.cu
```

Requisitos:

* CUDA Toolkit
* GPU compatível
* Compilador C/C++ configurado

---

## Execução

```
./dgemm_cuda <n>
```

Exemplo:

```
./dgemm_cuda 1024
```

---

## Fórmulas de Desempenho

### Speedup

```
speedup = T_cpu / T_gpu
```

### Eficiência (modelo didático)

Número de *processadores virtuais* = threads por bloco.

* Versão básica: 16×16 = 256 threads
* Versão shared: 32×32 = 1024 threads

```
eficiência = speedup / P
```

### Diferença Relativa

```
max |C_cpu[i] − C_gpu[i]| / (|C_cpu[i]| + ε)
```

---

## Objetivo Acadêmico

Este trabalho demonstra:

* Como a hierarquia de memória da GPU impacta a performance.
* Diferenças práticas entre kernels simples e otimizados.
* Métodos para validar corretude numérica.
* Como interpretar speedup e eficiência.

---

## Licença

Uso livre para fins acadêmicos e educacionais.
